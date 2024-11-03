import os
import re
import logging
import numpy as np
import pandas as pd
import pyedflib
import mne
from scipy import ndimage
from scipy.io import savemat
from datetime import datetime
from pathlib import Path
import warnings
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='preprocessing_batch.log',
    filemode='w'
)

# Constants from original preprocessing.py
BANDPASS_FILTER = (0.5, 50)
DOWNSAMPLE_FREQ = 100
BIPOLAR_CHANNELS = {
    'anode': ['F3-M2', 'C3-M2', 'F4-M1', 'C4-M1'],
    'cathode': ['C3-M2', 'O1-M2', 'C4-M1', 'O2-M1'],
    'names': ['F3-C3', 'C3-O1', 'F4-C4', 'C4-O2']
}
EPOCH_DURATION = 30
ARTIFACT_THRESHOLD = 500
FLAT_THRESHOLD = 1e-6
FLAT_DURATION = 5

# Target values for scaling
IQR_TARGET = np.array([7.90, 11.37, 7.92, 11.56])
MED_TARGET = np.array([0.0257, 0.0942, 0.02157, 0.1055])

def process_single_recording(edf_path, sleep_stages_path, output_dir):
    """
    Process a single EEG recording and its corresponding sleep stages.
    
    Args:
        edf_path: Path to the .edf file
        sleep_stages_path: Path to the sleep stages file
        output_dir: Directory to save the processed data
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        # Load and preprocess EEG data
        raw = mne.io.read_raw_edf(edf_path, preload=True)
        raw._data *= 1e6  # Convert to microvolts
        
        # Apply bandpass filter and downsample
        raw.filter(*BANDPASS_FILTER, fir_design='firwin')
        raw.resample(DOWNSAMPLE_FREQ)
        
        # Create bipolar montage
        bipolar_montage = mne.set_bipolar_reference(
            raw,
            anode=BIPOLAR_CHANNELS['anode'],
            cathode=BIPOLAR_CHANNELS['cathode'],
            ch_name=BIPOLAR_CHANNELS['names']
        )
        raw = bipolar_montage.pick_channels(BIPOLAR_CHANNELS['names'])
        
        # Create epochs
        epochs = mne.make_fixed_length_epochs(raw, duration=EPOCH_DURATION, preload=True)
        
        # Initialize arrays to track artifacts
        num_epochs = epochs._data.shape[0]
        amplitude_artifact_counts = np.zeros(num_epochs)
        flat_artifact_counts = np.zeros(num_epochs)
        epochs_to_remove = []
        
        # Process epochs and detect artifacts
        for i in range(num_epochs):
            for j in range(epochs._data.shape[1]):
                data_epoch = epochs._data[i, j, :]
                
                # Detect artifacts
                amplitude_mask = np.abs(data_epoch) > ARTIFACT_THRESHOLD
                diff = np.abs(np.diff(data_epoch))
                flat_mask = np.zeros_like(data_epoch, dtype=bool)
                flat_mask[:-1] = ndimage.uniform_filter1d(
                    diff < FLAT_THRESHOLD, 
                    size=int(FLAT_DURATION * raw.info['sfreq'])
                ) > 0.99
                
                artifact_mask = amplitude_mask | flat_mask
                
                # Count artifacts
                amplitude_artifact_counts[i] += np.sum(amplitude_mask)
                flat_artifact_counts[i] += np.sum(flat_mask)
                
                # Mark epoch for removal if too many artifacts
                if (np.sum(amplitude_mask) >= 5 * raw.info['sfreq']) or \
                   (np.sum(flat_mask) >= 5 * raw.info['sfreq']):
                    epochs_to_remove.append(i)
                    break
                
                # Handle artifacts
                data_epoch[artifact_mask] = 0
                epochs._data[i, j, :] = data_epoch
        
        # Remove marked epochs
        epochs.drop(epochs_to_remove)
        
        # Scale each channel
        for i, ch in enumerate(raw.ch_names):
            data = epochs.get_data(picks=ch).reshape(-1)
            median = np.median(data)
            iqr = np.percentile(data, 75) - np.percentile(data, 25)
            scaled_data = (data - median) / iqr * IQR_TARGET[i] + MED_TARGET[i]
            epochs._data[:, epochs.ch_names.index(ch), :] = scaled_data.reshape(epochs._data.shape[0], -1)
        
        # Process sleep stages
        stage_mapping = {7: 4, 5: 3, 1: 2, 2: 1, 3: 0, 0: -1}
        
        if sleep_stages_path.endswith('.xlsx'):
            df = pd.read_excel(sleep_stages_path, skiprows=1)
            sleep_stages = df.iloc[:, 0].values
        else:
            with open(sleep_stages_path, 'r') as file:
                lines = [line.strip() for line in file.readlines()[2:] if line.strip()]
            sleep_stages = np.array([int(line.split()[0]) for line in lines])
        
        # Find stage 7 boundaries
        start_index = np.argmax(sleep_stages == 7)
        end_index = len(sleep_stages) - np.argmax(sleep_stages[::-1] == 7)
        
        # Expand stage 7 range
        sleep_stages[max(0, start_index - 50):start_index] = 7
        sleep_stages[end_index:min(len(sleep_stages), end_index + 50)] = 7
        
        # Map stages and remove epochs
        renumbered_stages = np.vectorize(stage_mapping.get)(sleep_stages)
        labels = np.delete(renumbered_stages, list(epochs_to_remove), axis=0)
        
        # Prepare and save data
        preprocessed_data = epochs.get_data()
        
        # Generate output filename
        subject_id = re.search(r'(\d+)\s*N[12]', edf_path).group(1)
        night = 'N1' if 'N1' in edf_path else 'N2'
        output_filename = f'preprocessed_data_{subject_id}_{night}.mat'
        output_path = os.path.join(output_dir, output_filename)
        
        # Save preprocessed data
        savemat(output_path, {
            'sig1': preprocessed_data[:, 0, :],
            'sig2': preprocessed_data[:, 1, :],
            'sig3': preprocessed_data[:, 2, :],
            'sig4': preprocessed_data[:, 3, :],
            'Fs': raw.info['sfreq'],
            'ch_names': raw.ch_names,
            'labels': labels.reshape(1, -1)
        })
        
        return True
        
    except Exception as e:
        logging.error(f"Error processing {edf_path}: {str(e)}")
        return False

def batch_process_recordings(data_dir, output_dir, max_subjects=50):
    """
    Process multiple EEG recordings in batch.
    
    Args:
        data_dir: Root directory containing subject folders
        output_dir: Directory to save processed data
        max_subjects: Maximum number of subjects to process
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all subject directories
    subject_dirs = []
    for entry in os.scandir(os.path.join(data_dir, "Scored 1")):
        if entry.is_dir() and re.match(r'\d+\s*N[12]$', entry.name):
            subject_dirs.append(entry.path)
    
    # Sort directories by subject number
    subject_dirs.sort(key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group(1)))
    
    # Process only up to max_subjects
    subject_dirs = subject_dirs[:max_subjects]
    
    # Process each subject
    successful = 0
    failed = 0
    
    progress_bar = tqdm(subject_dirs, desc="Processing subjects")
    for subject_dir in progress_bar:
        try:
            # Find EDF and sleep stages files
            edf_file = next((f for f in os.listdir(subject_dir) if f.endswith('.edf')), None)
            sleep_stages_file = next((f for f in os.listdir(subject_dir) 
                                   if f.startswith('SASE Sleep Stages') and f.endswith('.xlsx')), None)
            
            if edf_file and sleep_stages_file:
                edf_path = os.path.join(subject_dir, edf_file)
                sleep_stages_path = os.path.join(subject_dir, sleep_stages_file)
                
                if process_single_recording(edf_path, sleep_stages_path, output_dir):
                    successful += 1
                else:
                    failed += 1
                    
            progress_bar.set_postfix({
                'successful': successful,
                'failed': failed
            })
            
        except Exception as e:
            logging.error(f"Error processing directory {subject_dir}: {str(e)}")
            failed += 1
            continue
    
    # Log final statistics
    logging.info(f"\nProcessing completed:")
    logging.info(f"Successfully processed: {successful}")
    logging.info(f"Failed: {failed}")
    logging.info(f"Total attempted: {successful + failed}")

if __name__ == "__main__":
    # Suppress warnings
    warnings.filterwarnings("ignore")
    
    # Define paths
    training_data_dir = "../../trainingSleepData"  # Adjust path as needed
    output_dir = "./preprocessed_data"
    
    # Run batch processing
    batch_process_recordings(training_data_dir, output_dir, max_subjects=50)
