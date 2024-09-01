import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Add, Activation, TimeDistributed, Bidirectional, LSTM
from tensorflow.keras.layers import Flatten, Conv1D, MaxPooling1D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #This is to force the model to use the CPU
import numpy as np
import scipy.stats


def sleepdetector_cnn_cpu(n_filters = [8, 16, 32], kernel_size = [50, 8, 8], Fs = 100, n_classes = 5):
    input_sig1 = Input(shape=(30*Fs,1))
    input_sig2 = Input(shape=(30*Fs,1))
    input_sig3 = Input(shape=(30*Fs,1))
    input_sig4 = Input(shape=(30*Fs,1))

    def conv_block(input_layer, filters, kernel_size):
        x = Conv1D(filters, kernel_size=kernel_size, strides=1, padding='same', kernel_initializer='glorot_uniform')(input_layer)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(pool_size=8, strides=None)(x)
        return x

    # Apply conv blocks to each input
    x0 = conv_block(input_sig1, n_filters[0], kernel_size[0])
    x0 = conv_block(x0, n_filters[1], kernel_size[1])
    x0 = conv_block(x0, n_filters[2], kernel_size[2])

    x1 = conv_block(input_sig2, n_filters[0], kernel_size[0])
    x1 = conv_block(x1, n_filters[1], kernel_size[1])
    x1 = conv_block(x1, n_filters[2], kernel_size[2])

    x2 = conv_block(input_sig3, n_filters[0], kernel_size[0])
    x2 = conv_block(x2, n_filters[1], kernel_size[1])
    x2 = conv_block(x2, n_filters[2], kernel_size[2])

    x3 = conv_block(input_sig4, n_filters[0], kernel_size[0])
    x3 = conv_block(x3, n_filters[1], kernel_size[1])
    x3 = conv_block(x3, n_filters[2], kernel_size[2])

    merged_vector = tf.keras.layers.concatenate([x0, x1, x2, x3], axis=-1)
    flattened_vector = Flatten()(merged_vector)
    final_x = Dense(n_classes, activation='softmax')(flattened_vector)
    
    model = Model(inputs=[input_sig1, input_sig2, input_sig3, input_sig4], outputs=[final_x])
    
    return model

def sleepdetector_lstm_cpu(timesteps = 32, vec_len = 640, n_units = 64, n_layers = 4):
    x_in = Input(shape = (timesteps, vec_len))
    
    x = Bidirectional(LSTM(units = n_units, return_sequences=True))(x_in)
    
    for _ in range(n_layers - 1):
        x = Bidirectional(LSTM(units = n_units, return_sequences=True))(x)
    
    final_x = TimeDistributed(Dense(5, activation = 'softmax'))(x)   
    
    model = Model(inputs=[x_in], outputs=[final_x])
    
    return model



class Sleepdetector:
    def __init__(self, cnn_path=None, lstm_path=None, seq_length=32):
        self.cnn = sleepdetector_cnn_cpu()
        if cnn_path is not None:
            self.cnn.load_weights(cnn_path)
        self.lstm = sleepdetector_lstm_cpu()
        if lstm_path is not None:
            self.lstm.load_weights(lstm_path)

        # Create a new model that outputs the desired layer
        self.cnn_intermediate = tf.keras.Model(inputs=self.cnn.inputs, outputs=self.cnn.layers[53].output)
        self.get_cnn_output = tf.function(self.cnn_intermediate)
        
        
        # Rest of the init method...
            
        self.seq_length = seq_length
        self.iqr_target = [7.90, 11.37, 7.92, 11.56]
        self.med_target = [0.0257, 0.0942, 0.02157, 0.1055]
        
        # self.get_cnn_output = tf.function(self.cnn)
    
    def get_lstm_output(self, x_cnn):
        c = self.check_input_dimensions_lstm(x_cnn)
        
        if c == -1:
            return None
        
        n_examples = np.shape(x_cnn)[0]
        n_seqs = int(n_examples/self.seq_length)
        last_idx = n_seqs*self.seq_length
        x_lstm = np.reshape(x_cnn[0:last_idx], (-1, self.seq_length, 640))
        y_lstm = self.lstm.predict(x_lstm, verbose = 0)
        
        y_hat = np.argmax(np.reshape(y_lstm, (-1, 5)), axis = -1)
        
        return y_hat
        
    def check_input_dimensions_lstm(self, x_cnn):
        shape_x = np.shape(x_cnn)
        if len(shape_x) != 2:
            print("Input to LSTM must be of dimension 2")
            return -1
        
        if shape_x[0] <= 0:
            print("The first input must be a positive integer")
            return -1
        
        if shape_x[1] != 640:
            print("The second input must have a dimension = 640")
            return -1
        
        return 1
        
    def check_input_dimensions(self, x):
        shape_x = np.shape(x)
        if len(shape_x) != 4:
            print("Input dimensions is different than 4")
            return -1
        
        if shape_x[0] != 4:
            print("First dimension should be equal to 4")
            return -1
        
        if shape_x[1] <= 0:
            print("Second dimension should be a positive integer")
            return -1
        
        if shape_x[2] != 3000:
            print("Third dimension should be equal to 3000")
            return -1
        
        if shape_x[3] != 1:
            print("Final dimension should be equal to 1")
            return -1
        
        return 1
    
    def predict(self, x):
        c = self.check_input_dimensions(x)
        
        if c == -1:
            print("Error in input dimensions")
            return -1
        
        for i in range(4):
            x[i] = self.med_target[i] + (x[i] - np.median(x[i]))*(self.iqr_target[i]/scipy.stats.iqr(x[i]))
        
        x_cnn = self.get_cnn_output([x[0], x[1], x[2], x[3]])
        y_lstm = self.get_lstm_output(x_cnn)
        
        if y_lstm is None:
            return -1
        
        return y_lstm