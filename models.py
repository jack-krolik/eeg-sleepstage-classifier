import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Add, Activation, TimeDistributed, Bidirectional, LSTM
from tensorflow.keras.layers import Flatten, Conv1D, MaxPooling1D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

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