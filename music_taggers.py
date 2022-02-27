# -*- coding: utf-8 -*-
'''MusicTaggerCNN model for Keras.
# Reference:
- [Automatic tagging using deep convolutional neural networks](https://arxiv.org/abs/1606.00298)
- [Music-auto_tagging-keras](https://github.com/keunwoochoi/music-auto_tagging-keras)
'''
from __future__ import print_function
from __future__ import absolute_import

from keras import backend as K
from keras.models import Model
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.layers.recurrent import GRU
from keras.utils.data_utils import get_file
from keras.layers import (Conv1D, MaxPool1D, BatchNormalization,
                          Dense, Dropout, Activation, Flatten, Reshape, Permute, Input, Embedding, Multiply)


def MusicTaggerLR(AUDIO_FEATURE_NUM, NUM_CLASS, PROPAGATION=False):
    audio_input = Input(shape=(AUDIO_FEATURE_NUM,),
                        dtype='float32', name='audio_input')
    # Model(audio_input,audio_input).predict(np.ones((10,1000)))
    predict = Dense(NUM_CLASS, activation='sigmoid',
                    name='predict')(audio_input)
    if PROPAGATION:
        # Model(audio_input,predict).predict(np.ones((10,1000)))
        # should extend by batch size
        fixed_input = Input(tensor=K.constant([0]), name='constant')
        tag_variables = Embedding(
            1, NUM_CLASS, embeddings_initializer='uniform', name='tag_variable')(fixed_input)
        propagation_probability = Activation(
            'sigmoid', name='tag_probability')(tag_variables)
        neighbor_predict = Multiply(name='neighbor_predict_layer')([
            predict, propagation_probability])
        model = Model(inputs=[audio_input, fixed_input],
                      outputs=[predict, neighbor_predict,propagation_probability])
        #propagation_probability_model = Model(input=[fixed_input,audio_input], output=[propagation_probability])
    else:
        model = Model(inputs=[audio_input], outputs=[predict])
    '''
    TODO: 
    1.1 how to ignore the fixed_input? when input tuple , (inputs, outputs) , only one input ! 
    1.2 how to train and what optimization to use ? https://keras.io/models/sequential/ 
    "see : fit_generator" 
    "see compile"
    loss: String (name of objective function) or objective function. See losses. If the model has multiple outputs, you can use a different loss on each output by passing a dictionary or a list of losses. The loss value that will be minimized by the model will then be the sum of all individual losses.

    2. how to ignore neighbor_predict at validation? choose left result ! 
    3. 
    '''
    return model


def MusicTaggerSampleCNN(SEGMENT_LEN, NUM_CLASS, PROPAGATION=False):
    dropout_rate = 0.5
    kernel_initializer = 'he_uniform'
    activation = 'relu'
    audio_input = Input(shape=(SEGMENT_LEN,), dtype='float32')
    net = Reshape([-1, 1])(audio_input)
    # 59049 X 1
    net = Conv1D(128, 3, strides=3, padding='valid',
                 kernel_initializer=kernel_initializer)(net)
    net = BatchNormalization()(net)
    net = Activation(activation)(net)
    # 19683 X 128
    net = Conv1D(128, 3, padding='same',
                 kernel_initializer=kernel_initializer)(net)
    net = BatchNormalization()(net)
    net = Activation(activation)(net)
    net = MaxPool1D(3)(net)
    # 6561 X 128
    net = Conv1D(128, 3, padding='same',
                 kernel_initializer=kernel_initializer)(net)
    net = BatchNormalization()(net)
    net = Activation(activation)(net)
    net = MaxPool1D(3)(net)
    # 2187 X 128
    net = Conv1D(256, 3, padding='same',
                 kernel_initializer=kernel_initializer)(net)
    net = BatchNormalization()(net)
    net = Activation(activation)(net)
    net = MaxPool1D(3)(net)
    # 729 X 256
    net = Conv1D(256, 3, padding='same',
                 kernel_initializer=kernel_initializer)(net)
    net = BatchNormalization()(net)
    net = Activation(activation)(net)
    net = MaxPool1D(3)(net)
    # 243 X 256
    net = Conv1D(256, 3, padding='same',
                 kernel_initializer=kernel_initializer)(net)
    net = BatchNormalization()(net)
    net = Activation(activation)(net)
    net = MaxPool1D(3)(net)
    # 81 X 256
    net = Conv1D(256, 3, padding='same',
                 kernel_initializer=kernel_initializer)(net)
    net = BatchNormalization()(net)
    net = Activation(activation)(net)
    net = MaxPool1D(3)(net)
    # 27 X 256
    net = Conv1D(256, 3, padding='same',
                 kernel_initializer=kernel_initializer)(net)
    net = BatchNormalization()(net)
    net = Activation(activation)(net)
    net = MaxPool1D(3)(net)
    # 9 X 256
    net = Conv1D(256, 3, padding='same',
                 kernel_initializer=kernel_initializer)(net)
    net = BatchNormalization()(net)
    net = Activation(activation)(net)
    net = MaxPool1D(3)(net)
    # 3 X 256
    net = Conv1D(512, 3, padding='same',
                 kernel_initializer=kernel_initializer)(net)
    net = BatchNormalization()(net)
    net = Activation(activation)(net)
    net = MaxPool1D(3)(net)
    # 1 X 512
    net = Conv1D(512, 1, padding='same',
                 kernel_initializer=kernel_initializer)(net)
    net = BatchNormalization()(net)
    net = Activation(activation)(net)
    # 1 X 512
    net = Dropout(dropout_rate)(net)
    net = Flatten()(net)

    predict = Dense(units=NUM_CLASS, activation='sigmoid', name = 'predict')(net)
    if PROPAGATION:
        # Model(audio_input,predict).predict(np.ones((10,1000)))
        # should extend by batch size
        fixed_input = Input(tensor=K.constant([0]), name='constant')
        # Model([audio_input,fixed_input],fixed_input).predict(np.ones((10,1000)))
        tag_variables = Embedding(
            1, NUM_CLASS, embeddings_initializer='uniform', name='tag_variable')(fixed_input)
        # Model(inputs = [audio_input,fixed_input],outputs=tag_variables).predict(np.ones((10,1000)))
        propagation_probability = Activation(
            'sigmoid', name='tag_probability')(tag_variables)
        # Model(inputs = [audio_input,fixed_input],outputs=propagation_probability).predict(np.ones((10,1000)))
        neighbor_predict = Multiply(name='neighbor_predict_layer')([
            predict, propagation_probability])
        model = Model(inputs=[audio_input, fixed_input],
                      outputs=[predict, neighbor_predict,propagation_probability])
        # model.predict(np.ones((10,1000)))
    else:
        model = Model(inputs=audio_input, outputs=predict)
    return model


def MusicTaggerCNN(NUM_CLASS, PROPAGATION=False, weights='msd', input_tensor=None):
    '''Instantiate the MusicTaggerCNN architecture,
    optionally loading weights pre-trained
    on Million Song Dataset. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.
    For preparing mel-spectrogram input, see
    `audio_conv_utils.py` in [applications](https://github.com/fchollet/keras/tree/master/keras/applications).
    You will need to install [Librosa](http://librosa.github.io/librosa/)
    to use it.
    # Arguments
        weights: one of `None` (random initialization)
            or "msd" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        include_top: whether to include the 1 fully-connected
            layer (output layer) at the top of the network.
            If False, the network outputs 256-dim features.
    # Returns
        A Keras model instance.
    '''
    if weights not in {'msd', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `msd` '
                         '(pre-training on Million Song Dataset).')

    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (1, 96, 1366)
    else:
        input_shape = (96, 1366, 1)

    if input_tensor is None:
        melgram_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            melgram_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            melgram_input = input_tensor
    # Determine input axis
    if K.image_dim_ordering() == 'th':
        channel_axis = 1
        freq_axis = 2
        time_axis = 3
    else:
        channel_axis = 3
        freq_axis = 1
        time_axis = 2

    # Input block
    x = BatchNormalization(axis=freq_axis, name='bn_0_freq')(melgram_input)

    # Conv block 1
    x = Convolution2D(64, 3, 3, border_mode='same', name='conv1')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn1')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 4), name='pool1')(x)

    # Conv block 2
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv2')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn2')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 4), name='pool2')(x)

    # Conv block 3
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv3')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn3')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 4), name='pool3')(x)

    # Conv block 4
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv4')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn4')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(3, 5), name='pool4')(x)

    # Conv block 5
    x = Convolution2D(64, 3, 3, border_mode='same', name='conv5')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn5')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), name='pool5')(x)

    # Output
    x = Flatten()(x)
    # if include_top:
    predict = Dense(NUM_CLASS, activation='sigmoid', name='predict')(x)

    # Create model
    if PROPAGATION:
        # should extend by batch size
        fixed_input = Input(tensor=K.constant([0]), name='constant')
        tag_variables = Embedding(
            1, NUM_CLASS, embeddings_initializer='uniform', name='tag_variable')(fixed_input)
        propagation_probability = Activation(
            'sigmoid', name='tag_probability')(tag_variables)
        neighbor_predict = Multiply(name='neighbor_predict_layer')([
            predict, propagation_probability])
        model = Model(inputs=[melgram_input, fixed_input],
                      outputs=[predict, neighbor_predict,propagation_probability])
    else:
        model = Model(inputs=melgram_input, outputs=predict)
    return model
    '''
    if weights is None:
        return model    
    else: 
        # Load input
        if K.image_dim_ordering() == 'tf':
            raise RuntimeError("Please set image_dim_ordering == 'th'."
                               "You can set it at ~/.keras/keras.json")
        model.load_weights('data/music_tagger_cnn_weights_%s.h5' % K._BACKEND,
                           by_name=True)
        return model
        
    '''


def MusicTaggerCRNN(NUM_CLASS, PROPAGATION=False, weights='msd', input_tensor=None):
    '''Instantiate the MusicTaggerCRNN architecture,
    optionally loading weights pre-trained
    on Million Song Dataset. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.
    For preparing mel-spectrogram input, see
    `audio_conv_utils.py` in [applications](https://github.com/fchollet/keras/tree/master/keras/applications).
    You will need to install [Librosa](http://librosa.github.io/librosa/)
    to use it.
    # Arguments
        weights: one of `None` (random initialization)
            or "msd" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        include_top: whether to include the 1 fully-connected
            layer (output layer) at the top of the network.
            If False, the network outputs 32-dim features.
    # Returns
        A Keras model instance.
    '''
    if weights not in {'msd', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `msd` '
                         '(pre-training on Million Song Dataset).')

    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (1, 96, 1366)
    else:
        input_shape = (96, 1366, 1)

    if input_tensor is None:
        melgram_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            melgram_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            melgram_input = input_tensor

    # Determine input axis
    if K.image_dim_ordering() == 'th':
        channel_axis = 1
        freq_axis = 2
        time_axis = 3
    else:
        channel_axis = 3
        freq_axis = 1
        time_axis = 2

    # Input block
    x = ZeroPadding2D(padding=(0, 37))(melgram_input)
    x = BatchNormalization(axis=freq_axis, name='bn_0_freq')(x)

    # Conv block 1
    x = Convolution2D(64, 3, 3, border_mode='same', name='conv1')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn1')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)
    x = Dropout(0.1, name='dropout1')(x)

    # Conv block 2
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv2')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn2')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), name='pool2')(x)
    x = Dropout(0.1, name='dropout2')(x)

    # Conv block 3
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv3')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn3')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool3')(x)
    x = Dropout(0.1, name='dropout3')(x)

    # Conv block 4
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv4')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn4')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool4')(x)
    x = Dropout(0.1, name='dropout4')(x)

    # reshaping
    if K.image_dim_ordering() == 'th':
        x = Permute((3, 1, 2))(x)
    x = Reshape((15, 128))(x)

    # GRU block 1, 2, output
    x = GRU(32, return_sequences=True, name='gru1')(x)
    x = GRU(32, return_sequences=False, name='gru2')(x)
    x = Dropout(0.3)(x)
    # if include_top:
    predict = Dense(NUM_CLASS, activation='sigmoid', name='predict')(x)

    # Create model
    if PROPAGATION:
        # Model(audio_input,predict).predict(np.ones((10,1000)))
        # should extend by batch size
        fixed_input = Input(tensor=K.constant([0]), name='constant')
        # Model([audio_input,fixed_input],fixed_input).predict(np.ones((10,1000)))
        tag_variables = Embedding(
            1, NUM_CLASS, embeddings_initializer='uniform', name='tag_variable')(fixed_input)
        # Model(inputs = [audio_input,fixed_input],outputs=tag_variables).predict(np.ones((10,1000)))
        propagation_probability = Activation(
            'sigmoid', name='tag_probability')(tag_variables)
        # Model(inputs = [audio_input,fixed_input],outputs=propagation_probability).predict(np.ones((10,1000)))
        neighbor_predict = Multiply(name='neighbor_predict_layer')([
            predict, propagation_probability])
        model = Model(inputs=[melgram_input, fixed_input],
                      outputs=[predict, neighbor_predict,propagation_probability])
        # model.predict(np.ones((10,1000)))
    else:
        model = Model(inputs=melgram_input, outputs=predict)
    return model
    '''
    if weights is None:
        return model
    else: 
        # Load input
        if K.image_dim_ordering() == 'tf':
            raise RuntimeError("Please set image_dim_ordering == 'th'."
                               "You can set it at ~/.keras/keras.json")
    
        model.load_weights('data/music_tagger_crnn_weights_%s.h5' % K._BACKEND,
                           by_name=True)
        return model
    '''
