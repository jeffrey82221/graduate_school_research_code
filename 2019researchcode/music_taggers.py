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
from keras.layers.core import Lambda
from keras.utils.data_utils import get_file
from keras.layers import (Conv1D, MaxPool1D, BatchNormalization,
                          Layer,Dense, Dropout, Activation, Flatten, Reshape, Permute,Reshape, Input, Embedding, Multiply, Subtract)
from keras import initializers,regularizers
#from Custom_Keras_Layers import ResizeSignal
from keras.layers.pooling import GlobalAveragePooling2D
from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D
#from tensorflow import image as tfi
import numpy as np
'''
def FixWeightMultiTaskLayer(context_count = None):
    '''
    input: alpha=one , the ratio of auxilirary task to target task 
    '''
    def apply(target_predict,auxiliary_predict):
        if type(context_count)!=type(None):
            outputs = [target_predict]+[auxiliary_predict]*context_count
        else:
            outputs = [target_predict,auxiliary_predict]
        return outputs 
    return apply '''
'''def ValidTunedOutputAttentionLayer(NUM_CLASS, alpha = 1.0):
    # Notes: 
    # 1. Now, the inputs are target_attention, and auxiliary_attention 
    # 2. These two inputs vectors should add up to one during tunning in the Evaluation.py 
    # 3. Need to alter input constant during training 
    K.set_value(target_attention,np.ones((1,NUM_CLASS))*0.21) # 
    K.get_value(target_attention)
    target_attention = Input(tensor=K.variable([[(1./(1.+alpha))]*NUM_CLASS])) # shape
    auxiliary_attention = Input(tensor=K.variable([[(alpha/(1.+alpha))]*NUM_CLASS]))
    additional_inputs = target_attention,auxiliary_attention
    additional_outputs = additional_inputs
    return additional_inputs, additional_outputs
def OutputAttentionLayer(NUM_CLASS, alpha = 1.0):
    lock = Input(tensor=K.constant([0]))
    v = Embedding(1, NUM_CLASS, embeddings_initializer=initializers.Constant(np.log(alpha)))(lock) # initialize embedding using np.log(alpha)
    auxiliary_attention = Activation('sigmoid')(v)
    ones = Input(tensor=K.constant([[1.0]*NUM_CLASS]))
    target_attention = Subtract()([ones, auxiliary_attention])
    constant_inputs = [lock,ones]
    additional_outputs = target_attention,auxiliary_attention
    return constant_inputs,additional_outputs'''
def InterTaskAttentionWeights(NUM_CLASS,TRAIN_TUNED,alpha=1.0):
    if not TRAIN_TUNED:
        target_attention = Input(tensor=K.variable([[(1./(1.+alpha))]*NUM_CLASS]))
        auxiliary_attention = Input(tensor=K.variable([[(alpha/(1.+alpha))]*NUM_CLASS])) 
        Input_Holder = {
            "Target":target_attention,
            "Auxiliary":auxiliary_attention
        }
        Output_Holder = Input_Holder
    else:
        lock = Input(tensor=K.constant([0]))
        v = Embedding(1, NUM_CLASS, embeddings_initializer=initializers.Constant(np.log(alpha)))(lock) # initialize embedding using np.log(alpha)
        auxiliary_attention = Activation('sigmoid')(v)
        ones = Input(tensor=K.constant([[1.0]*NUM_CLASS]))
        target_attention = Subtract()([ones, auxiliary_attention])
        Input_Holder = {
            "Lock":lock,"Ones":ones
        }
        Output_Holder = {
            "Target":target_attention,
            "Auxiliary":auxiliary_attention
        }
    return Input_Holder,Output_Holder
def InterClassAttentionWeights(NUM_CLASS,TRAIN_TUNED):
    if not TRAIN_TUNED:
        existence_attention = Input(tensor=K.variable([[1.]*NUM_CLASS]))
        nonexistence_attention = Input(tensor=K.variable([[1.]*NUM_CLASS])) 
        Input_Holder = {
            "Existence":existence_attention,
            "NonExistence":nonexistence_attention
        }
        Output_Holder = Input_Holder
    else:
        lock = Input(tensor=K.constant([0]))
        v = Embedding(1, NUM_CLASS, embeddings_initializer=initializers.Constant(np.log(1.0)))(lock) # initialize embedding using np.log(alpha)
        existence_attention = Activation('sigmoid')(v)
        ones = Input(tensor=K.constant([[1.0]*NUM_CLASS]))
        nonexistence_attention = Subtract()([ones, existence_attention])
        existence_attention = Lambda(lambda x: x * 2.)(existence_attention)
        nonexistence_attention = Lambda(lambda x: x * 2.)(nonexistence_attention)
        Input_Holder = {
            "Lock":lock,"Ones":ones
        }
        Output_Holder = {
            "Target":existence_attention,
            "Auxiliary":nonexistence_attention
        }
    return Input_Holder,Output_Holder
def InterAttributeAttentionWeights(NUM_CLASS,TRAIN_TUNED):
    if not TRAIN_TUNED:
        tag_weights = Input(tensor=K.variable([[1.]*NUM_CLASS]))
        Input_Holder = {
            "TAGWISE_WEIGHT":tag_weights
        }
        Output_Holder = Input_Holder
    else:
        lock = Input(tensor=K.constant([0]))
        v =Embedding(1,NUM_CLASS,embeddings_initializer=initializers.Constant(0.),activity_regularizer=regularizers.l2(1./NUM_CLASS))(lock)
        tag_weights = Lambda(lambda x: x * float(NUM_CLASS))(Activation('softmax')(v))
        Input_Holder = {
            "Lock":lock
        }
        Output_Holder = {
            "TAGWISE_WEIGHT":tag_weights
        }
    return Input_Holder,Output_Holder

'''def TagWiseWeightingLayer(NUM_CLASS,mode="train"): 
    if mode=="valid":
        target_tag_weights = Input(tensor=K.variable([[1.]*NUM_CLASS])) 
        auxiliary_tag_weights = Input(tensor=K.variable([[1.]*NUM_CLASS])) 
        additional_inputs = target_tag_weights,auxiliary_tag_weights
        additional_outputs = additional_inputs
        return additional_inputs,additional_outputs
    elif mode=='train':
        lock = Input(tensor=K.constant([0]))
        target_v =Embedding(1,NUM_CLASS,embeddings_initializer=initializers.Constant(0.),activity_regularizer=regularizers.l2(0.001))(lock)
        target_tag_weights = Lambda(lambda x: x * float(NUM_CLASS))(Activation('softmax')(target_v))
        auxiliary_v=Embedding(1,NUM_CLASS,embeddings_initializer=initializers.Constant(0.),activity_regularizer=regularizers.l2(0.001))(lock)
        auxiliary_tag_weights = Lambda(lambda x: x * float(NUM_CLASS))(Activation('softmax')(auxiliary_v))
        constant_inputs = [lock]
        additional_outputs = target_tag_weights,auxiliary_tag_weights
        return constant_inputs,additional_outputs'''
def LayerWiseAttentionLayer(block_index):
    lock = Input(tensor=K.constant([0]), name='locker'+str(block_index))
    v = Embedding(1, 1, embeddings_initializer='zeros', name='v'+str(block_index))(lock)
    auxiliary_attention = Activation('sigmoid')(v)
    one = Input(tensor=K.constant([1.0]), name='one'+str(block_index)) 
    target_attention = Subtract()([one, auxiliary_attention])
    constant_inputs = [lock,one] 
    def apply_function(target_net,auxiliary_net):
        target_net = Multiply(name='target_net'+str(block_index))(
                [target_net, target_attention])
        auxiliary_net = Multiply(name='auxiliary_predict'+str(block_index))(
                [auxiliary_net, auxiliary_attention])
        return constant_inputs,target_net,auxiliary_net,auxiliary_attention
    return apply_function


def ConnectNetwork(blocks):
    def apply(net):
        for block in blocks: 
            try:
                net = block.apply(net)
            except:
                net = block(net)
        return net 
    return apply 
def ConnectAttentionNetwork(blocks):
    def apply(target_net, auxiliary_net):
        # 1. target_x , auxiliary_x = attention_weighted(target_x,auxiliary_x) 
        # 2. target_x = block.apply(target_x); auxiliary_x = block.apply(auxiliary_x); 
        # 3. 
        constant_inputs_list = []
        auxilirary_attention_list = []
        for i,block in enumerate(blocks): 
            constant_inputs, target_net, auxiliary_net, auxilirary_attention = LayerWiseAttentionLayer(i)(target_net, auxiliary_net)
            constant_inputs_list.extend(constant_inputs)
            auxilirary_attention_list.append(auxilirary_attention)
            try:
                target_net = block.apply(target_net)
                auxiliary_net = block.apply(auxiliary_net)
            except:
                target_net = block(target_net)
                auxiliary_net = block(auxiliary_net)
        return constant_inputs_list, target_net, auxiliary_net,auxilirary_attention_list
    return apply
# output attention for monitoring result:  
# ConnectAttentionNetwork 
# LayerWiseAttentionLayer
# OutputAttentionLayer 
# adding monitoring attention into a list, where the last output is the final attention layer ! 
def ConfigureNetwork(ex_parameters,context_count,hidden_blocks,hidden_input,audio_input):
    NUM_CLASS = ex_parameters['NUM_CLASS'] 
    MULTI_OUTPUT = ex_parameters['MULTI_OUTPUT']
    SPLIT = ex_parameters['SPLIT'] 
    ATTENTION_SCHEME = ex_parameters['ATTENTION_SCHEME']
    LAYER_WISE_ATTENTION = ex_parameters['LAYER_WISE_ATTENTION'] 
    alpha = ex_parameters['alpha'] 
    '''
    NUM_CLASS,MULTI_OUTPUT,SPLIT,TAG_WISE_WEIGHTING,VALID_TAG_WISE_WEIGHTING,OUTPUT_ATTENTION,BINARY_DEPENDENCE,VALID_TUNED,LAYER_WISE_ATTENTION,alpha
    '''
    Input_Dict = {}
    Output_Dict = {}
    # Network Connecting : from audio_input to predict (and predict_neighbor)
    if MULTI_OUTPUT:
        if SPLIT: # using two last layers for each output 
            if LAYER_WISE_ATTENTION:
                additional_inputs_layer_wise,predict,predict_neighbor,attention_list = ConnectAttentionNetwork(hidden_blocks[:-1])(hidden_input,hidden_input)
                # 
                constant_inputs,predict,predict_neighbor,auxilirary_attention = LayerWiseAttentionLayer(len(hidden_blocks[:-1]))(predict, predict_neighbor)
                additional_inputs_layer_wise.extend(constant_inputs)
                attention_list.append(auxilirary_attention)
                predict = Dense(units=NUM_CLASS, activation='sigmoid')(predict) 
                predict_neighbor = Dense(units=NUM_CLASS, activation='sigmoid')(predict_neighbor) 
                Input_Dict["LayerWise"] = additional_inputs_layer_wise
                Output_Dict["LayerWise"] = attention_list
            else:
                predict = ConnectNetwork(hidden_blocks)(hidden_input)
                predict_neighbor = Dense(units=NUM_CLASS, activation='sigmoid')(ConnectNetwork(hidden_blocks[:-1])(hidden_input))
                
        else:
            if LAYER_WISE_ATTENTION:
                additional_inputs_layer_wise,predict,predict_neighbor,attention_list = ConnectAttentionNetwork(hidden_blocks)(hidden_input,hidden_input)
                Input_Dict["LayerWise"] = additional_inputs_layer_wise
                Output_Dict["LayerWise"] = attention_list
            else:
                predict = ConnectNetwork(hidden_blocks)(hidden_input)
                predict_neighbor = predict 
        outputs = [predict,predict_neighbor]
    else:
        predict = ConnectNetwork(hidden_blocks)(hidden_input)
        outputs = [predict]
        # no predict_neighbor this time 
    # generate additional inputs 
    def GenerateInputOutputHolders(MODE,TurnOn,ValidTuned):
        if MODE == "InterClass":
            inputs, outputs = InterClassAttentionWeights(NUM_CLASS,(TurnOn and not ValidTuned))
        elif MODE == "InterAttribute":
            inputs, outputs = InterAttributeAttentionWeights(NUM_CLASS,(TurnOn and not ValidTuned))
        elif MODE == "InterTask":
            inputs, outputs = InterTaskAttentionWeights(NUM_CLASS,(TurnOn and not ValidTuned),alpha)
        return inputs,outputs
    def GenerateDualInputOutputHolders(MODE,TurnOn,ValidTuned):
        Input_Holders = {}
        Output_Holders = {}
        if MODE == "InterClass" or MODE == "InterAttribute":
            inputs,outputs = GenerateInputOutputHolders(MODE,TurnOn,ValidTuned)
            


    if ATTENTION_SCHEME["InterClass"][0]:
        inputs, outputs = InterClassAttentionWeights(NUM_CLASS,ATTENTION_SCHEME["InterClass"][1]["VALID_TUNED"])
        Input_Dict["InterClass"] 
        Output_Dict["InterClass"]
    if ATTENTION_SCHEME["InterAttribute"][0]:
    if ATTENTION_SCHEME["InterTask"][0]:
        InterTaskAttentionWeights(NUM_CLASS,)
    


    model = Model(inputs=audio_input, outputs=predict)
    return model
    '''if MULTI_OUTPUT:
                    # 1) InterClass 
                            
                    if TAG_WISE_WEIGHTING:
                        if VALID_TAG_WISE_WEIGHTING:
                            additional_inputs, additional_outputs = TagWiseWeightingLayer(NUM_CLASS,mode='valid')
                        else:
                            additional_inputs, additional_outputs = TagWiseWeightingLayer(NUM_CLASS,mode='train')
                    else:
                        additional_inputs, additional_outputs = TagWiseWeightingLayer(NUM_CLASS,mode='valid')#all weight equals one and fixed 
                    additional_inputs_layer_wise.extend(additional_inputs)
                    attention_list.extend(additional_outputs)
                    if OUTPUT_ATTENTION:
                        outputs = [predict,predict_neighbor]
                        if VALID_TUNED:
                            output_attention_layer = ValidTunedOutputAttentionLayer
                        else:
                            output_attention_layer = OutputAttentionLayer
                        #if BINARY_DEPENDENCE== True:
                        # [-5] -> auxiliary tag-wise weight 
                        # [-6] -> target tag-wise weight 
                        #else:
                        # [-3] -> auxiliary tag-wise weight 
                        # [-4] -> target tag-wise weight 
                        if BINARY_DEPENDENCE:
                            additional_inputs, additional_outputs = output_attention_layer(NUM_CLASS,alpha=alpha)
                            additional_inputs_layer_wise.extend(additional_inputs)
                            attention_list.extend(additional_outputs)
                            additional_inputs, additional_outputs = output_attention_layer(NUM_CLASS,alpha=alpha)
                            additional_inputs_layer_wise.extend(additional_inputs)
                            attention_list.extend(additional_outputs)
                            # inputs, outputs[-1] -> auxiliary attention 1 (for exist tag)
                            # inputs, outputs[-2] -> target attention 1    (for exist tag)
                            # inputs, outputs[-3] -> auxiliary attention 2 (for non-exist tag)
                            # inputs, outputs[-4] -> target attention 2    (for non-exist tag)
                        else:
                            additional_inputs, additional_outputs = output_attention_layer(NUM_CLASS,alpha=alpha)
                            additional_inputs_layer_wise.extend(additional_inputs)
                            attention_list.extend(additional_outputs)
                            # inputs, outputs[-1] -> auxiliary attention 
                            # inputs, outputs[-2] -> target attention 
                    else:
                       #outputs = FixWeightMultiTaskLayer(context_count = context_count)(predict,predict_neighbor)
                    model = Model(inputs=[audio_input]+additional_inputs_layer_wise, outputs=outputs+attention_list)
                    print("input output built:",
                            additional_inputs_layer_wise,attention_list) 
                else:
                    model = Model(inputs=audio_input, outputs=predict)
                return model '''
def MusicTaggerLR(AUDIO_FEATURE_NUM, ex_parameters,context_count = None):
    NUM_CLASS = ex_parameters["NUM_CLASS"]
    MULTI_OUTPUT = ex_parameters["MULTI_OUTPUT"]
    SPLIT = ex_parameters["SPLIT"]
    alpha = ex_parameters["alpha"]
    audio_input = Input(shape=(AUDIO_FEATURE_NUM,),
                        dtype='float32', name='audio_input')
    # Model(audio_input,audio_input).predict(np.ones((10,1000)))
    predict = Dense(NUM_CLASS, activation='sigmoid',
                    name='predict')(audio_input)
    if MULTI_OUTPUT:
        outputs = FixWeightMultiTaskLayer(alpha = alpha , context_count = context_count)(predict,predict_neighbor)
        model = Model(inputs=[audio_input], outputs=outputs)
    else:
        model = Model(inputs=[audio_input], outputs=[predict])
    
    return model
def MusicTaggerSampleCNN(SEGMENT_LEN, ex_parameters,context_count = None):
    NUM_CLASS = ex_parameters["NUM_CLASS"]
    trainable = ex_parameters["trainable"]
    last_dimension = ex_parameters["last_dimension"]
    '''
                    MULTI_OUTPUT=False,SPLIT=False,,alpha = 1.0, last_dimension
                    = 512,trainable=True'''
    if last_dimension == None: 
        last_dimension = 512
    dropout_rate = 0.5
    kernel_initializer = 'he_uniform'
    activation = 'relu'
    audio_input = Input(shape=(SEGMENT_LEN,), dtype='float32')
    net = Reshape([-1, 1])(audio_input)
    # 59049 X 1
    net = Conv1D(128, 3, strides=3, padding='valid', name = "conv1",
                 kernel_initializer=kernel_initializer,trainable=trainable)(net)
    net = BatchNormalization(name = "bn1")(net)
    hidden_input = Activation(activation)(net)

    
    class Block():
        def __init__(self,D,block_index):
            self.layers = [Conv1D(D, 3, padding='same',
                     kernel_initializer=kernel_initializer,name = "conv"+str(block_index),trainable=trainable),
                BatchNormalization(name = "bn"+str(block_index)),
                Activation(activation),
                MaxPool1D(3,name = "pool"+str(block_index))
            ]
            
        def apply(self,net):
            for layer in self.layers:
                net = layer(net) 
            return net 
    class Last_Block():
        def __init__(self,D,block_index):
            self.layers = [Conv1D(D, 1, padding='same',name = "conv"+str(block_index),
                     kernel_initializer=kernel_initializer,trainable=trainable),
                BatchNormalization(name = "bn"+str(block_index)),
                Activation(activation),
                Dropout(dropout_rate),
                Flatten()
            ]
        def apply(self,net):
            for layer in self.layers:
                net = layer(net) 
            return net 
    hidden_blocks = [Block(128,2),Block(128,3),Block(256,4),
        Block(256,5),Block(256,6),Block(256,7),Block(256,8),Block(256,9),
        Block(last_dimension,10),Last_Block(last_dimension,11),
        Dense(units=NUM_CLASS, activation='sigmoid')
        ]
    model = ConfigureNetwork(ex_parameters,context_count,hidden_blocks,hidden_input,audio_input)
    '''
    In ex_parameters , there are : NUM_CLASS,MULTI_OUTPUT,SPLIT,TAG_WISE_WEIGHTING,VALID_TAG_WISE_WEIGHTING,OUTPUT_ATTENTION,BINARY_DEPENDENCE,VALID_TUNED,LAYER_WISE_ATTENTION,alpha
    '''
    return model 


def MusicTaggerCNN(ex_parameters,context_count=None):
    NUM_CLASS = ex_parameters["NUM_CLASS"]
    trainable = ex_parameters["trainable"]
    last_dimension = ex_parameters["last_dimension"]
    if last_dimension == None: 
        last_dimension = 64
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
    ORIGINAL_SR = 22050
    SR = 12000
    DURA = 29.125
    audio_input = Input(shape=(int(DURA*ORIGINAL_SR),))
    x = Reshape((int(DURA*ORIGINAL_SR),1,1))(audio_input) 
#x = ResizeSignal(output_dim=(int(DURA*SR),1))(x) 
    x = Lambda(lambda signal:K.tf.image.resize_images(signal,(int(DURA*SR),1)))(x)
    x = Reshape((1,int(DURA*SR)))(x) 
    x = Melspectrogram(n_dft=512, n_hop=256, power_melgram=2.0,
                             input_shape=(1, int(DURA*SR)),
                             trainable_kernel=False,
                             trainable_fb=False,
                             return_decibel_melgram=True,
                             sr=SR, n_mels=96,
                             name='melgram')(x) 
    
    if K.image_dim_ordering() == 'th':
        channel_axis = 1
        freq_axis = 2
        time_axis = 3
    else:
        channel_axis = 3
        freq_axis = 1
        time_axis = 2

    # Input block
    x = BatchNormalization(axis=freq_axis, name='bn_0_freq')(x)

    # Conv block 1
    x = Convolution2D(64, 3, 3, border_mode='same', name='conv1',trainable=trainable)(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn1')(x)
    x = ELU()(x)
    hidden_input  = MaxPooling2D(pool_size=(2, 4), name='pool1')(x)
    class Block():
        def __init__(self,D,pool_size,block_index):
            self.layers = [Convolution2D(D, 3, 3, border_mode='same',name = 'conv'+str(block_index),trainable=trainable),
                BatchNormalization(axis=channel_axis, mode=0,name = "bn"+str(block_index)),
                ELU(),
                MaxPooling2D(pool_size=pool_size,name = "pool"+str(block_index))
            ]

        def apply(self,x):
            for layer in self.layers:
                x = layer(x)
            return x
    class Last_Block():
        def __init__(self,D,pool_size,block_index):
            self.layers = [Convolution2D(D, 3, 3, border_mode='same',name = 'conv'+str(block_index),trainable=trainable),
                BatchNormalization(axis=channel_axis, mode=0,name = "bn"+str(block_index)),
                ELU(),
                MaxPooling2D(pool_size=pool_size,name = "pool"+str(block_index)),
                Flatten()
            ]

        def apply(self,x):
            for layer in self.layers:
                x = layer(x)
            return x
    hidden_blocks = [
        Block(128,(2,4),2),Block(128,(2,4),3),Block(last_dimension*2,(3,5),4),Last_Block(last_dimension,(4,4),5),
        Dense(NUM_CLASS, activation='sigmoid')
    ]
    
    
    # Create model
    model = ConfigureNetwork(ex_parameters,context_count,hidden_blocks,hidden_input,audio_input)
    return model
    


def MusicTaggerCRNN(ex_parameters,context_count=None):
    NUM_CLASS = ex_parameters["NUM_CLASS"]
    trainable = ex_parameters["trainable"]
    last_dimension = ex_parameters["last_dimension"]
    if last_dimension==None: 
        last_dimension=32
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
    ORIGINAL_SR = 22050
    SR = 12000
    DURA = 29.125
    audio_input = Input(shape=(int(DURA*ORIGINAL_SR),))
    x = Reshape((int(DURA*ORIGINAL_SR),1,1))(audio_input)
    x = Lambda(lambda signal:K.tf.image.resize_images(signal,(int(DURA*SR),1)))(x)
#x = ResizeSignal(output_dim=(int(DURA*SR),1))(x)
    x = Reshape((1,int(DURA*SR)))(x)
    x = Melspectrogram(n_dft=512, n_hop=256, power_melgram=2.0,
                             input_shape=(1, int(DURA*SR)),
                             trainable_kernel=False,
                             trainable_fb=False,
                             return_decibel_melgram=True,
                             sr=SR, n_mels=96,
                             name='melgram')(x)
    
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
    x = ZeroPadding2D(padding=(0, 37))(x)
    x = BatchNormalization(axis=freq_axis, name='bn_0_freq')(x)

    # Conv block 1
    x = Convolution2D(64, 3, 3, border_mode='same', name='conv1',trainable=trainable)(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn1')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)
    hidden_input = Dropout(0.1, name='dropout1')(x)

    # Hidden Block Started Here 
    class Block():
        def __init__(self,pool_size,strides,block_index):
            self.layers = [
                Convolution2D(128, 3, 3, border_mode='same', name='conv'+str(block_index),trainable=trainable),
                BatchNormalization(axis=channel_axis, mode=0, name='bn'+str(block_index)),
                ELU(),
                MaxPooling2D(pool_size=pool_size, strides=strides, name='pool'+str(block_index)),
                Dropout(0.1, name='dropout'+str(block_index))
            ]
        def apply(self,x):
            for layer in self.layers:
                x = layer(x) 
            return x 
    class RecurrentBlock():
        def __init__(self,gru_dim,return_sequences,with_dropout,block_index): 
            self.recurrent_layer = GRU(gru_dim, return_sequences=return_sequences, name='gru'+str(block_index)+"_"+str(gru_dim),trainable=trainable)
            self.with_dropout = with_dropout
            self.block_index = block_index
        def apply(self,x):
            if self.block_index == 1:
                if K.image_dim_ordering() == 'th':
                    x = Permute((3, 1, 2))(x)
                x = Reshape((15, 128))(x)
            x = self.recurrent_layer(x)
            if self.with_dropout:
                x = Dropout(0.3)(x) 
            return x 

    hidden_blocks = [Block((3, 3),(3, 3),2),Block((4, 4),(4, 4),3),Block((4, 4),(4, 4),4),
        RecurrentBlock(last_dimension,True,False,1),RecurrentBlock(last_dimension,False,True,2),
        Dense(NUM_CLASS, activation='sigmoid')
        ]
    model = ConfigureNetwork(ex_parameters,context_count,hidden_blocks,hidden_input,audio_input)
    return model
    
