import pandas as pd
from Batch_Generation_Class import *
import glob
#from Evaluation import IntervalEvaluation, LrReducer, Tag_Propability_Visualizer
from Evaluation import *
#from music_taggers import MusicTaggerLR
from keras import optimizers, metrics
from audio_processor import compute_melgram, random_select_segment
from music_taggers import MusicTaggerLR, MusicTaggerCNN, MusicTaggerCRNN, MusicTaggerSampleCNN
from keras.callbacks import Callback, ModelCheckpoint
import gzip
import pickle


def save_object(object, filename, protocol=0):
    """Saves a compressed object to disk
    """
    file = gzip.GzipFile(filename, 'wb')
    file.write(pickle.dumps(object, protocol))
    file.close()


def load_object(filename):
    """Loads a compressed object from disk
    """
    file = gzip.GzipFile(filename, 'rb')
    buffer = ""
    while True:
        data = file.read()
        if data == "":
            break
        buffer += data
    object = pickle.loads(buffer)
    file.close()
    return object

def load_folded_data(mode, style, NUM_CLASS=50,folder = '/home/mpac/'):
    f = open('MSD_split_for_tagging/' + mode + '_x_msd_id.txt')
    tids = map(lambda x: x.split("\n")[0], f)
    print mode, style, len(tids)
    if style == "audio":
        sample_rate = 22050
        duration = 29.125
        n_sample = 59049
        n_segment = 10
#folder = '../jeffrey_data/audio_tmp/'
        f = open('MSD_split_for_tagging/' + mode + '_x_msd_id.txt')
        tids = map(lambda x: x.split("\n")[0], f)
        folded_audio_data = np.memmap(folder + mode + '.audio.segmentized',
                               mode='r', dtype='float32', shape=(len(tids)*n_segment, int(n_sample)))
        return folded_audio_data
    elif style == "tag":
        filename = "/home/mpac/saved_test_val_data/" + \
            str(NUM_CLASS) + ".tag." + mode + ".dat"
        if not os.path.isfile(filename):
            tag_data_generator = Batch_Generation_Class(None, NUM_CLASS)
            tag_data = tag_data_generator.groundtruth_tag_batch_convertor(tids)
            fp = np.memmap(filename, dtype='float32',
                           mode='w+', shape=tag_data.shape)
            fp[:] = tag_data[:]
            print 'data load'
        tag_data = np.memmap(filename, dtype='float32', mode='r', shape=(len(tids), NUM_CLASS)) 
        n_segment = 10
        duplicated_tag_data = np.repeat(tag_data,n_segment,axis = 0)
        return duplicated_tag_data

def load_data(mode, style, NUM_CLASS=50,folder = '/home/mpac/'):
    f = open('MSD_split_for_tagging/' + mode + '_x_msd_id.txt')
    tids = map(lambda x: x.split("\n")[0], f)
    print mode, style, len(tids)
    if style == "audio":
        sample_rate = 22050
        duration = 29.125
        audio_data = np.memmap(folder + mode + '.audio',
                               mode='r', dtype='float32', shape=(len(tids), int(sample_rate * duration)))
        return audio_data
    elif style == "tag":
        filename = "/home/mpac/saved_test_val_data/" + \
            str(NUM_CLASS) + ".tag." + mode + ".dat"
        if not os.path.isfile(filename):
            tag_data_generator = Batch_Generation_Class(None, NUM_CLASS)
            tag_data = tag_data_generator.groundtruth_tag_batch_convertor(tids)
            fp = np.memmap(filename, dtype='float32',
                           mode='w+', shape=tag_data.shape)
            fp[:] = tag_data[:]
            print 'data load'
        return np.memmap(filename, dtype='float32', mode='r', shape=(len(tids), NUM_CLASS))


def get_initial_epoch(ex_parameters, file_names):
    # INPUT : a dict with parameters specifying a experiement model
    # OUTPUT : the last epoch trained for the model
    try:
        performance_table_list = []
        for file_name in file_names:
            performance_table = pd.read_hdf(
                file_name + "_".join(list(ex_parameters.keys())) + '.performance.h5', key='average')
            print file_name + "_".join(list(ex_parameters.keys())) + '.performance.h5', "LOAD"
            performance_table_list.append(performance_table)
        performance_table = pd.concat(performance_table_list, axis=0)
    except:
        INITIAL_EPOCH = 0
        print "Training of ", ex_parameters, "START FROM EPOCH", INITIAL_EPOCH
        return INITIAL_EPOCH
    #performance_table = pd.read_hdf("performance_table.h5")
    for i, parameter_type in enumerate(ex_parameters.keys()):
        if i == 0:
            constrain = (
                performance_table[parameter_type] == ex_parameters[parameter_type])
        else:
            constrain = (constrain) & (
                performance_table[parameter_type] == ex_parameters[parameter_type])
    if sum(list(constrain)) > 0:
        max_epoch = max(performance_table[constrain].EPOCH)
        INITIAL_EPOCH = max_epoch + 1
        print "Training of ", ex_parameters, "START FROM EPOCH", INITIAL_EPOCH
        return INITIAL_EPOCH
    else:
        INITIAL_EPOCH = 0
        print "Training of ", ex_parameters, "START FROM EPOCH", INITIAL_EPOCH
        return INITIAL_EPOCH


def model_name(ex_parameters):
    return "_".join(map(lambda x: "=".join([x[0], str(x[1])]), ex_parameters.items()))


def link_to_checkpoint(model, ex_parameters, load=False): # model_name can be too long 
    check_point_directory = "/data/jeffrey82221/tag_models/"
    weight_file_dir = check_point_directory + \
        model_name(ex_parameters) + ".hdf5"
    if load:
        model.load_weights(weight_file_dir)
        print "model weight LOAD from", weight_file_dir
    checkpoint = ModelCheckpoint(weight_file_dir,
                                 monitor=None,
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min',
                                 period=1)
    return checkpoint


def connect_network(ex_parameters,BATCH_SIZE = 10):
    # LOAD PARAMETERS:
    #NUM_CLASS = ex_parameters['NUM_CLASS']
    TAGGING_MODEL = ex_parameters['TAGGING_MODEL']
    CONTEXT = ex_parameters['CONTEXT']
    # CONNECT NETWORK
    if type(CONTEXT)==set:
        context_count = len(CONTEXT)
    else:
        context_count = None
    print "context_count",context_count
#if CONTEXT != "None" and (alpha!=0.0 and alpha!=1.0):
    if CONTEXT != "None":
        MULTI_OUTPUT = True
    else:
        MULTI_OUTPUT = False
    if TAGGING_MODEL == "LR":
        model = MusicTaggerLR(1216, ex_parameters,context_count = context_count)
        content_retriever = FeatureRetriever()
#BATCH_SIZE = 1000
        #LEARNING_RATE = 0.01
    elif TAGGING_MODEL == "SampleCNN" or TAGGING_MODEL == "OldSampleCNN":
        model = MusicTaggerSampleCNN(59049, ex_parameters,context_count= context_count)
        content_retriever = NumpyAudioRetriever(
            ex_parameters['train_mode'], processor=random_select_segment)
#       BATCH_SIZE = 100
        #LEARNING_RATE = 0.0005
    elif TAGGING_MODEL == "CNN" or TAGGING_MODEL == "CRNN":
        if TAGGING_MODEL == "CNN":
            model = MusicTaggerCNN(ex_parameters,context_count = context_count)
#           BATCH_SIZE = 10
        elif TAGGING_MODEL == "CRNN":
            model = MusicTaggerCRNN(ex_parameters,context_count=context_count)
#           BATCH_SIZE = 10
        content_retriever = NumpyAudioRetriever(ex_parameters['train_mode'])
    print model, content_retriever, BATCH_SIZE
    return model, content_retriever, BATCH_SIZE
def get_val_loss_calculator(model):                              
    def val_loss(y_true,y_predict):                                              
        return 1.234                                                             
    return val_loss 
#############################################################################################
# # # Customized Losses # # # 
from keras import backend as K
# TODO: how to apply exist_prob to the loss ? 
# labels.shape = (S, 1000) 
# exist_prob.shape = (1000,) => become like this np.array([[1.0]*1000]) 
# np.tile(np.expand_dims(exist_prob,1),labels.shape[0]).T
# 
from keras.layers import Multiply
def balanced_exist_focal_loss(): # change to tag-wise loss 
                                                  
    #w=1.-exist_prob
    def exist_focal_loss(labels,y_pred,gamma=2):
        L=-labels*((1-y_pred)**gamma)*K.log(y_pred)
        L = K.mean(L,axis=0)
        #L = Multiply()(
        #        [L, K.variable(w,dtype=L.dtype)])
        return L
    return exist_focal_loss
def balanced_nonexist_focal_loss():# change to tag-wise loss  
    #w = exist_prob
    def nonexist_focal_loss(labels,y_pred,gamma=2):
        L = -(1-labels)*(y_pred**gamma)*K.log(1-y_pred)
        L = K.mean(L,axis=0)
        #L = Multiply()(
        #        [L, K.variable(w,dtype=L.dtype)])
        return L
    return nonexist_focal_loss
def balanced_focal_loss(weights): # change to tag-wise loss 
    # exist_prob = exist_prob
    weights = K.squeeze(weights,axis = 0)
    def focal_loss(labels,y_pred,gamma=2):
        L = balanced_exist_focal_loss()(labels,y_pred,gamma=gamma)+balanced_nonexist_focal_loss()(labels,y_pred,gamma=gamma)
        L = Multiply()(
                [L, weights])
        return L
    return focal_loss
def fix_weight_binary_crossentropy(tagwise_weights,weight,gamma=2):
    def loss(y_true,y_pred):
        total_loss = K.mean(balanced_focal_loss(tagwise_weights)(y_true,y_pred,gamma=gamma)) 
        return total_loss*weight
    return loss
def attention_binary_crossentropy(tagwise_weights,weights,gamma=2):
    #auxiliary_attention = K.squeeze(auxiliary_attention,axis = 0)                   
    weights = K.squeeze(weights,axis = 0)
    def loss(y_true,y_pred):
        tag_wise_loss = balanced_focal_loss(tagwise_weights)(y_true,y_pred,gamma=gamma)
        tag_wise_loss = Multiply()(
                [tag_wise_loss, weights])
        return K.mean(tag_wise_loss)
    return loss
def classwise_attention_binary_crossentropy(tagwise_weights,tag_exist_weights,tag_nonexist_weights,gamma=2):
    # how this is apply with tagwise_weights? 
    tag_exist_weights = K.squeeze(tag_exist_weights,axis = 0)
    tag_nonexist_weights = K.squeeze(tag_nonexist_weights,axis = 0)
    tagwise_weights = K.squeeze(tagwise_weights,axis=0)
    def loss(y_true,y_pred):
        tag_wise_exist_loss = balanced_exist_focal_loss()(y_true,y_pred,gamma=gamma)
        tag_wise_nonexist_loss = balanced_nonexist_focal_loss()(y_true,y_pred,gamma=gamma)
        tag_wise_exist_loss = Multiply()(
                [tag_wise_exist_loss, tag_exist_weights])
        tag_wise_nonexist_loss = Multiply()(
                [tag_wise_nonexist_loss, tag_nonexist_weights])
        L = tag_wise_exist_loss + tag_wise_nonexist_loss
        L = Multiply()([L, tagwise_weights])
        return K.mean(L)#+K.mean(tag_wise_nonexist_loss)
    return loss
#############################################################################################
def connect_network_optimization(model, ex_parameters, optimizer_setup = "sgd"): 
    # LOAD PARAMETERS:                                                           
    CONTEXT = ex_parameters['CONTEXT']   
    gamma = ex_parameters["gamma"]
    single_output = ex_parameters[                    
        'single_output']                              
    # CONNECT NETWORK                                                            
    if optimizer_setup == "sgd":                                                 
        optimizer = optimizers.SGD(lr=0.01, decay=1e-6,momentum=0.9,nesterov=True)
    elif optimizer_setup == "nadam_better_stopper":                              
        optimizer = optimizers.Nadam(lr=0.002,beta_1=0.9,beta_2=0.999,schedule_decay=0.004)
    elif optimizer_setup == "nadam_no_schedule":                                 
        optimizer = optimizers.Nadam(lr=0.002,beta_1=0.9,beta_2=0.999,schedule_decay=0.0)
    def model_compile_configuration(objective_list):
        monitoring_count = len(model.outputs) - len(objective_list)
        model.compile(loss=objective_list + [None]*monitoring_count,optimizer=optimizer,metrics=[])
    if CONTEXT != "None":                                                        
        if single_output:  
            model_compile_configuration(["binary_crossentropy"])
        else:                                                                    
            if type(CONTEXT) == set:
                # TODO: when fixing methods , does the performance improves? 
                model_compile_configuration(["binary_crossentropy"]+["binary_crossentropy"]*len(CONTEXT))
            else:# using attention-based or fix_weight binary cross entropy 
                if ex_parameters["OUTPUT_ATTENTION"]:
                    if ex_parameters["BINARY_DEPENDENCE"]:
                        target_loss = classwise_attention_binary_crossentropy(model.outputs[-6],model.outputs[-2],model.outputs[-4],gamma=gamma)
                        auxiliary_loss = classwise_attention_binary_crossentropy(model.outputs[-5],model.outputs[-1],model.outputs[-3],gamma=gamma)
                        # outputs[-1] -> axuiliary attention 1 (for exist tag)
                        # outputs[-2] -> target attention 1    (for exist tag)
                        # outputs[-3] -> axuiliary attention 2 (for non-exist tag)
                        # outputs[-4] -> target attention 2    (for non-exist tag)
                        # outputs[-5] -> auxiliary tag-wise weights   
                        # outputs[-6] -> target tag-wise weights  
                    else:
                        target_loss = attention_binary_crossentropy(model.outputs[-4],model.outputs[-2],gamma=gamma)
                        auxiliary_loss = attention_binary_crossentropy(model.outputs[-3],model.outputs[-1],gamma=gamma)
                        # outputs[-3] -> auxiliary tag-wise weights   
                        # outputs[-4] -> target tag-wise weights   
                else:
                    alpha = ex_parameters["alpha"]
                    target_loss = fix_weight_binary_crossentropy(model.outputs[-2],(1./(1.+alpha)),gamma=gamma)
                    auxiliary_loss = fix_weight_binary_crossentropy(model.outputs[-1],(alpha/(1.+alpha)),gamma=gamma)
                    # outputs[-1] -> auxiliary tag-wise weights   
                    # outputs[-2] -> target tag-wise weights   
                model_compile_configuration([target_loss, auxiliary_loss])                     
    else:                                                                        
        model_compile_configuration(["binary_crossentropy"])
def load_ex_parameters_dict():
    filenames = glob.glob("/workspace/PROJECT/jeffrey_data/saved_result_matrix/*.ex_parameters")
    dicts = dict()
    for f in filenames:
        dicts[f.split('/')[-1].split('.ex')[0]] = load_object(f)
    return dicts

def select_name(attribute,parameter,dicts):
    result = []
    for k in dicts.keys():
        if dicts[k][attribute] == parameter:
            result.append(k) 
    return set(result)

# match to matrix 
def matrix(name):
    return load_object('/workspace/PROJECT/jeffrey_data/saved_result_matrix/'+name+".matrix")

def ex_parameters(name):
    return load_object('/workspace/PROJECT/jeffrey_data/saved_result_matrix/'+name+".ex_parameters")

# ex_parameters_info = load_ex_parameters_dict()
def select_file_names_by_parameter_constrain(parameter_constrain,ex_parameters_info = load_ex_parameters_dict()):
    result_set = set()
    for i,key in enumerate(parameter_constrain.keys()): 
        if i == 0: 
            result_set = select_name(key,parameter_constrain[key],ex_parameters_info)  
        else:
            result_set = result_set & select_name(key,parameter_constrain[key],ex_parameters_info)  
    return list(result_set)
