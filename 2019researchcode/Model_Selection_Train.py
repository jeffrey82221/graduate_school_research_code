# CONTROL PARAMETERS:
from Model_Util import *
import sys
import os
import tensorflow as tf
tf.set_random_seed(0)
import pprint
import datetime
import numpy_indexed as npi
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
from keras.utils import multi_gpu_model
from scipy.stats.mstats import gmean
config = tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
config.gpu_options.per_process_gpu_memory_fraction = 0.11
# 1/2, 1/3, 1/4 or 1/5 , CRNN needs 0.49 to work, CNN needs 0.25 to work,
# SampleCNN needs 0.11 to work
K.set_session(tf.Session(graph=tf.get_default_graph(),config=config))
file_name = '../jeffrey_data/saved_result/cloud_2_'
reduced_dimension = None
sample_count = 0
EPOCHS = 1001
using_MSD_Benchmark_Split = True
########## Testing Tids: ##############################################
if using_MSD_Benchmark_Split:
    f = open('MSD_split_for_tagging/test_x_msd_id.txt')
    test_tids = map(lambda x: x.split("\n")[0], f)
    f = open('MSD_split_for_tagging/valid_x_msd_id.txt')
    validation_tids = map(lambda x: x.split("\n")[0], f)
    print "test size:", len(test_tids)
    print "validation size:", len(validation_tids)
else:
    tids = list(FeatureRetriever().cleaned_saved_tids() & RawRetriever().cleaned_saved_tids() & set(context_match_tids(
        ['tag'])))  # 176992
    train_tids_whole = tids[:100000]
    train_tids = list(FeatureRetriever().cleaned_saved_tids() & RawRetriever().cleaned_saved_tids() & set(context_match_tids(
        ['listen', 'tag', 'playlist', 'artist'])))  # 62311
    all_experimental_train_tids = list(set(train_tids_whole) | set(train_tids))
    suitable_test_tids = list(set(tids) - set(all_experimental_train_tids))
    # match sure all can be match to all the tags
    tag_matcher = Top_Tag_Matcher(50)
    top_50_tags = tag_matcher.match_one_by_one(suitable_test_tids)
    tag_matcher = Top_Tag_Matcher(1000)
    top_1000_tags = tag_matcher.match_one_by_one(suitable_test_tids)
    test_tid_tags_table = pd.DataFrame(
        [suitable_test_tids, top_50_tags, top_1000_tags]).T
    test_tid_tags_table.columns = ['tid', 'top50', 'top1000']
    suitable_test_tids = list(test_tid_tags_table.ix[(test_tid_tags_table.top50.map(
        lambda x:len(x) > 0)) & (test_tid_tags_table.top1000.map(lambda x:len(x) > 0))].tid)
    print "suitable_test_tids", len(suitable_test_tids)
    validation_tids = suitable_test_tids[:10000]
    test_tids = suitable_test_tids[10000:]
    print "test size:", len(test_tids)
    print "validation size:", len(validation_tids)
########## Save Testing Data ##############################################
class Tag_Batch_Matcher():
    def __init__(self,train_tids,train_mode,NUM_CLASS):
        self.train_tids = np.array(train_tids) 
        self.train_data = load_data(train_mode, "tag", NUM_CLASS=NUM_CLASS) 
    def match(self,tids):
        idx = npi.indices(self.train_tids, tids) 
        return self.train_data[idx,:]
########## Generate Train Tids ##############################################
def segmentize_data(input_data,n_sample_per_segment = 59049,validation = True):
    audio_data = input_data[0]
    ground_truth_data = input_data[1]
    n_segment_per_audio = audio_data.shape[1]/n_sample_per_segment
    audio_data = audio_data[:,:n_sample_per_segment*n_segment_per_audio]
    #n_sample = 59049
    n_example = audio_data.shape[0]*audio_data.shape[1]/n_sample_per_segment
    reshaped_audio_data = audio_data.reshape((n_example,n_sample_per_segment))
    print "audio reshape"
    if validation:
        duplicated_ground_truth_data = np.repeat(ground_truth_data,n_segment_per_audio,axis = 0)
        print "ground_truth copying"
        output_data = reshaped_audio_data,duplicated_ground_truth_data
    else:
        output_data = reshaped_audio_data,ground_truth_data
    return output_data
def load_model_weights_file_name(ex_parameters,folder ="./saved_model_weights"):
    file_names = filter(lambda x:".model_weight_ex_parameters" in x,os.listdir(folder))
    for name in file_names: 
        ex_parameters_ = load_object(folder+"/"+name)
#print ex_parameters_,ex_parameters 
        if ex_parameters == ex_parameters_: 
            result_name = name
    return folder+"/"+result_name.split(".model_weight_ex_parameters")[0]+".h5"
def get_val_loss_calculator(model,validation_data):
    def val_loss(y_true,y_predict):
        return 1.234
    return  val_loss
def val_batch_generator(validation_data,batch_size):
    def batches_generator(array, batch_size):
        def batch_yieldor(array):
            for i in range(int(len(array) / batch_size)): 
                yield array[i * batch_size:(i + 1) * batch_size]
        def train_batch_generator(array):
            while True:
                yield batch_yieldor(array)

        return it.chain.from_iterable(train_batch_generator(array))

    audio_data_generator = batches_generator(validation_data[0], batch_size)
    tag_data_generator = batches_generator(validation_data[1], batch_size)
    while True:
        yield next(audio_data_generator),[next(tag_data_generator),next(tag_data_generator)]
train_used_percentage = 1.0#default: 1.0, ex. 0.25
optimizer = "nadam_better_stopper" # originally is sgd
early_stopping_metric = ("retrieval","AUROC")
MODEL = str(sys.argv[1])
MEAN = "geometric"#"arithmetric" # geometric
last_dimension = int(sys.argv[4]) #SampleCNN 256, CNN, CRNN, 512
CONTEXT_STRING = str(sys.argv[2])
single_output = False #default:False, if context is None, auto become True 
SPLIT = False # meaning the use of single-output or dual-output for tag-propagation 
alpha = float(sys.argv[3]) #Initial Inter-task weight (baseline: 0. , with-propagation: 1.)
auxiliary_weight = 1.0 # this is the sampling weight of auxiliary tags (very similar to the function of alpha, but implemented with sampling scheme)
gamma = 0. # weight of focal loss 

ATTENTION_SCHEME = {
    "InterClass":(True,{"VALID_TUNED":False,"DUAL":False}), #LOSS_BALANCED = False # 
    "InterAttribute":(False,{"VALID_TUNED":False,"DUAL":False}), # TAG_WISE_WEIGHTING = True # VALID_TAG_WISE_WEIGHTING = False # 
    "InterTask":(True,{"VALID_TUNED":False,"DUAL":True}) #OUTPUT_ATTENTION = False#VALID_TUNED = False#BINARY_DEPENDENCE = False ## DECREASE_BASED = False # 
}
LAYER_WISE_ATTENTION = False

trainable = True 
transfering = False

if CONTEXT_STRING == "None":
    single_output = True# when CONTEXT = None, set this to True
    SPLIT = False
    alpha = 0.0
    auxiliary_weight = -1.
    trainable = True
    ATTENTION_SCHEME["InterTask"] = (False,{"VALID_TUNED":False,"DUAL":False})
    LAYER_WISE_ATTENTION = False
    
for train_mode in ['context_matched_train']:
    if train_mode == "context_matched_train":
        # retrieved from overlap version audio file
        f = open('MSD_split_for_tagging/context_matched_train_x_msd_id.txt')
        train_tids_whole = map(lambda x: x.split("\n")[0], f)
        train_data_size_range = [int(len(train_tids_whole)*train_used_percentage)]
    elif train_mode == "train":
        f = open('MSD_split_for_tagging/train_x_msd_id.txt')
        train_tids_whole = map(lambda x: x.split("\n")[0], f)
        train_data_size_range = [int(len(train_tids_whole)*train_used_percentage)]
    # shoud overlap with the retrieved audio file
    for train_data_size in train_data_size_range:  # 0.2, 0.4, 0.6, 0.8, 1.0
        train_tids = train_tids_whole[:train_data_size]
        PAIRS_PER_EPOCH = len(train_tids)
        print "PAIRS_PER_EPOCH", PAIRS_PER_EPOCH
        # for decay in [0.1]:
        for NUM_CLASS in [1000]:
            for TAGGING_MODEL in [MODEL]:  # ,"CNN","CRNN", "OldSampleCNN", CNN512
                # 3: random 4: naive_playlist, 5: listen , 6: tag, # not yet7 (our server): artist, 
                for CONTEXT in [CONTEXT_STRING]: # 'listen', 'playlist', ,'tag','artist', 'naive_playlist'
                    if CONTEXT == "playlist":
                        #decays = list(np.arange(0.0, 1.0, 0.1)) 
                        decays = [auxiliary_weight]
                    else:
                        decays = [0.]
                    for decay in decays:
                        ex_parameters = {"TAGGING_MODEL": TAGGING_MODEL,
                                         "CONTEXT": CONTEXT,
                                         "single_output": single_output,
                                         "NUM_CLASS": NUM_CLASS,
                                         "optimizer":optimizer,
                                         "early_stopping_metric":early_stopping_metric,
                                         "MEAN":MEAN,
                                         "train_mode": train_mode,
                                         "train_data_size": train_data_size,
                                         "sample_count": sample_count,
                                         "last_dimension": last_dimension,
                                         "trainable":trainable,
                                         "SPLIT":SPLIT,
                                         "ATTENTION_SCHEME":ATTENTION_SCHEME,
                                         "LAYER_WISE_ATTENTION":LAYER_WISE_ATTENTION,
                                         "alpha":alpha,
                                         "gamma":gamma,
                                         "output_activation":"loss_weight",# meaning: weight is apply on weight rather than the final output 
                                         "auxiliary_weight":auxiliary_weight,
                                         'transfering':transfering
#                                        "BATCH_SIZE":BATCH_SIZE
                                         }
                        # INITIAL_EPOCH = get_initial_epoch(ex_parameters,["artist_","playlist_","None"])
                        INITIAL_EPOCH = 0
                        if INITIAL_EPOCH >= EPOCHS:
                            continue
                        print trainable
                        model, content_retriever, BATCH_SIZE = connect_network(
                            ex_parameters)
                        model.summary()
                        print "summarized"
                        if CONTEXT!="None" and transfering:
                            ex_parameters_tmp = ex_parameters.copy()
                            ex_parameters_tmp["CONTEXT"] = "None"
                            ex_parameters_tmp["trainable"]=True
                            del ex_parameters_tmp["SPLIT"]
                            del ex_parameters_tmp["alpha"]
                            del ex_parameters_tmp["auxiliary_weight"] 
                            del ex_parameters_tmp["transfering"]
                            model.load_weights(load_model_weights_file_name(ex_parameters_tmp),by_name=True,skip_mismatch=True)
                            print "Model Weight Loaded"
                        tag_batch_matcher = Tag_Batch_Matcher(train_tids,train_mode,NUM_CLASS)
                        '''if LOSS_BALANCED:
                                                                                                    exist_prob = np.sum(tag_batch_matcher.train_data,axis=0)/train_data_size
                                                                                                else:
                                                                                                    exist_prob = np.array([0.5]*NUM_CLASS)'''
                        connect_network_optimization(model,ex_parameters,optimizer_setup=optimizer)
                        ## NETWORK OPTIMIZATION CONSTRUCTED #########################################################################
                        checkpoint = link_to_checkpoint(
                            model, ex_parameters, load=INITIAL_EPOCH != 0)
                        
                        ## WEIGHT LOADED #########################################################################
                        batch_generation_object = Batch_Generation_Class(
                            content_retriever, NUM_CLASS, context=CONTEXT,
                            fix_tid_list=train_tids, random_select=True,
                            reduced_dimension=reduced_dimension,tag_batch_matcher=tag_batch_matcher,self_select_P=1.-auxiliary_weight)
                        print "batch_generation_object BUILD"
                        train_batch_generator = batch_generation_object.get_generator(
                            train_tids, batch_size=BATCH_SIZE,
                            decay=decay,MULTI_OUTPUT = CONTEXT!="None", 
                            single_output=single_output)                        
                        print "train_batch_generator BUILD"
                        if TAGGING_MODEL == "SampleCNN":
                            validation_data = load_folded_data(
                                "valid", "audio",folder = "/home/mpac/"),load_data("valid", "tag", NUM_CLASS=NUM_CLASS)
                            test_data = load_folded_data("test", "audio"),load_data(
                                "test", "tag", NUM_CLASS=NUM_CLASS)
                        else:
                            validation_data = load_data(
                                "valid", "audio",folder = "/home/mpac/"), load_data("valid", "tag", NUM_CLASS=NUM_CLASS)
                            test_data = load_data("test", "audio"), load_data(
                                "test", "tag", NUM_CLASS=NUM_CLASS)

                        monitoring_adaptor = MonitoringTrainingAdaptor(tag_batch_matcher.train_data,validation_data,ex_parameters,PAIRS_PER_EPOCH
            // BATCH_SIZE,early_stopping_metric=early_stopping_metric,resolution=5)
                        pp = pprint.PrettyPrinter(indent=3)
                        pp.pprint(ex_parameters)
                        model.fit_generator(train_batch_generator,
                                            steps_per_epoch=PAIRS_PER_EPOCH // BATCH_SIZE,
                                            epochs=EPOCHS,
                                            verbose=1,
                                            callbacks=[monitoring_adaptor],
                                            initial_epoch=INITIAL_EPOCH,
                                            use_multiprocessing=False,
                                            workers = 1
                                            )
                        monitoring_adaptor.save_result()
                        print "monitoring result saved"
                        predicted_tag_matrix = result_predict(model, ex_parameters, test_data)
                        if TAGGING_MODEL == "SampleCNN":
                            if MEAN == 'geometric':
                                predicted_tag_matrix = gmean(predicted_tag_matrix.reshape((10,-1,NUM_CLASS),order = 'F'),axis = 0)
                            else:
                                predicted_tag_matrix = np.mean(predicted_tag_matrix.reshape((10,-1,NUM_CLASS),order = 'F'),axis = 0)	
                        print "FINAL Tag Predicted"
                        code = str(datetime.datetime.now())
                        model.save("./saved_model/"+code+".h5")
                        save_object(ex_parameters,"./saved_model/"+ code+".model_ex_parameters")
                        print TAGGING_MODEL+"MODEL SAVED"
                        print "final evaluation DONE"
                        code = str(datetime.datetime.now())
                        f = open(
                            "saved_predicted_tag_matrix/file_name.txt", 'wb')
                        f.write(code + "\n")
                        f.close()
                        save_object(
                            ex_parameters, "saved_predicted_tag_matrix/" + code + ".ex_parameters")
                        save_object(
                            predicted_tag_matrix, "saved_predicted_tag_matrix/" + code + ".matrix")
                        print "predicted matrix saved"
                        # save predicted result for latter evaluation
                        del test_data
                        del validation_data
                        content_retriever.close()
                        del content_retriever
                        print "data delete"
