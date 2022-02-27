from keras import optimizers, metrics
import keras.backend as K
from keras.callbacks import Callback,ProgbarLogger
from Plotting_Notebooks.Plotting_Package import *
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
import pprint
from datetime import datetime
import numpy as np
import pandas as pd
import time
import itertools as it
from audio_processor import random_select_segment
import sys
import math
import gzip
import pickle
from scipy.stats.mstats import gmean


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
def generate_model_predictor(model, processor=None):
    if processor == None:
        return model.predict
    else:
        # the input data list should be generate
        def batching(x, batch_size, processor=processor):
            if type(x) == list:
                cyc_generators = map(it.cycle, x)
                while True:
                    yielding_data = []
                    for i, cyc_generator in enumerate(cyc_generators):
                        current_batch = list(
                            it.islice(cyc_generator, batch_size))
                        if i == 0:
                            yielding_data.append(
                                np.vstack(map(processor, current_batch)))
                        else:
                            yielding_data.append(np.vstack(current_batch))
                    yield yielding_data
            else:
                cyc_generator = it.cycle(x)
                while True:
                    current_batch = list(it.islice(cyc_generator, batch_size))
                    yield np.vstack(map(processor, current_batch))

        def model_predict_with_processor(x, batch_size=None, verbose=0, steps=None):
            if batch_size == None:
                batch_size = 32
            if steps == None:
                if type(x) == list:
                    count_list = map(len, x)
                    assert all(x == count_list[0] for x in count_list)
                    steps = len(x[0]) / batch_size
                else:
                    steps = len(x) / batch_size
            batch_generator = batching(x, batch_size, processor=processor)
            result = model.predict_generator(batch_generator,
                                             steps=steps - 2,
                                             workers=1,
                                             use_multiprocessing=False,
                                             verbose=verbose)
            if type(x) == list:
                final_index = len(x[0])
            else:
                final_index = len(x)
            if type(result) == list:
                end_index = len(result[0])
            else:
                end_index = len(result)
            if end_index < final_index:
                if type(x) == list:
                    end_data = map(lambda d: d[end_index:, :], x)
                    end_data[0] = np.vstack(map(processor, end_data[0]))
                else:
                    end_data = x[end_index:, :]
                end_result = model.predict(np.vstack(map(processor, end_data)))
                if type(result) == list:
                    for i in range(len(result)):
                        result[i] = np.vstack([result[i], end_result[i]])
                else:
                    result = np.vstack([result, end_result])
            return result
        return model_predict_with_processor


def result_predict(model, ex_parameters, test_data):
    audio_input = test_data[0]
    #ground_truth = test_data[1]
    if ex_parameters['TAGGING_MODEL'] == "OldSampleCNN" or ex_parameters['TAGGING_MODEL']=="OldSampleCNN64":
        batch_size = 10
        processor = random_select_segment# 
    else:
        batch_size = 10
        processor = None
    predict = generate_model_predictor(model, processor=processor)(audio_input, batch_size=batch_size, verbose=1)
    if len(model.outputs) == 1:
        pass 
    else:
        predict = predict[0]
    if ex_parameters["CONTEXT"]!="None" and ex_parameters['OUTPUT_ATTENTION'] == False:
        predict = predict * (1. + ex_parameters["alpha"]) # convert back to global output.  
    return predict 
def attention_generate(model, ex_parameters):
    '''
    model = MusicTaggerCRNN(1000,MULTI_OUTPUT=True,SPLIT=False,OUTPUT_ATTENTION = True,LAYER_WISE_ATTENTION=True,alpha = 1.,last_dimension=512,trainable=True)
    '''
    output_attention = None
    layer_wise_attention = None
    outputs = model.predict(np.ones((1,int(model.inputs[0].shape[1]))))
    if ex_parameters["OUTPUT_ATTENTION"]:
        if ex_parameters["BINARY_DEPENDENCE"]:
            output_attention = outputs[-1][0],outputs[-3][0] # existence attention / non-existence attention
        else:
            output_attention = outputs[-1][0]
    if ex_parameters["LAYER_WISE_ATTENTION"]:
        layer_wise_attention = map(lambda o:o[0][0],filter(lambda o:o.shape == (1,1),outputs))
    return layer_wise_attention,output_attention
def tagwise_weight_generate(model,ex_parameters):
    outputs = model.predict(np.ones((1,int(model.inputs[0].shape[1]))))
    if ex_parameters["OUTPUT_ATTENTION"]:
        if ex_parameters["BINARY_DEPENDENCE"]:
            #output_attention = outputs[-1][0],outputs[-3][0] # existence attention / non-existence attention
            target_weight = outputs[-6][0]
            auxiliary_weight = outputs[-5][0]
        else:
            target_weight = outputs[-4][0]
            auxiliary_weight = outputs[-3][0]
    else:
        target_weight = outputs[-2][0]
        auxiliary_weight = outputs[-1][0]
    return target_weight,auxiliary_weight
def last_evaluation(predict,test_data, tag_list, ex_parameters, file_name=""):
    file_name = file_name
    test_data = test_data

    tag_list = tag_list
    ex_parameters = ex_parameters
    parameter_types = map(str, list(ex_parameters.keys()))
    columns_of_tagwise = ['tag', 'measurement',
                               'performance'] + parameter_types
    columns_of_average = ['measurement',
                               'performance'] + parameter_types
    measurements = {"AUC-ROC": roc_auc_score,
                         "MAP": average_precision_score}
    pprint(ex_parameters)
    ground_truth = test_data[1]
  
    reduced_indices = (np.sum(ground_truth, axis=1) > 0)
    predict = predict[reduced_indices, :]
    ground_truth = ground_truth[reduced_indices, :]
    #
    for measurement_name in measurements.keys():

        performance = measurements[measurement_name](
            ground_truth, predict, average='samples')  # error

        print measurement_name,performance
        performance_table = pd.DataFrame([[measurement_name, performance] + map(
            lambda key:ex_parameters[key], parameter_types)], columns=columns_of_average, index=[datetime.now()])
        performance_table.to_hdf(file_name + "_".join(parameter_types) + '.performance.h5', key='average', model='a', format='t', append=True,
                                 complib='blosc', min_itemsize=dict([("measurement", 7)] + [(k, 25) for k in map(lambda x:x[0], filter(lambda x:type(x[1]) == str, ex_parameters.items()))]))  # for other parameters use 10

        for i in range(len(tag_list)):
            performance = measurements[measurement_name](
                ground_truth[:, i], predict[:, i])
            performance_table = pd.DataFrame([[str(tag_list[i]), measurement_name, performance] + map(
                lambda key:ex_parameters[key], parameter_types)], columns=columns_of_tagwise, index=[datetime.now()])
            performance_table.to_hdf(file_name + "_".join(parameter_types) + '.performance.h5', key='tag_wise', model='a', format='t', append=True,
                                     complib='blosc', min_itemsize=dict([("measurement", 7), ("tag", 52)] + [(k, 25) for k in map(lambda x:x[0], filter(lambda x:type(x[1]) == str, ex_parameters.items()))])) 

  
  
  
  
class IntervalEvaluation(Callback):
    def __init__(self, test_data, tag_list, ex_parameters, file_name=""):
        '''
        INPUT:
        1. eval_data : a dict with key as data name and item as data
        2. tag_list :  an array with str representing each tag
        3. save_parameter : a dict with key to be the type of a parameters (ex: CONTEXT, decay, AUTO_TAGGING_MODEL) and values the parameter (ex. playlist, 0.7, CNN)
        '''
        self.file_name = file_name
        self.test_data = test_data
        # self.validation_data = validation_data
        self.tag_list = tag_list
        self.ex_parameters = ex_parameters
        self.parameter_types = map(str, list(self.ex_parameters.keys()))
        self.columns_of_tagwise = ['EPOCH', 'tag', 'measurement',
                                   'performance'] + self.parameter_types
        self.columns_of_average = ['EPOCH', 'measurement',
                                   'performance'] + self.parameter_types
        self.measurements = {"AUC-ROC": roc_auc_score,
                             "MAP": average_precision_score}
        self.start_time = time.clock()

    def on_batch_begin(self, batch, logs={}):
        None

    def on_batch_end(self, batch, log={}):
        
        None
        
    def on_epoch_end(self, epoch, logs={}):
        
        pprint(self.ex_parameters)
        if epoch % 10 == 0:
            print "EPOCH:", epoch,
            ground_truth = self.test_data[1]
            predict = result_predict(
                self.model, self.ex_parameters, self.test_data)
            # remove data without tags:
            reduced_indices = (np.sum(ground_truth, axis=1) > 0)
            predict = predict[reduced_indices, :]
            ground_truth = ground_truth[reduced_indices, :]
            #
            for measurement_name in self.measurements.keys():
            #measurement_name = "MAP"
                performance = self.measurements[measurement_name](
                    ground_truth, predict, average='samples')  # error
                #print predict
                #print ground_truth
                print measurement_name,performance
                performance_table = pd.DataFrame([[epoch, measurement_name, performance] + map(
                    lambda key:self.ex_parameters[key], self.parameter_types)], columns=self.columns_of_average, index=[datetime.now()])
                performance_table.to_hdf(self.file_name + "_".join(self.parameter_types) + '.performance.h5', key='average', model='a', format='t', append=True,
                                         complib='blosc', min_itemsize=dict([("measurement", 7)] + [(k, 25) for k in map(lambda x:x[0], filter(lambda x:type(x[1]) == str, self.ex_parameters.items()))]))  # for other parameters use 10
                # save result tag wise by the performance of retrieving documents
                #measurement_name = "AUC-ROC"
                for i in range(len(self.tag_list)):
                    performance = self.measurements[measurement_name](
                        ground_truth[:, i], predict[:, i])
                    #print "AUC-ROC of ", str(self.tag_list[i]), performance
                    performance_table = pd.DataFrame([[epoch, str(self.tag_list[i]), measurement_name, performance] + map(
                        lambda key:self.ex_parameters[key], self.parameter_types)], columns=self.columns_of_tagwise, index=[datetime.now()])
                    performance_table.to_hdf(self.file_name + "_".join(self.parameter_types) + '.performance.h5', key='tag_wise', model='a', format='t', append=True,
                                             complib='blosc', min_itemsize=dict([("measurement", 7), ("tag", 52)] + [(k, 25) for k in map(lambda x:x[0], filter(lambda x:type(x[1]) == str, self.ex_parameters.items()))]))  # for other parameters use 10



class Tag_Propability_Visualizer(Callback):
    def __init__(self, ex_parameters, tag_list):
        super(Callback, self).__init__()
        self.NUM_CLASS = len(tag_list)
        self.ex_parameters = ex_parameters
        self.tag_list = tag_list

    def on_epoch_end(self, epoch, logs={}):
        if self.ex_parameters['CONTEXT'] != "None":
            if epoch % 50 == 0:
                if type(self.model.input) == list:
                    Feature_Size = int(self.model.input[0].shape[1])
                else:
                    Feature_Size = int(self.model.input.shape[1])
                if self.ex_parameters["single_output"]:
                    propagation_probability = self.model.predict(
                        [np.zeros((1, Feature_Size)), np.zeros((1, self.NUM_CLASS))], verbose=0)[-2]
                else:
                    propagation_probability = self.model.predict(
                        np.zeros((1, Feature_Size)), verbose=0)[-2]
                for tag_name, value in zip(self.tag_list, np.squeeze(propagation_probability)):
                    print '{:>18} {:>8.3f}'.format(tag_name, value)

def obtain_and_save_validation_result(model,ex_parameters,val_data,epoch=-1,save=True,code=None):
    audio_input,ground_truth = val_data
    predicted_tag_matrix = result_predict(model,ex_parameters,val_data)
    if ex_parameters['TAGGING_MODEL'] == "SampleCNN":
        if ex_parameters["MEAN"]=="geometric":
            predicted_tag_matrix = gmean(predicted_tag_matrix.reshape((10,-1,ex_parameters["NUM_CLASS"]),order= 'F'),axis = 0)
        else:
            predicted_tag_matrix = np.mean(predicted_tag_matrix.reshape((10,-1,ex_parameters["NUM_CLASS"]),order='F'),axis = 0)
    if save:
        save_object(predicted_tag_matrix, "saved_validation_tag_matrix/" + code + ".epoch." + str(epoch) + ".matrix")
    return predicted_tag_matrix
def get_scores(predicted_tag_matrix,ground_truth,task="retrieval",epoch=-1,save=True,code=None):
    # retrieval scores 
    if task == "retrieval":
        Pred = predicted_tag_matrix.T
        Grnd = ground_truth.T
    elif task == "annotation":
        Pred = predicted_tag_matrix
        Grnd = ground_truth
    precision_at_10 = multi_label_score(precision,10,Pred,Grnd,average=False)
    recall_at_10 = multi_label_score(recall,10,Pred,Grnd,average=False)
    MAP = multi_relevance_score(average_precision_score,Pred,Grnd,average=False)
    AUROC = multi_relevance_score(roc_auc_score,Pred,Grnd,average=False)
    scores = {"precision@10":precision_at_10,"recall@10":recall_at_10,"MAP":MAP,"AUROC":AUROC}
    if save and epoch!=-1:
        save_object(scores,"saved_validation_tag_matrix/" + code + ".epoch." + str(epoch) + ".scores")
    elif save:
        save_object(scores,"saved_validation_tag_matrix/" + code +  ".scores")
    return scores
def tag_wise_log_loss(y_true,y_pred,mode=None,eps=1e-15):
    y_pred = np.array(y_pred).astype(np.float64)
    y_pred[y_pred>=1.0] = 1.-eps
    y_pred[y_pred<=0.0] = eps
    if mode=="positive":
        return np.sum(-(y_true*np.log(y_pred)),axis=0)
    elif mode=="negative":
        return np.sum(-((1. - y_true)*np.log(1. - y_pred)),axis=0)
    else:
        return np.sum(-(y_true*np.log(y_pred) + (1. - y_true)*np.log(1. - y_pred)),axis = 0)
def configure_scores_list(scores_list):
    score_dict = {}
    for metric in scores_list[0].keys():
        score_dict[metric] = map(lambda sc:np.array(sc[metric]).tolist(),scores_list)
        score_dict[metric] = np.array(score_dict[metric])
    return score_dict
def generate_propagated_tag_matrix(train_tag_matrix,val_tag_matrix,ex_parameters,mode="validation"):  
    train_tids = map(lambda x: x.split("\n")[0],open('MSD_split_for_tagging/context_matched_train_x_msd_id.txt'))
    valid_tids = map(lambda x: x.split("\n")[0],open('MSD_split_for_tagging/valid_x_msd_id.txt'))
    if mode=="validation":
        tag_matrix = np.vstack([train_tag_matrix,val_tag_matrix])
        tids = train_tids+valid_tids
    elif mode=="train":
        tag_matrix = train_tag_matrix
        tids = train_tids
	stm = Similar_Track_Matcher(ex_parameters["CONTEXT"],tid_matcher=lambda
			x:x,playlist_source='aotm',fix_tid_list=tids,labeled_tid_list=None,random_select=True,reduced_dimension=None)
    while True:
        if mode == "validation":
            similar_tids = stm.get_similar_track(valid_tids)
        elif mode == "train":
            similar_tids = stm.get_similar_track(train_tids)
        propagated_tags = tag_matrix[npi.indices(tids,similar_tids),:]
        yield propagated_tags
class LR_Reducer():
    # TODO: wait longer before reducing learning rate 
    def __init__(self,patience=3, reduce_rate=0.2, reduce_nb=5, verbose=1):
        self.patience = patience
        self.wait = 0
        self.best_performance = 0.0
        self.reduce_rate = reduce_rate
        self.current_reduce_nb = 0
        self.reduce_nb = reduce_nb
        self.verbose = verbose
    def LR_Reducing(self,model,iteration,performance):
        current_performance = performance#round(loss,2)
        if current_performance > self.best_performance:
            self.best_performance = current_performance
            self.wait = 0
            if self.verbose > 0:
                print('---current best val performance: %.3f' % current_performance)
        else:  # Performance not increasing
            if self.wait >= self.patience:
                self.current_reduce_nb += 1
                if self.current_reduce_nb < self.reduce_nb:
                    lr = K.get_value(model.optimizer.lr)
                    K.set_value(model.optimizer.lr, lr * self.reduce_rate)
                else:
                    if self.verbose > 0:
                        print("iteration %d: early stopping" % (iteration))
                    model.stop_training = True
            self.wait += 1
class MonitoringTrainingAdaptor(Callback):
    def setting_propagated_tag_generator(self,train_tag_matrix,val_tag_matrix,ex_parameters,sample_count=5,mode="validation"):
        self.train_tids = map(lambda x: x.split("\n")[0],open('MSD_split_for_tagging/context_matched_train_x_msd_id.txt'))
        self.valid_tids = map(lambda x:x.split("\n")[0],open('MSD_split_for_tagging/valid_x_msd_id.txt'))
        if mode=="validation":
            self.tag_matrix = np.vstack([train_tag_matrix,val_tag_matrix])
            self.tids = self.train_tids+self.valid_tids
        elif mode=="train":
            self.tag_matrix = train_tag_matrix
            self.tids = self.train_tids
        if ex_parameters["CONTEXT"] == "playlist":
            CONTEXT = "naive_playlist"
        else:
            CONTEXT = ex_parameters["CONTEXT"]
        stm = Similar_Track_Matcher(CONTEXT,tid_matcher=lambda x:x,playlist_source='aotm',fix_tid_list=self.tids,labeled_tid_list=None,random_select=True,reduced_dimension=None)
        self.similar_tids_list = []
        for i in range(sample_count):
            if mode == "validation":
                similar_tids = stm.get_similar_track(self.valid_tids)
            elif mode == "train":
                similar_tids = stm.get_similar_track(self.train_tids)
            self.similar_tids_list.append(similar_tids)
    def generate_propagated_tag(self,similar_tids,mode="validation"):
        return self.tag_matrix[npi.indices(self.tids,similar_tids),:]
    def __init__(self, train_tag_data,
			validation_data,ex_parameters,steps_per_epoch,early_stopping_metric=("retrieval","MAP"),resolution=1,patience=3, reduce_rate=0.2, reduce_nb=5, verbose=1):
        super(Callback, self).__init__()
        self.train_tag_data = train_tag_data
        self.validation_data = validation_data
        self.ex_parameters = ex_parameters
        self.batch_per_monitoring = steps_per_epoch//resolution
        self.early_stopping_metric=early_stopping_metric

        self.lr_reducer = LR_Reducer(patience=patience*resolution)
        self.save = True
        self.setting_propagated_tag_generator(train_tag_data,validation_data[1],ex_parameters)
        self.scores_list = []
        self.annotation_scores_list = []
        self.losses_list = []
        self.output_attention_list = []
        self.existence_attention_list = []
        self.nonexistence_attention_list = []
        self.layer_wise_attention_list = []
        self.target_tag_wise_weights_list = []
        self.auxiliary_tag_wise_weights_list = []
    def on_batch_begin(self,batch,logs={}):
        if batch%self.batch_per_monitoring==0:
            self.monitoring(batch,logs)
    def auxiliary_loss(self,log_loss_function,predict):
        auxiliary_loss_list = []
        for similar_tids in self.similar_tids_list:
            propagated_ground_truth = self.generate_propagated_tag(similar_tids)
            auxiliary_loss_list.append(log_loss_function(propagated_ground_truth,predict))
        auxiliary_loss = sum(auxiliary_loss_list)/float(len(auxiliary_loss_list))#np.mean(auxiliary_loss_list)
        return auxiliary_loss
    def attention_monitoring(self):
        layer_wise_attention,output_attention = attention_generate(self.model, self.ex_parameters)
        if self.ex_parameters["LAYER_WISE_ATTENTION"]:
            self.layer_wise_attention_list.append(layer_wise_attention)
            print "layer-wise attention:",
            pprint.PrettyPrinter(indent=3).pprint(map(lambda x:round(x,3),layer_wise_attention))
        if self.ex_parameters["OUTPUT_ATTENTION"]:
            def printing(attentions,mode=""):
                print "top 5 output attention ("+mode+"):",
                pprint.PrettyPrinter(indent=3).pprint(map(lambda x:round(x,3),attentions[:5]))
                print "last 5 output attention ("+mode+"):",
                pprint.PrettyPrinter(indent=3).pprint(map(lambda x:round(x,3),attentions[-5:]))
                print "median:",np.median(attentions)
            if self.ex_parameters["BINARY_DEPENDENCE"]:
                existence_attention = output_attention[0]
                nonexistence_attention = output_attention[1] 
                self.existence_attention_list.append(existence_attention)
                self.nonexistence_attention_list.append(nonexistence_attention)
                printing(existence_attention,mode = 'existence')
                printing(nonexistence_attention,mode = 'non-existence')
            else:
                self.output_attention_list.append(output_attention)
                printing(output_attention,mode = '')
    def tag_wise_weight_monitoring(self,epoch,mode='train'):
        def printing(weights,mode='target'):
            print "top 5 tag-wise weight ("+mode+"):",
            pprint.PrettyPrinter(indent=3).pprint(map(lambda x:round(x,3),weights[:5]))
            print "last 5 tag-wise weight ("+mode+"):",
            pprint.PrettyPrinter(indent=3).pprint(map(lambda x:round(x,3),weights[-5:]))
            print "median:",np.median(weights),"max:",np.max(weights),"min:",np.min(weights)
        if mode=="valid":
            try:
                printing(self.target_tag_wise_weights,mode='target')
                self.target_tag_wise_weights_list.append(self.target_tag_wise_weights)
            except:
                pass 
            try:
                printing(self.auxiliary_tag_wise_weights,mode='auxiliary')
                self.auxiliary_tag_wise_weights_list.append(self.target_tag_wise_weights)
            except:
                pass 
        elif mode=="train":
            target_weight,auxiliary_weight = tagwise_weight_generate(self.model,self.ex_parameters)
            printing(target_weight,mode='target')
            printing(auxiliary_weight,mode='auxiliary')
    def scores_monitoring(self,epoch):
        ground_truth = self.validation_data[1]
        # 1. retrieval scores 
        
        scores = get_scores(self.predict,ground_truth,epoch=epoch,save=False)
        pprint.PrettyPrinter(indent=3).pprint(dict(map(lambda x:(x[0],np.mean(x[1])),scores.items())))
        # 2. annotation scores 
        annotation_scores=get_scores(self.predict,ground_truth,epoch=epoch,task="annotation",save=False)
        annotation_scores=dict(map(lambda x:(x[0],np.mean(x[1])),annotation_scores.items()))
        pprint.PrettyPrinter(indent=3).pprint(annotation_scores)
        self.scores_list.append(scores)
        self.annotation_scores_list.append(annotation_scores)
        return scores,annotation_scores
    def losses_monitoring(self,epoch):
        ground_truth = self.validation_data[1]
#if self.ex_parameters["BINARY_DEPENDENCE"]:
        existence_target_loss = tag_wise_log_loss(ground_truth,self.predict,mode="positive")
        target_loss = tag_wise_log_loss(ground_truth, self.predict)
        nonexistence_target_loss = target_loss - existence_target_loss
        existence_auxiliary_loss = self.auxiliary_loss(lambda y1,y2:tag_wise_log_loss(y1,y2,mode="positive"),self.predict)
        auxiliary_loss = self.auxiliary_loss(tag_wise_log_loss,self.predict)
        nonexistence_auxiliary_loss = auxiliary_loss - existence_auxiliary_loss
        print "existence target loss (val):",round(np.mean(existence_target_loss),4)
        print "nonexistence target loss (val):",round(np.mean(nonexistence_target_loss),4)
        print "existence auxiliary loss (val):",round(np.mean(existence_auxiliary_loss),4)
        print "nonexistence auxiliary loss (val):",round(np.mean(nonexistence_auxiliary_loss),4)
        print "target loss (val):",round(np.mean(target_loss),4)
        print "auxiliary loss (val):",round(np.mean(auxiliary_loss),4)

        losses = {"target_loss":target_loss,"auxiliary_loss":auxiliary_loss,"existence_target_loss":existence_target_loss,"existence_auxiliary_loss":existence_auxiliary_loss,"nonexistence_target_loss":nonexistence_target_loss,"nonexistence_auxiliary_loss":nonexistence_auxiliary_loss}
        self.losses_list.append(losses)
    def tag_wise_weight_tuning(self,epoch,T=5.,mo=0.9):
        # implement decrease-based scheme (more accurate according to paper) 
        # [v] No binary dependence , but there is task dependency (if propagation is considered) 
        # [v] for 0 loss (ex. alpha = 0.0) , fix weights as one 
        # NOTE FIX: poorly trained task should have larger loss, in order to be
# more focused. So, 1) The larger the loss is, the larger the weight should be. 2) The smaller the
# decreasing rate is, the larger the weight should be.  
        cur_ls = self.losses_list[-1]
        pre_ls = self.losses_list[-2]
        if self.ex_parameters["BINARY_DEPENDENCE"]:
            model_auxiliary_tag_weight = self.model.inputs[-5]
            model_target_tag_weight = self.model.inputs[-6]
        elif self.ex_parameters["OUTPUT_ATTENTION"]:
            model_auxiliary_tag_weight = self.model.inputs[-3]
            model_target_tag_weight = self.model.inputs[-4]
        else:
            model_auxiliary_tag_weight = self.model.inputs[-1]
            model_target_tag_weight = self.model.inputs[-2]
            # [-5] -> auxiliary tag-wise weight 
            # [-6] -> target tag-wise weight 
            #else:
            # [-3] -> auxiliary tag-wise weight 
            # [-4] -> target tag-wise weight 
        d_tl = cur_ls['target_loss']/pre_ls['target_loss']/T # the d_tl1 is the inverse of slope 
        try:
            self.target_d_tl = self.target_d_tl*mo + d_tl*(1.-mo) 
        except:
            self.target_d_tl = d_tl
        # apply softmax 
        w = np.exp(self.target_d_tl)
        w = w/np.sum(w)*self.ex_parameters["NUM_CLASS"]
#try:
#  self.target_tag_wise_weights = self.target_tag_wise_weights*mo+w*(1.-mo)
# except:	
        self.target_tag_wise_weights = w
        # assign to input values 
        K.set_value(model_target_tag_weight,np.expand_dims(self.target_tag_wise_weights,axis=0))
        if self.ex_parameters['alpha'] != 0.0:
            d_tl = cur_ls['auxiliary_loss']/pre_ls['auxiliary_loss']/T # the d_tl1 is the inverse of slope 
            try:
                self.auxiliary_d_tl = self.auxiliary_d_tl*mo + d_tl*(1.-mo)
            except:
                self.auxiliary_d_tl = d_tl
            # apply softmax 
            w = np.exp(self.auxiliary_d_tl)
            w = w/np.sum(w)*self.ex_parameters["NUM_CLASS"]    
            # assign to input values 
#try:
#           self.auxiliary_tag_wise_weights = self.auxiliary_tag_wise_weights*mo+w*(1.-mo)
#   except:
            self.auxiliary_tag_wise_weights = w
            K.set_value(model_auxiliary_tag_weight,np.expand_dims(self.auxiliary_tag_wise_weights,axis=0))
    def attention_tuning(self,epoch,T=1.):
        cur_ls = self.losses_list[-1] 
        pre_ls = self.losses_list[-2] 
        # notes loss decrease rate => softmax => attention weight 
        if self.ex_parameters["BINARY_DEPENDENCE"]:
            if self.ex_parameters["DECREASE_BASED"]:
                d_tl1 = pre_ls['existence_target_loss']/cur_ls['existence_target_loss']/T
                d_al1 = pre_ls['existence_auxiliary_loss']/cur_ls['existence_auxiliary_loss']/T
                d_tl2 = pre_ls['nonexistence_target_loss']/cur_ls['nonexistence_target_loss']/T
                d_al2 = pre_ls['nonexistence_auxiliary_loss']/cur_ls['nonexistence_auxiliary_loss']/T
                a1 = 1./(1.+np.exp(d_tl1-d_al1))#d_al1/(d_tl1+d_al1)
                a2 = 1./(1.+np.exp(d_al1-d_tl1))#d_al2/(d_tl2+d_al2)
            else:
                tl1 = cur_ls['existence_target_loss']/T
                al1 = cur_ls['existence_auxiliary_loss']/T
                tl2 = cur_ls['nonexistence_target_loss']/T
                al2 = cur_ls['nonexistence_auxiliary_loss']/T
                a1 = 1./(1.+np.exp(al1-tl1))
                a2 = 1./(1.+np.exp(al2-tl2))
            t1 = 1. - a1
            t2 = 1. - a2
            K.set_value(self.model.inputs[-1],np.expand_dims(a1,axis=0))
            K.set_value(self.model.inputs[-2],np.expand_dims(t1,axis=0))
            K.set_value(self.model.inputs[-3],np.expand_dims(a2,axis=0))
            K.set_value(self.model.inputs[-4],np.expand_dims(t2,axis=0))
            # self.model.inputs[-1] -> axuiliary attention 1 (for exist tag)
            # self.model.inputs[-2] -> target attention 1    (for exist tag)
            # self.model.inputs[-3] -> axuiliary attention 2 (for non-exist tag)
            # self.model.inputs[-4] -> target attention 2    (for non-exist tag)
        else:
            if self.ex_parameters["DECREASE_BASED"]: 
                d_tl = pre_ls['target_loss']/cur_ls['target_loss']/T
                d_al = pre_ls['auxiliary_loss']/cur_ls['auxiliary_loss']/T
                a = 1./(1.+np.exp(d_tl-d_al))#d_al/(d_tl+d_al)
            else:
                tl = cur_ls['target_loss']/T
                al = cur_ls['auxiliary_loss']/T
                a = 1./(1.+np.exp(al-tl))
            t = 1. - a
            K.set_value(self.model.inputs[-1],np.expand_dims(a,axis=0))
            K.set_value(self.model.inputs[-2],np.expand_dims(t,axis=0))
            # self.model.inputs[-1] -> axuiliary attention 1 (for exist tag)
            # self.model.inputs[-2] -> target attention 1    (for exist tag)
    def monitoring(self, epoch, logs={}):
        # Monitoring Attention
        self.tag_wise_weight_monitoring(epoch,mode='train')# only show when validation tuning is used. 
        if self.ex_parameters["OUTPUT_ATTENTION"] or self.ex_parameters["LAYER_WISE_ATTENTION"]:
            self.attention_monitoring()
        # Monitoring Scores and Losses 
        self.predict = obtain_and_save_validation_result(self.model,self.ex_parameters,self.validation_data,epoch=epoch,save=False)
        scores,annotation_scores = self.scores_monitoring(epoch)
        # 3. losses 
        self.losses_monitoring(epoch)
        # 4. tuning attention based on losses 
        if self.ex_parameters["OUTPUT_ATTENTION"] and self.ex_parameters["VALID_TUNED"]:
            if len(self.losses_list)>=2:
                self.attention_tuning(epoch)
            else:
                print "No previous loss for attention calculation"
        if self.ex_parameters["TAG_WISE_WEIGHTING"] and self.ex_parameters["VALID_TAG_WISE_WEIGHTING"]:
            if len(self.losses_list)>=2:
                self.tag_wise_weight_tuning(epoch)
        # Do Early Stopping 
        if epoch>1:
            if self.early_stopping_metric[0] == "retrieval":
                self.lr_reducer.LR_Reducing(self.model,epoch,np.mean(scores[self.early_stopping_metric[1]]))
            elif self.early_stopping_metric[0]=="annotation":
                self.lr_reducer.LR_Reducing(self.model,epoch,annotation_scores[self.early_stopping_metric[1]])
    def save_result(self):
        code = str(datetime.now())
        save_object(self.ex_parameters, "saved_monitoring_result/" + code + ".ex_parameters")
        if self.output_attention_list:
            save_object(self.output_attention_list, "saved_monitoring_result/" + code + ".output_attention")
        if self.existence_attention_list:
            save_object(self.existence_attention_list, "saved_monitoring_result/" + code + ".existence_attention")
        if self.nonexistence_attention_list:
            save_object(self.nonexistence_attention_list, "saved_monitoring_result/" + code + ".nonexistence_attention")
        if self.layer_wise_attention_list:
            save_object(self.layer_wise_attention_list, "saved_monitoring_result/" + code + ".layer_wise_attention")
        if self.target_tag_wise_weights_list:
            save_object(self.target_tag_wise_weights_list, "saved_monitoring_result/" + code + ".target_tag_wise_weights")
        if self.auxiliary_tag_wise_weights_list:
            save_object(self.auxiliary_tag_wise_weights_list, "saved_monitoring_result/" + code + ".auxiliary_tag_wise_weights")
        save_object(configure_scores_list(self.scores_list),"saved_monitoring_result/" + code + ".scores")
        save_object(configure_scores_list(self.annotation_scores_list),"saved_monitoring_result/" + code + ".annotation_scores")
        save_object(configure_scores_list(self.losses_list),"saved_monitoring_result/" + code + ".losses")


class Tag_Propability_Visualizer(Callback):
    def __init__(self, ex_parameters, tag_list):
        super(Callback, self).__init__()
        self.NUM_CLASS = len(tag_list)
        self.ex_parameters = ex_parameters
        self.tag_list = tag_list
    def on_epoch_end(self, epoch, logs={}):
        if self.ex_parameters['CONTEXT'] != "None":
            if epoch % 50 == 0:
                if type(self.model.input) == list:
                    Feature_Size = int(self.model.input[0].shape[1])
                else:
                    Feature_Size = int(self.model.input.shape[1])
                if self.ex_parameters["ground_truth_based_propagation_constrain"]:
                    propagation_probability = self.model.predict(
                        [np.zeros((1, Feature_Size)), np.zeros((1, self.NUM_CLASS))], verbose=0)[-2]
                else:
                    propagation_probability = self.model.predict(
                        np.zeros((1, Feature_Size)), verbose=0)[-2]
                for tag_name, value in zip(self.tag_list, np.squeeze(propagation_probability)):
                    print '{:>18} {:>8.3f}'.format(tag_name, value)
