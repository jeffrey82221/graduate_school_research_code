
from keras import optimizers, metrics
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score, average_precision_score
from pprint import pprint
from datetime import datetime
import numpy as np
import pandas as pd


class IntervalEvaluation(Callback):
    def __init__(self, eval_data, tag_list, ex_parameters):
        '''
        INPUT:
        1. eval_data : a dict with key as data name and item as data
        2. tag_list :  an array with str representing each tag
        3. save_parameter : a dict with key to be the type of a parameters (ex: CONTEXT, decay, AUTO_TAGGING_MODEL) and values the parameter (ex. playlist, 0.7, CNN)
        '''
        self.eval_data = eval_data
        self.tag_list = tag_list
        self.ex_parameters = ex_parameters
        self.parameter_types = list(self.ex_parameters.keys())
        self.columns = ['EPOCH', 'tag', 'measurement',
                        'performance'] + self.parameter_types
        self.measurements = {"AUC-ROC": roc_auc_score,
                             "MAP": average_precision_score}

    def on_batch_begin(self, batch, logs={}):
        if self.ex_parameters['CONTEXT'] != None:
            if batch % 1 == 0:
                predict, neighbor_predict, propagation_probability = self.model.predict(
                    self.eval_data['train_data'][0][:1], verbose=0)
                #predict, neighbor_predict, propagation_probability = self.model.predict([],verbose=0)
                # print "sameness:",np.sum((predict[0,:]/neighbor_predict[0,:]) - (predict[1,:]/neighbor_predict[1,:]))
                # print propagation_probability.shape
                # print tag_propability[0,:]
                for tag_name, value in zip(self.tag_list, np.squeeze(propagation_probability)):
                    print '{:>18} {:>8.3f}'.format(tag_name, value)
            
            '''
            tag_propagation_constant = np.sum(
                (neighbor_predict / predict), axis=0) / float(predict.shape[0])
            '''
        # print predict.shape,np.sum(predict),np.sum(self.eval_data['train_data'][1]) # more and more fit!?
        # problem: why LR is OK but sampleCNN is not ? not the multiple output problem (happend for Traditional SampleCNN , too)
        # first error 540 , second 540
        # what happend when generate batch for train_id 5400?
        # problem: which layer cause the problem ?

    def on_epoch_end(self, epoch, logs={}):
        # printing :
        pprint(self.ex_parameters)
        print "EPOCH:", epoch, ""
        for test_data_name in self.eval_data.keys():
            ground_truth = self.eval_data[test_data_name][1]
            audio_input = self.eval_data[test_data_name][0]
            if self.ex_parameters['CONTEXT'] == None:
                predict = self.model.predict(audio_input, verbose=0)
            else:
                predict = self.model.predict(audio_input, verbose=0)[0]
            for measurement_name in self.measurements.keys():
                print test_data_name, measurement_name,
                print self.measurements[measurement_name](ground_truth, predict)
                # save result
                for i in range(len(self.tag_list)):
                    performance = self.measurements[measurement_name](
                        ground_truth[:, i], predict[:, i])
                    performance_table = pd.DataFrame([[epoch, self.tag_list[i], measurement_name, performance] + map(
                        lambda key:self.ex_parameters[key], self.parameter_types)], columns=self.columns, index=[datetime.now()])
                    performance_table.to_hdf("_".join(self.parameter_types) + '.performance.h5', key=test_data_name, model='a', format='t', append=True,
                                             complib='blosc', min_itemsize=dict([("measurement", 7), ("tag", 18)] + [(k, 10) for k in map(lambda x:x[0], filter(lambda x:type(x[1]) == str, self.ex_parameters.items()))]))  # for other parameters use 10
        # TODO:
        # only save the testing data
