

# TODO: Performance on Training Data or Testing Data
# TODO: expand the tags (no tag removal now! )


# about 5 days
# train_batch_generation_object.matcher.tid_matcher = lambda x:x
# len(train_batch_generation_object.matcher.get_similar_track(train_tids[:10]))
# train_batch_generation_object.matcher.tid_song_id_table.loc[train_tids[:96]].song_id
# len(train_batch_generation_object.matcher.tid_list) -> 120381
# len(train_batch_generation_object.matcher.tid_song_id_table)
# len(train_batch_generation_object.matcher.song_id_tid_table) # same song id can match to two different tid?
# train_batch_generation_object.matcher.neighbor_table.loc[list(train_batch_generation_object.matcher.tid_song_id_table.loc[train_tids[:96]].song_id)].neighbor_list
# train_batch_generation_object.groundtruth_tag_matcher.match_one_by_one(train_tids[:11])
# train_batch_generation_object.groundtruth_tag_batch_convertor(train_tids[:11])


'''
make sure the returning neighbor songs are all in the constrain and not stopping ! 
from Batch_Generation_Class import *
CONTEXT = 'artist' 
content_retriever = FeatureRetriever()
train_tids = list(content_retriever.check_saved_tid() & set(context_match_tids(['tag','artist','webpage','lyrics','listen','playlist'])))
#labeled_train_tids = train_tids[:100] 
labeled_train_tids = None
train_batch_generation_object = Batch_Generation_Class(
                content_retriever, context = CONTEXT, fix_tid_list = train_tids,random_select=True,labeled_tid_list = labeled_train_tids) 
train_batch_generation_object.matcher.tid_matcher = lambda x:x 
for i in range(100):
    assert len(set(train_batch_generation_object.matcher.get_similar_track(train_tids[:100]))-set(labeled_train_tids))== 0 
        assert train_batch_generation_object.matcher.get_similar_track([id])[0] in labeled_train_tids
train_batch_generation_object.matcher.get_similar_track(train_tids[:100])
train_batch_generation_object.matcher.labeled_tid_list
# what happen? so slow 

train_batch_generation_object.matcher.get_similar_track(train_tids[100:200]).count(None)
# Problem : 
# problem: : many tid of unlabeled cannot be match to a labeled !!! what the hell!!! (ofcource because only a small amount of labeled song !! ) (if all the similar or word overlap songs of a target song is not a labeled song , then the matcher cannot obtain the similar song!! ) 
# how to solve this problem ? random select a labeled song  
# if the labeled song number increase , will the matchable song number increase ?  





# artist: 
# unlabeled input error # if input an unlabeled song, what will the code do ? 
# webpage : labeled input error : many none ! 
# 
# limit label size in train_tids => get unlabeled training tids => only 
#unlabeled_train_tids = list(set(list(content_retriever.check_saved_tid() & set(context_match_tids(['artist','webpage','lyrics','listen','playlist']))))-set(labeled_train_tids)) 
# only 3781 unlabeled training tids (too small) 
# how to reduce ? set a constrain on tag matcher, so that some of the tids do not match with any tags (give 000000 tag) 
# Problem: 
# For NAN TID Generated from the Similar TID Matching Module, I random select a tid to replace them so that tag distribution still remain! 
# np.sum(train_batch_generation_object.neighbor_tag_batch_convertor(unlabeled_train_tids[:100],decay=0.8))
# train_batch_generation_object.neighbor_tag_matcher.match_one_by_one(unlabeled_train_tids[:100])
# train_batch_generation_object.neighbor_tag_matcher.unlabeled_tid_list
'''
'''




# TODO : webpage : after 255 itertation , training stop!!! 





# Test Batch Generator 
next(train_batch_generator)
# Two output One input !! 

# need to enable two kind of training scheme ! 1. two way fit 2. one way fit ! 

# set up temp training data : match with both raw tids and tags 

# build training data batch generator : get_generator(self, tid_list, batch_size)
'''

# CONTROL PARAMETERS:
import pandas as pd
from Batch_Generation_Class import *
from Evaluation import *
#from music_taggers import MusicTaggerLR
from keras import optimizers, metrics
from music_taggers import MusicTaggerLR, MusicTaggerCNN, MusicTaggerCRNN, MusicTaggerSampleCNN
from keras.callbacks import Callback, ModelCheckpoint


def get_initial_epoch(ex_parameters):
    # INPUT : a dict with parameters specifying a experiement model
    # OUTPUT : the last epoch trained for the model
    try:
        performance_table = pd.read_hdf(
            "_".join(list(ex_parameters.keys())) + '.performance.h5')
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


def link_to_checkpoint(model, ex_parameters, load=False):
    check_point_directory = "/data/jeffrey82221/tag_models/"
    if load:
        model.load_weights(check_point_directory +
                           model_name(ex_parameters) + ".hdf5")
        print "model weight LOAD"
    checkpoint = ModelCheckpoint(check_point_directory + model_name(ex_parameters) + ".hdf5",
                                 monitor=None,
                                 verbose=1,
                                 save_best_only=False,
                                 mode='max',
                                 period=1)
    return checkpoint


#labeled_train_tids = train_tids[:10000]
#unlabeled_train_tids = train_tids[10000:]
# train_batch_generation_object = Batch_Generation_Class(
#                    content_retriever, context = CONTEXT, fix_tid_list = train_tids,random_select=True,labeled_tid_list = labeled_train_tids)
#CONTEXT = 'webpage'

# TODO : save the evaluation on testing songs only !

NUM_CLASS = 145
EPOCHS = 100
train_tids = list(FeatureRetriever().cleaned_saved_tids() & set(context_match_tids(
    ['tag', 'artist', 'webpage', 'lyrics', 'listen', 'playlist'])))
# evaluation setup
#train_tids = list(train_tids[2000:]+train_tids[:2000])


PAIRS_PER_EPOCH = len(train_tids)
print "PAIRS_PER_EPOCH",PAIRS_PER_EPOCH
BATCH_SIZE = 100 # 100 is too large for Sample Based Method! 
for CONTEXT in ['artist',None, 'playlist', 'webpage', 'listen', 'lyrics']:
    # get max epoch
    for TAGGING_MODEL in ['LR',"SampleCNN"]:
        ex_parameters = {"TAGGING_MODEL": TAGGING_MODEL, "CONTEXT": CONTEXT}
        INITIAL_EPOCH = get_initial_epoch(ex_parameters)
        # get saved model name
        if TAGGING_MODEL == "LR":
            model = MusicTaggerLR(1216, NUM_CLASS, PROPAGATION=CONTEXT!=None)
            content_retriever = FeatureRetriever()
        elif TAGGING_MODEL == "SampleCNN":
            model = MusicTaggerSampleCNN(59049, NUM_CLASS,PROPAGATION=CONTEXT!=None)
            content_retriever = RawRetriever()
        elif TAGGING_MODEL == "CNN" or TAG_MODEL == "CRNN":
            if TAGGING_MODEL == "CNN":
                model = MusicTaggerCNN(NUM_CLASS, weights=None,PROPAGATION=CONTEXT!=None)
            elif TAGGING_MODEL == "CRNN":
                model = MusicTaggerCRNN(NUM_CLASS, weights=None,PROPAGATION=CONTEXT!=None)
            content_retriever = MelgramRetriever()

        adam = optimizers.Adam(lr=0.005)
        if CONTEXT!=None:
            print "two output"
            #model.compile(loss=["binary_crossentropy","kullback_leibler_divergence",None],
            #              optimizer=adam, metrics=[metrics.binary_accuracy])
            model.compile(loss=["kullback_leibler_divergence","kullback_leibler_divergence",None],
                          optimizer=adam, metrics=[metrics.binary_accuracy])
        else:
            print "one output"
            model.compile(loss=["binary_crossentropy"],
                          optimizer=adam, metrics=[metrics.binary_accuracy])
        model.summary()
        train_batch_generation_object = Batch_Generation_Class(
            content_retriever, context=CONTEXT, fix_tid_list=train_tids, random_select=True)
        train_batch_generator = train_batch_generation_object.get_generator(
            train_tids, batch_size=BATCH_SIZE, decay=0.8)
        train_data = Batch_Generation_Class(content_retriever, context=None).get_data(train_tids) # about 0.2 of training data is ok ! , check if the this large test data drain the memory ? 
        #print "test model is ok:", model.predict(train_data[0])
        checkpoint = link_to_checkpoint(
            model, ex_parameters, load=INITIAL_EPOCH != 0)
        #test_data = test_batch_generation_object.get_data(unlabeled_train_tids)
        ival = IntervalEvaluation(
            eval_data={"train_data": train_data},
            tag_list=map(str, train_batch_generation_object.tag_list),
            ex_parameters=ex_parameters)
        model.fit_generator(train_batch_generator,
                            steps_per_epoch=PAIRS_PER_EPOCH // BATCH_SIZE,
                            epochs=EPOCHS,
                            verbose=1,
                            callbacks=[ival,checkpoint],
                            initial_epoch=INITIAL_EPOCH,
                            # validation_data=test_data,
                            use_multiprocessing=False
                            )

'''
for CONTEXT in ['artist', 'playlist', 'webpage', 'listen', 'lyrics']:
    if CONTEXT == "artist":
        K_range = [0]
    elif CONTEXT == "playlist":
        K_range = range(0, 5)
    else:
        # 10 propagation distance for playlist may be too large
        K_range = range(0, 5)
    for K in K_range:
        # for TAG_MODEL in ['LR',"CNN","CRNN","SampleCNN"]: #  "LR" # "CNN", # "CRNN", # "SampleCNN"
        for TAG_MODEL in ['LR']:
            saved_model_name = CONTEXT + str(K) + TAG_MODEL
            performance_table = pd.read_hdf('performance_table.h5')
            if sum(list(performance_table['TAG_MODEL'] == TAG_MODEL) & (performance_table['K'] == K) & (performance_table['CONTEXT'] == CONTEXT)) > 0:
                # a model with the setting is trained before
                max_epoch = max(performance_table[(performance_table['TAG_MODEL'] == TAG_MODEL) & (
                    performance_table['K'] == K) & (performance_table['CONTEXT'] == CONTEXT)].EPOCH)
                # max epoch saved in the table for specific model with fix K, fix context and fix tag model
                INITIAL_EPOCH = max_epoch + 1
                print "Training of ", saved_model_name, "START FROM EPOCH", INITIAL_EPOCH
            else:
                INITIAL_EPOCH = 0
                print "Training of ", saved_model_name, "START FROM EPOCH", 0
            # FIX PARAMETERS:
            check_point_directory = "/data/jeffrey82221/tag_models/"
            EPOCHS = 100
            BATCH_SIZE = 10
            NUM_CLASS = 145
            if INITIAL_EPOCH >= EPOCHS:  # skip if epochs done
                continue
            # LOAD PREPROCESSING AND BATCH GENERATOR
            train_tids = generate_training_tids()
            PAIRS_PER_EPOCH = len(train_tids)
            print "train tid count:", PAIRS_PER_EPOCH
            if TAG_MODEL == "LR":
                AUDIO_FEATURE_NUM = 1216
                audio_retriever = FeatureRetriever()
            elif TAG_MODEL == "CNN" or TAG_MODEL == "CRNN":
                audio_retriever = MelgramRetriever()
            elif TAG_MODEL == "SampleCNN":
                SEGMENT_LEN = 59049
                audio_retriever = RawRetriever()
            audio_retriever.open()
            train_batch_generation_object = Batch_Generation_Class(
                context=CONTEXT, content_retriever=audio_retriever, random_select=True)
            train_batch_generator = train_batch_generation_object.get_generator(
                train_tids, batch_size=BATCH_SIZE)
            # RESTORE MODEL
            if TAG_MODEL == "LR":
                model = MusicTaggerLR(AUDIO_FEATURE_NUM, NUM_CLASS)
                # model = MusicTaggerLR(1000, 250)
            elif TAG_MODEL == "CNN":
                model = MusicTaggerCNN(NUM_CLASS, weights=None)
            elif TAG_MODEL == "CRNN":
                model = MusicTaggerCRNN(NUM_CLASS, weights=None)
            elif TAG_MODEL == "SampleCNN":
                model = MusicTaggerSampleCNN(59049, NUM_CLASS)
            adam = optimizers.Adam(lr=0.005)
            model.compile(loss="binary_crossentropy",
                          optimizer=adam, metrics=[metrics.binary_accuracy])
            model.summary()
            if sum(list(performance_table['TAG_MODEL'] == TAG_MODEL) & (performance_table['K'] == K) & (performance_table['CONTEXT'] == CONTEXT)) > 0:
                model.load_weights(check_point_directory +
                                   saved_model_name + ".hdf5")
            # EVALUATION
            test_batch_generation_object = Batch_Generation_Class(
                context=None, content_retriever=audio_retriever)
            test_tids = generate_saved_testing_tids()
            test_data = test_batch_generation_object.get_data(test_tids)
            tag_list = map(str, test_batch_generation_object.tag_list)
            print "test data generated", len(test_tids)
            train_data = test_batch_generation_object.get_data(train_tids)
            ival = IntervalEvaluation(
                train_data, test_data, tag_list, TAG_MODEL, K, CONTEXT)
            # STORE MODEL
            checkpoint = ModelCheckpoint(check_point_directory + saved_model_name + ".hdf5",
                                         monitor=None,
                                         verbose=1,
                                         save_best_only=False,
                                         mode='max',
                                         period=1)
            # TRAINING
            model.fit_generator(train_batch_generator,
                                steps_per_epoch=PAIRS_PER_EPOCH // BATCH_SIZE,
                                epochs=EPOCHS,
                                verbose=1,
                                callbacks=[ival, checkpoint],
                                initial_epoch=INITIAL_EPOCH,
                                validation_data=test_data,
                                use_multiprocessing=False
                                )
'''
