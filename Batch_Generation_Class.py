from Source_Matcher import *
import random
from itertools import cycle
import itertools as it

'''
def get_raw_matchable_tids():
    filenames = listdir("/data/jeffrey82221/MSD_Audio_Raw")
    tid_map_dict = dict(map(lambda x: (
        x.split(".")[0], "/data/jeffrey82221/MSD_Audio_Raw/" + x), filenames))
    tid_list = list(self.tid_map_dict.keys())
    return tid_list
'''

# (['tag','artist','webpage','lyrics','listen','playlist']


def context_match_tids(CONTEXTS):
    # read overlapped tids
    try:
        f = open("&".join(CONTEXTS) + ".overlap_tids", "r")
        merge_tids = json.loads(f.read())
        f.close()
        print "Overlap Length:", len(merge_tids), "LOAD"
        print "======================================="
        return merge_tids
    except:
        for i, CONTEXT in enumerate(CONTEXTS):

            if CONTEXT == 'tag':
                matcher = Tag_Matcher(None)
            else:
                matcher = Similar_Track_Matcher(
                    CONTEXT, playlist_source='aotm')
            print "Matched Length of ", CONTEXT, len(matcher.tid_list)
            if i == 0:
                merge_tids = list(set(matcher.tid_list))
            else:
                merge_tids = list(set(merge_tids) & set(matcher.tid_list))
            print "Overlap Length:", len(merge_tids)
            print "======================================="
        # save overlapped
        f = open("&".join(CONTEXTS) + ".overlap_tids", "w")
        f.write(json.dumps(merge_tids, ensure_ascii=False))
        f.close()
        return merge_tids


'''
from Batch_Generation_Class import *
target_tids = context_match_tids(['tag','artist','webpage','lyrics','listen','playlist'])
target_tids = context_match_tids(['artist','webpage','lyrics','listen','playlist'])
'''


def save_audio_with_context(audio_retriever):
    '''
    # STEP.1 : Save Audio into Table  
    # 1. in context & in tags => training labeled songs 
    # 2. in context  => testing labeled songs 
    '''
    #audio_retriever = RawRetriever()
    audio_retriever.save_target_tids(target_tids=context_match_tids(
        ['artist', 'webpage', 'lyrics', 'listen', 'playlist']))
    print "Audio with Context Save"
    #audio_retreiver = MelgramRetriever()
    # audio_retriever.save_target_tids(target_tids=context_match_tids(
    #    ['artist', 'webpage', 'lyrics', 'listen', 'playlist']))
    # print "Mel Audio with Context Save"
    #audio_retriever = FeatureRetriever()
    # audio_retriever.save_target_tids(target_tids=context_match_tids(
    #    ['artist', 'webpage', 'lyrics', 'listen', 'playlist']))
    # print "Feature Audio with Context Save"
def save_test_audio_with_tag(audio_retriever,training_tids,rate = 0.2):
    tid_with_tag = context_match_tids(['tag']) 
    testing_tids = list(set(tid_with_tag) - set(training_tids))
    testing_tids = testing_tids[:int(float(len(training_tids))*rate)] 
    print "target testing tid size:",len(testing_tids) 
    audio_retriever.save_target_tids(target_tids=testing_tids)
    print "Testing Audio with Tag Save"

'''
# Saving Training Audio Data  
from Batch_Generation_Class import * 
save_audio_with_context(RawRetriever())
save_audio_with_context(MelgramRetriever()) # save multiple time so that the error may be reduce ! 
save_audio_with_context(FeatureRetriever())


# Saving Testing Audio Data  
from Batch_Generation_Class import * 
content_retriever = FeatureRetriever()
train_tids = list(content_retriever.check_saved_tid() & set(context_match_tids(['tag','artist','webpage','lyrics','listen','playlist'])))
save_test_audio_with_tag(RawRetriever(),train_tids,rate = 0.2)
save_test_audio_with_tag(MelgramRetriever(),train_tids,rate = 0.2)
save_test_audio_with_tag(FeatureRetriever(),train_tids,rate = 0.2)
'''
'''
# labeled 
audio_retriever = RawRetriever() 
audio_retriever.save_training_tids(target_tids = context_match_tids(['tag','artist','webpage','lyrics','listen','playlist']))
audio_retreiver = MelgramRetriever()
audio_retriever.save_training_tids(target_tids = context_match_tids(['tag','artist','webpage','lyrics','listen','playlist']))
audio_retriever = FeatureRetriever()
audio_retriever.save_training_tids(target_tids = context_match_tids(['tag','artist','webpage','lyrics','listen','playlist']))

# unlabeled 
from Batch_Generation_Class import *
audio_retriever = RawRetriever() 
audio_retriever.save_training_tids(target_tids = context_match_tids(['artist','webpage','lyrics','listen','playlist']))
audio_retreiver = MelgramRetriever()
audio_retriever.save_training_tids(target_tids = context_match_tids(['artist','webpage','lyrics','listen','playlist']))
audio_retriever = FeatureRetriever()
audio_retriever.save_training_tids(target_tids = context_match_tids(['artist','webpage','lyrics','listen','playlist']))
# Testing Dataset : 
audio_retriever.save_training_tids(target_tids = context_match_tids(['tag'])) 
'''


# match directly from context matchable and + content , option: with unlabeled and size
def generate_training_tids(unlabeled=None):
    '''
    STEP.3 Generate Training Dataset 
    1. no unlabeled songs (LONG TAIL) 
    2. with unlabeled songs (SSL) 
    # RULE: 
    # sub RULE for 1. 
    # 1. match to all context 
    # 2. match to tags 
    # 3. match to content 
    # sub RULE for 2. 
    # 1. - label songs + unlabeled songs  
    # 1.1. labeled songs 
    # 1.1.1 match to all context  
    # 1.1.2 match to tags  
    # 1.1.3 match to content  
    # 1.2. unlabeled songs  
    # 1.2.1 match to all context  
    # 1.2.2 match to content 
    '''
    # generate labeled songs
    labeled_target_tids = context_match_tids(
        ['tag', 'artist', 'webpage', 'lyrics', 'listen', 'playlist'])
    audio_retriever = RawRetriever()
    labeled_target_tids = list(set(labeled_target_tids) & set(
        audio_retriever.check_saved_tid()))
    audio_retriever = MelgramRetriever()
    labeled_target_tids = list(set(labeled_target_tids) & set(
        audio_retriever.check_saved_tid()))
    audio_retriever = FeatureRetriever()
    labeled_target_tids = list(set(labeled_target_tids) & set(
        audio_retriever.check_saved_tid()))
    labeled_training_tids = labeled_target_tids
    # already match to all context and content
    if unlabeled != None:
        # no need to match to tags
        extend_tids = context_match_tids(
            ['artist', 'webpage', 'lyrics', 'listen', 'playlist'])
        audio_retriever = RawRetriever()
        extend_tids = list(set(extend_tids) & set(
            audio_retriever.check_saved_tid()))
        audio_retriever = MelgramRetriever()
        extend_tids = list(set(extend_tids) & set(
            audio_retriever.check_saved_tid()))
        audio_retriever = FeatureRetriever()
        extend_tids = list(set(extend_tids) & set(
            audio_retriever.check_saved_tid()))
        unlabeled_training_tids = list(
            set(extend_tids) - set(labeled_training_tids))
        # sample the unlabeled tids
        unlabeled_training_tids = numpy.random.choice(unlabeled_training_tids, size=int(
            float(len(unlabeled_training_tids)) * unlabeled), replace=False).tolist()
        # combining the labeled and unlabeled training tids
        return labeled_training_tids + unlabeled_training_tids
    else:
        return labeled_training_tids


# save testing audio

# generate testing tids


def generate_broad_testing_tids():
    tid_list_dict = dict()
    filenames = listdir("/data/jeffrey82221/MSD_Audio_Raw")
    tid_map_dict = dict(map(lambda x: (
        x.split(".")[0], "/data/jeffrey82221/MSD_Audio_Raw/" + x), filenames))
    mel_gram_matchable_tids = set(tid_map_dict.keys())
    print "melgram matchable  tid count:", len(mel_gram_matchable_tids)
    feature_matchable_tids = set(get_audio_matchable_tids())
    print "audio feature matchable tid count:", len(feature_matchable_tids)
    audio_matchable_tids = mel_gram_matchable_tids & feature_matchable_tids
    print "audio matchable tid count:", len(audio_matchable_tids)
    tag_matcher = Tag_Matcher(None)
    print "tag matchable tid count:", len(tag_matcher.tid_list)
    audio_tag_matchable_tids = set(
        audio_matchable_tids) & set(tag_matcher.tid_list)
    print "tag audio matchable tid count:", len(audio_tag_matchable_tids)
    # remove from audio feature
    f = open(u"training_tids", u"r")
    training_tids = json.loads(f.read())
    print "training tid count:", len(training_tids)
    f.close()
    testing_tids = list(audio_tag_matchable_tids - set(training_tids))
    print "testing tid count:", len(testing_tids)
    return testing_tids


def generate_saved_testing_tids():
    training_tids = generate_training_tids()
    testing_tids = generate_broad_testing_tids()
    mel_gram_retriever = MelgramRetriever()
    audio_feature_retriever = AudioFeatureRetriever()
    raw_retriever = RawRetriever()
    mel_gram_test_set = mel_gram_retriever.check_saved_test_tid(training_tids)
    audio_feature_test_set = audio_feature_retriever.check_saved_test_tid(
        training_tids)
    raw_test_set = raw_retriever.check_saved_test_tid(training_tids)
    testing_tids_saved = list(set(mel_gram_test_set) & set(
        audio_feature_test_set) & set(raw_test_set))
    assert len(set(testing_tids_saved) - set(testing_tids)) == 0
    return testing_tids_saved
    # if not enough number for testing , continue crawling:


def save_training_audio_into_table():
    training_tids = generate_training_tids()
    mel_gram_retriever = MelgramRetriever()
    mel_gram_retriever.save_training_tids(training_tids)
    audio_feature_retriever = AudioFeatureRetriever()
    audio_feature_retriever.save_training_tids(training_tids)
    raw_retriever = RawRetriever()
    raw_retriever.save_training_tids(training_tids)
    print "FINISH SAVING TRAINING TIDS"


def save_testing_audio_into_table():
    training_tids = generate_training_tids()
    print "training tid count:", len(training_tids)
    testing_tids = generate_broad_testing_tids()
    print "testing tid count:", len(testing_tids)
    while True:
        # 1. find saved tids not in training from both audio feature table and mel feature table
        # # save the unbalanced tids :
        # some belong to only mel_gram : save to the others
        # set(mel_gram_test_set)-(set(audio_feature_test_set)
        mel_gram_retriever = MelgramRetriever()
        audio_feature_retriever = AudioFeatureRetriever()
        raw_retriever = RawRetriever()
        #retrievers = [mel_gram_retriever,audio_feature_retriever,raw_retriever]
        mel_gram_test_set = mel_gram_retriever.check_saved_test_tid(
            training_tids)
        audio_feature_test_set = audio_feature_retriever.check_saved_test_tid(
            training_tids)
        raw_test_set = raw_retriever.check_saved_test_tid(training_tids)
        saved_count = len(set(mel_gram_test_set) & set(
            audio_feature_test_set) & set(raw_test_set))
        if saved_count >= len(training_tids):
            break
        # 2. if the saved test tids is not enough, continue to crawling else finish
        possible_test_tids = list(set(
            testing_tids) - (set(mel_gram_test_set) & set(audio_feature_test_set) & set(raw_test_set)))
        random.shuffle(possible_test_tids)
        audio_feature_retriever.save_testing_tids(
            possible_test_tids[:len(training_tids) - saved_count])
        mel_gram_retriever.save_testing_tids(
            possible_test_tids[:len(training_tids) - saved_count])
        raw_retriever.save_testing_tids(
            possible_test_tids[:len(training_tids) - saved_count])


import numpy as np
from collections import Counter
class Batch_Generation_Class():
    def __init__(self, content_retriever, context=None, fix_tid_list=None, random_select=True, labeled_tid_list=None):
        self.context = context
        self.random_select = random_select
        # construct tag matcher:
        if context != None:  # match to all labeled songs , some can be unlabeled
            self.neighbor_tag_matcher = Tag_Matcher(
                context, playlist_source='aotm')
        self.unlabeled_tid_list = None
        if fix_tid_list != None:
            if labeled_tid_list != None:
                self.unlabeled_tid_list = list(
                    set(fix_tid_list) - set(labeled_tid_list))
        # the tid in the unlabeled tid list will not be match to tags
        self.groundtruth_tag_matcher = Tag_Matcher(
            unlabeled_tid_list=self.unlabeled_tid_list)
        print "Tag Matcher LOAD and connected to", context
        # construct similar song matcher
        self.tag_list = self.groundtruth_tag_matcher.tag_list
        if context != None:
            self.matcher = Similar_Track_Matcher(
                context, tid_matcher=self.neighbor_tag_matcher.match_one_by_one, playlist_source='aotm', fix_tid_list=fix_tid_list, labeled_tid_list=labeled_tid_list, random_select=random_select)
            print "Similar_Track_Matcher LOAD and connected to", context
        # construct content matcher for call back !
        self.content_retriever = content_retriever
        self.content_retriever.open()
        self.call_back = self.content_retriever.get
        print "content_retriever Open"
    # data processing

    def binarize(self, tags):  # consider 0 or 1 only
        Y = [0. for _ in range(len(self.tag_list))]
        for t in set(tags):
            Y[self.tag_list.index(t)] = 1.
        return np.array(Y, dtype=np.float64)

    def downsampling_tags(self, tag_set_list):
        if len(tag_set_list) != 0:
            tag_count = Counter(it.chain(*imap(list, tag_set_list)))
            if len(tag_count) != 0:
                Nmax = max(tag_count.values())
                k = 1. / Nmax
                dropout_probs = map(
                    lambda x: k * (Nmax - x), tag_count.values())
                dropout_samples = map(lambda x: np.random.choice(
                    [1, 0], p=[1 - x, x]), dropout_probs)
                return np.array(tag_count.keys())[np.where(dropout_samples)[0]].tolist()
            else:
                return {}
        else:
            return tags

    def tag_aggregate(self, input):
        return self.binarize(self.downsampling_tags(input))
    # tid to batch convertor

    def content_batch_convertor(self, tid_batch):
        return self.call_back(tid_batch)

    def neighbor_tag_batch_convertor(self, tid_batch, decay):
        training_tags = self.matcher.get_similar_track(tid_batch)
        # TODO: do not match to not tag matchable similar songs !
        return np.vstack(map(self.binarize, training_tags))

    def groundtruth_tag_batch_convertor(self, tid_batch):
        return np.vstack(map(self.binarize, self.groundtruth_tag_matcher.match_one_by_one(tid_batch)))
    # batch generator

    def generate_tid_batches(self, tid_list, batch_size):
        # TODO: circular yielding tid batches
        def batch_yieldor(tid_list):
            for i in range(int(len(tid_list) / batch_size)):
                yield tid_list[i * batch_size:(i + 1) * batch_size]

        def train_batch_generator(tid_list):
            while True:  # endless epoch
                #random.shuffle(tid_list)  # suffle data in the epoch no suffle 
                yield batch_yieldor(tid_list)  # yield batch by batch
        return it.chain.from_iterable(train_batch_generator(tid_list))

    def get_data(self, test_tids):  # FOR TESTING
        return self.content_batch_convertor(test_tids), self.groundtruth_tag_batch_convertor(test_tids)

    def get_generator(self, tid_list, batch_size, decay=None):  # FOR TRAINING
        if self.context == 'playlist':
            assert decay != None
        if self.context != None:
            for tid_batch in self.generate_tid_batches(tid_list, batch_size):
                yield [self.content_batch_convertor(tid_batch)], [self.groundtruth_tag_batch_convertor(tid_batch), self.neighbor_tag_batch_convertor(tid_batch, decay)]
        else:  # if context = None => no PROPAGATION
            for tid_batch in self.generate_tid_batches(tid_list, batch_size):
                yield self.content_batch_convertor(tid_batch), self.groundtruth_tag_batch_convertor(tid_batch)


'''

'''

'''
Test script : 
from Batch_Generation_Class import * 


f = open(u"training_tids", u"r")
content_matchable_tids = json.loads(f.read()) # all content matchable

content_retriever = RawRetriever()
content_matchable_tids = list(content_retriever.check_saved_tid())
bg = Batch_Generation_Class('artist',content_retriever)  
bg.content_batch_convertor(content_matchable_tids[:1000]) 

tag_matchable_tids = context_match_tids(['tag','artist','webpage','lyrics','listen','playlist']) 
bg.groundtruth_tag_batch_convertor(target_tids[:100]) # float object is not iterable 


# after adding constrain to similarity track matcher 
bg.neighbor_tag_batch_convertor(training_tids[:100]) 

'''
