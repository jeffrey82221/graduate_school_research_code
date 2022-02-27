from __future__ import absolute_import
import pandas as pd
import itertools as it
import sqlite3
from io import open
from itertools import imap, ifilter, izip
from igraph import *
import h5py
import random
from functools import partial
# build an id to audio matcher (melgram)
from madmom.audio.signal import Signal
from os import listdir
from random import randint
import numpy as np
from sklearn.preprocessing import normalize
import multiprocessing
from Crawling_Tools2 import *
from sklearn.feature_extraction.text import TfidfVectorizer
#from scipy.spatial.distance import cosine
#from sklearn.metrics import pairwise_distances
from melgram_audio_processor import compute_melgram
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, rbf_kernel
from sklearn.decomposition import TruncatedSVD
import timeout_decorator
import time
import csv
import re
import simplejson as json
import ijson

#import eventlet
import scipy.sparse

import os

# result many segment , each with size : n_samples


def istid(tid):
    try:
        return tid[0] == 'T'
    except:
        return False


def filtering(p, fix_song_id_list):
    return filter(lambda x: x in fix_song_id_list, p)


def audio_to_segments(filename, sample_rate, n_samples):
    """Loads, and splits an audio into N segments.
    Args:
        filename: A path to the audio.
        sample_rate: Sampling rate of the audios. If the sampling rate is different 
          with an audio's original sampling rate, then it re-samples the audio.
        n_samples: Number of samples one segment contains.

      Returns:
        A list of numpy arrays; segments.
    """
    # Load an audio file as a numpy array
    sig = Signal(filename, sample_rate=sample_rate, dtype=np.float32)
    total_samples = sig.shape[0]
    try:
        if sig.shape[1] == 2:
            sig = sig[:, 0] / 2 + sig[:, 1] / 2
    except:
        None
    n_segment = total_samples // n_samples
    segments = [sig[i * n_samples:(i + 1) * n_samples]
                for i in range(n_segment)]
    return segments


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


class AudioRetriever():
    def open(self):
        self.hf = h5py.File(self.database, 'r')

    def close(self):
        self.hf.close()

    def check_saved_tid(self):
        if os.path.isfile(self.database):
            hf = h5py.File(self.database, 'r')
            saved_audio_tid_set = set(hf.keys())
            print "saved audio feature count:", len(saved_audio_tid_set)
            hf.close()
            return saved_audio_tid_set
        else:
            return set()

    def check_saved_test_tid(self, training_tids):
        hf = h5py.File(self.database, 'r')
        saved_audio_tid_set = set(hf.keys())
        print "saved audio feature count:", len(saved_audio_tid_set)
        hf.close()
        return saved_audio_tid_set - set(training_tids)

    def cleaned_saved_tids(self):
        try:
            f = open(self.name + ".clean.tids", "r")
            clean_tids = json.loads(f.read())
            f.close()
            return set(clean_tids)
        except:
            saved_tids = list(self.check_saved_tid())
            # print "saved_tids",len(saved_tids)
            self.open()

            def check_not_none(tids):
                #global clean_tid_list
                for tid_batch in batch(tids, n=int(len(tids) / 100 + 1)):
                    if type(self.get(tid_batch)) != type(None):
                        print len(tid_batch), "+",
                        yield tid_batch
                    else:
                        if len(tid_batch) == 1:
                            print tid_batch[0], "error",
                        else:
                            yield list(it.chain.from_iterable(check_not_none(tid_batch)))
            # print "test getting",self.get([saved_tids[0]])
            #cleaned_tids_set = set(filter(lambda x:type(self.get([x]))!=type(None),saved_tids))
            clean_tids = list(it.chain.from_iterable(
                check_not_none(saved_tids)))
            print "clean audio tid size:", len(clean_tids)
            f = open(self.name + ".clean.tids", "w")
            f.write(json.dumps(clean_tids, ensure_ascii=False))
            f.close()
            # print "cleaned_tids_set",len(cleaned_tids_set)
            self.close()
            return set(clean_tids)

    def save_target_tids(self, target_tids=None):
        # 0. remove not raw matchable tids
        filenames = listdir("/data/jeffrey82221/MSD_Audio_Raw")
        self.tid_map_dict = dict(map(lambda x: (
            x.split(".")[0], "/data/jeffrey82221/MSD_Audio_Raw/" + x), filenames))
        file_tid_list = list(self.tid_map_dict.keys())
        target_tids = list(set(file_tid_list) & set(target_tids))
        # 1. remove saved tids
        saved_tids = self.check_saved_tid()
        target_tids = list(set(target_tids) - set(saved_tids))

        # 2. remove missing tids
        try:
            f = open("missing_tids", "r")
            missing_tids = map(lambda x: x.split("\n")[0], list(f))
            f.close()
            target_tids = list(set(target_tids) - set(missing_tids))
        except:
            None
        # 3. matching
        if os.path.isfile(self.database):
            # hf = h5py.File('/data/jeffrey82221/audio_tmp/raw_data.h5', 'a')
            hf = h5py.File(self.database, 'a')
        else:
            hf = h5py.File(self.database, 'w')
        miss_count = 0
        saved_count = 0
        for tid in target_tids:
            try:
                print "===================================================="
                print "getting audio..."
                audio = self.audio_processor(self.tid_map_dict[tid])
                print "get audio"
                # if remove compression ?
                hf.create_dataset(tid, data=audio, dtype=np.float32,
                                  chunks=True, compression="gzip")
                print "SAVE audio", saved_count
                # time.sleep(1)
                saved_count += 1
                if saved_count % 10 == 0:
                    print saved_count, "|",
            except Exception, e:
                print e
                miss_count += 1
                if isinstance(e, timeout_decorator.TimeoutError):
                    print "Timed Out"
                else:
                    print miss_count, "missing"
                    f = open("missing_tids", "a")
                    f.write(tid + "\n")
                    f.close()
                # if miss_count>50:
                #    break
            # if saved_count>5:
            #    break
        hf.close()


class RawRetriever(AudioRetriever):
    # TODO: check list in saved tid :
    # TODO: save training tids into table
    # TODO: save testining tids into table (input: testing possible tids, length of selection)
    # TODO: retrieving testing tids in the table (input training tids)
    def __init__(self):
        #self.database = '/data/jeffrey82221/raw_data.h5'
        self.database = '/data/jeffrey82221/audio_tmp/raw_data.h5'
        self.name = "raw"
        self.audio_processor = partial(
            audio_to_segments, sample_rate=22050, n_samples=59049)

    def random_get_raw_segment(self, tid):
        n_segment = self.hf[tid].shape[0]
        # print n_segment
        rand_ = randint(0, n_segment - 1)
        # print rand_
        return self.hf[tid][rand_, :]

    def get(self, id):
        try:
            return np.vstack(map(lambda tid: np.array(self.random_get_raw_segment(tid)), id))
        except:
            return None


# build an id to audio matcher (melgram)

class MelgramRetriever(AudioRetriever):
    # TODO: check list in saved tid :
    # TODO: save training tids into table
    # TODO: save testining tids into table (input: testing possible tids, length of selection)
    # TODO: retrieving testing tids in the table (input training tids)
    def __init__(self):
        self.database = '/data/jeffrey82221/audio_tmp/mel_data.h5'
        self.name = "mel"
        #@timeout_decorator.timeout(2)

        def librosa_melgram_computation(input):
            print "start computing melgram..."
            audio = compute_melgram(input)
            print "finish compute melgram"
            return audio
        self.audio_processor = librosa_melgram_computation

    def get(self, id):
        # np.rollaxis(mel_gram_retriever.get([u'TRMLMNO128EF3522E6'])[0],1,4)
        try:
            # return map(lambda tid:np.rollaxis(np.array(self.hf[tid]),1,4),id)#compute_melgram(self.tid_map_dict[id]).tolist()
            # compute_melgram(self.tid_map_dict[id]).tolist()
            return np.rollaxis(np.vstack(map(lambda tid: np.array(self.hf[tid]), id)), 1, 4)
        except:
            return None


class FeatureRetriever(AudioRetriever):
    def __init__(self):
        self.database = '/data/jeffrey82221/audio_tmp/feature_data.h5'
        self.name = "feature"
        self.audio_processor = lambda audio: filter(
            lambda x: type(x) != unicode, audio)

    def get(self, id):
        try:
            return np.vstack(map(lambda tid: np.array(self.hf[tid]), id))
        except:
            return None

    def get_audio_feature_matchable_tids(self):
        return list(self.hf.keys())

    def save_target_tids(self, target_tids=None):
        # 1. remove saved tids
        saved_tids = self.check_saved_tid()
        target_tids = list(set(target_tids) - set(saved_tids))
        saved_count = len(set(target_tids) & set(saved_tids))
        miss_count = 0
        # 2. matching
        if os.path.isfile(self.database):
            hf = h5py.File(self.database, 'a')
        else:
            hf = h5py.File(self.database, 'w')
        for tid, audio in audio_robust_generator(target_tids, 10, 10):
            try:
                hf.create_dataset(tid, data=self.audio_processor(
                    audio), dtype=np.float32, chunks=True, compression="gzip")
                saved_count += 1
                if saved_count % 10 == 0:
                    print saved_count, "|",
            except:
                miss_count += 1
                if miss_count % 10 == 0:
                    print miss_count, "MISSING", '|',
        hf.close()


class Tag_Matcher(object):
    def __init__(self, context=None, playlist_source=None, unlabeled_tid_list=None):
        # initialization setting
        self.context = context
        if context == "playlist":
            assert playlist_source == "aotm" or playlist_source == "yes"
        self.playlist_source = playlist_source
        if self.playlist_source == 'yes':
            self.yes_tag_table = pd.DataFrame.from_csv(
                u"/data/jeffrey82221/YES/tag_hash.txt", header=None, index_col=None, sep=u", ")
            self.yes_tag_table.columns = [u'id', u'tag']
            self.yes_tag_table = self.yes_tag_table.set_index('id')
            print u"yes tag info LOAD"
        self.unlabeled_tid_list = unlabeled_tid_list
        # tag organization
        f = open(u'tag_merge_rule.txt')
        lines = imap(lambda x: list(imap(lambda y: y.strip(), x)),
                     imap(lambda x: x.split(u"\n")[0].split(u','), f))
        merging_tags = list(it.chain(lines))
        pairs = []
        for merge_rule in imap(lambda x: (x[0:-1], x[-1]), merging_tags):
            for key in merge_rule[0]:
                pairs.append((key, merge_rule[1]))
        self.replace_dict = dict(pairs)
        print u"merge rules LOAD"
        f = open(u'irrelevant_tag.txt')
        self.irrelevant_tags = list(
            imap(lambda x: x.strip(), f.readline().split(u",")))
        print u"irrelevant tag LOAD"
        genre = [u'progressive rock', u'progressive', u'experimental', u'psychedelic', u'acoustic', u'psychedelic rock', u'blues', u'blues rock', u'lounge', u'reggae', u'close harmony', u'classic rock', u'folk', u'folk rock', u'garage rock', u'alt rock', u'grunge', u'punk', u'punk rock', u'alternative  punk', u'hard rock', u'metal', u'heavy metal', u'alternative metal', u'emo', u'hardcore', u'rock', u'alternative', u'alternative rock', u'ballad', u'indie',
                 u'indie rock', u'dance', u'indie pop', u'lo-fi', u'glam rock', u'classic', u'hip-hop', u'rap', u'r&b', u'soul', u'classic soul', u'old school soul', u'motown', u'country', u'classic country', u'southern rock', u'alt-country', u'post-punk', u'new wave', u'synthpop', u'pop rock', u'soft rock', u'adult contemporary', u'urban', u'funk', u'groovy', u'electronic', u'disco', u'jazz', u'smooth jazz', u'pop', u'fusion', u'swing', u'vocal jazz', u'jazz vocal']
        theme = [u'nostalgic', u'political', u'christian rock', u'christian', u'worship', u'retro',
                 u'romantic', u'love song', u'old school', u'epic', u'summer', u'gospel', u'christmas', u'driving']
        mood = [u'oldies', u'dreamy', u'atmospheric', u'melodic', u'mellow', u'memories', u'dark', u'powerful', u'heavy', u'uplifting', u'melancholic', u'smooth', u'easy', u'sad', u'emotional', u'chill', u'relax', u'chillout', u'sensual',
                u'energetic', u'happy', u'upbeat', u'lovely', u'sweet', u'cute', u'guilty pleasure', u'downtempo', u'calm', u'party', u'quiet storm', u'ambient', u'slow', u'sexy', u'sex', u'hot', u'soft', u'easy listening', u'slow jams']
        instrument = [u'male vocalist', u'female vocalist', u'vocal',
                      u'guitar', u'bass', u'piano', u'saxophone', u'instrumental']
        '''
        other = [u'driving', u'loved', u'legend', u'aitch', u'catchy', u'fun', u'beautiful',
                 u'cool', u'love', u'major key tonality', u'a subtle use of vocal harmony', u'sing along']
        '''
        location = [u'canadian', u'california', u'new york', u'american',
                    u'americana', u'british', u'english', u'latin', u'spanish']
        era = [u'50s', u'60s', u'70s', u'80s', u'90s',
               u'2007', u'00s', u'2008', u'2009', u'10s']
        self.tag_categories_dict = {
            u"genre": genre,
            u"theme": theme,
            u"mood": mood,
            u"instrument": instrument,
            u"location": location,
            u"era": era
        }
        print u"categorical tag LOAD"
        print u"tag processing info LOAD"
        # get tag matchable tids
        self.tag_list = list(it.chain(*self.tag_categories_dict.values()))
        f = open("/data/jeffrey82221/MSD_Tags/tracks_with_tag.txt", "r")
        self.tid_list = list(imap(lambda x: x.split("\n")[0], f))
        print "tag matchable tid LOAD"
        if playlist_source != 'yes':
            try:
                self.tid_tag_table = pd.read_pickle("tid_tag_table.p")
                print u"tag matching info LOAD"
            except:
                print "start loading tid tag table ... "
                self.cnx_tag = sqlite3.connect(
                    '/data/jeffrey82221/MSD_Tags/lastfm_tags.db')  # ensure in the same folder
                print "tag matchable tid count:", len(self.tid_list)
                tid_iter = iter(self.tid_list)
                tid_iter_key, tid_iter_match = it.tee(tid_iter)
                post_counter = Count_Class(tid_iter_match, resolution=int(
                    len(self.tid_list) / 10), name=u"tid tag matching :")
                tid_iter_match = post_counter.passing()
                matched_tags = imap(
                    lambda x: self.match_tid_to_tag_set(x), tid_iter_match)
                self.tid_tag_table = pd.DataFrame.from_records(
                    izip(tid_iter_key, matched_tags))
                self.tid_tag_table.columns = ['tids', 'tags']
                self.tid_tag_table = self.tid_tag_table.set_index('tids')
                print u"tag matching info LOAD"
                self.tid_tag_table.to_pickle("tid_tag_table.p")
                print u"tag matching info SAVE"
        else:
            f = open("/data/jeffrey82221/YES/tags.txt")

            def yes_tag_txt_process(line):
                if line[0] == "#":
                    return []
                else:
                    return map(int, line.split("\n")[0].split(' '))
            lines = imap(yes_tag_txt_process, f)
            yes_song_tag_table = pd.DataFrame(
                list(enumerate(lines)), columns=['yes_id', 'tag_ids'])
            self.yes_song_tag_table = yes_song_tag_table.set_index('yes_id')
            print "yes tag matching info LOAD"
            self.yes_song_tag_table['tags'] = self.yes_song_tag_table.tag_ids.map(
                lambda x: set(self.processing(map(lambda id: self.yes_tag_table.loc[id].tag, x))))
            print "yes song tag table PROCESSED"

    def match_one_by_one(self, ids):
        # if by_tid: # ground truth tags matching do not depend on context selection
        #    return list(self.tid_tag_table.loc[ids].tags)
        # for tid in the self.unlabeled_tid_list match to empty set
        if self.playlist_source == "yes":
            return list(self.yes_song_tag_table.loc[ids].tags)
        else:
            if self.unlabeled_tid_list != None:
                if len(set(ids) - set(self.unlabeled_tid_list)) == 0:
                    return [set()] * len(ids)
                else:
                    return map(lambda x: x if type(x) == set else set(), list(self.tid_tag_table.loc[map(lambda x:"<EMPTY>" if x in self.unlabeled_tid_list else x, ids)].tags))
            else:
                # for the neighbor tids , the tag matching step goes in here
                return list(self.tid_tag_table.loc[map(lambda x:x if istid(x) else np.random.choice(self.tid_list), ids)].tags)

    def match_tid_to_tag_set(self, id):
        sql = "SELECT tags.tag, tid_tag.val FROM tid_tag, tids, tags WHERE tags.ROWID=tid_tag.tag AND tid_tag.tid=tids.ROWID and tids.tid='%s'" % id
        tag_table = pd.read_sql_query(sql, self.cnx_tag)
        # processing by pipe
        return set(self.processing(tag_table.tag.tolist()))

    def match(self, id):
        if self.context == "playlist":
            return set(self.processing(map(lambda x: self.yes_tag_table.loc[x].tag, self.yes_song_tag_table.loc[id])[0].tolist()))
        else:
            sql = "SELECT tags.tag, tid_tag.val FROM tid_tag, tids, tags WHERE tags.ROWID=tid_tag.tag AND tid_tag.tid=tids.ROWID and tids.tid='%s'" % id
            tag_table = pd.read_sql_query(sql, self.cnx_tag)
            # processing by pipe
            return set(self.processing(tag_table.tag.tolist()))

    def match_all(self, ids):  # merged all matched tags
        if self.playlist_source == "yes":
            return list(self.yes_song_tag_table.loc[ids].tags)
        else:
            try:
                match_lines = list(self.tid_tag_table.loc[list(ids)].tags)
                # if a tid is not match : only NONE return
                return filter(lambda x: type(x) == set, match_lines)
            except:
                return [set()]
    # PROCESSING FUNCTION

    def match_to_yes(self, tag):
        if tag in list(self.yes_tag_table.tag):
            return True
        else:
            return False

    def replace_tag(self, input):
        if input in self.replace_dict:
            return self.replace_dict[input]
        else:
            return input

    def irrelevant(self, input):
        if input in self.irrelevant_tags:
            return False
        else:
            return True

    def processing(self, input_iter):
        return ifilter(self.irrelevant, imap(self.replace_tag, ifilter(self.match_to_yes, imap(lambda x: x.lower(), input_iter))))


class Id_to_Title_Artist_Matcher():  # playlist need a new kind of matching scheme
    def __init__(self, source):
        self.context = source
        if source != 'playlist':
            cnx_meta = sqlite3.connect(u'/data/jeffrey82221/track_metadata.db')
            whole_song_info_table = pd.read_sql_query(
                u"SELECT track_id,title,artist_name FROM songs", cnx_meta)
            self.whole_song_info_table = whole_song_info_table.set_index(
                u"track_id")
            cnx_meta.close()
            print u"track_metadata.db", u"LOAD"
        else:
            num_lines = sum(1 for line in open(
                u'/data/jeffrey82221/YES/song_hash.txt'))
            f = open(u'/data/jeffrey82221/YES/song_hash.txt', u'rt')
            splitted = csv.reader(f, delimiter='\t', quotechar='|')

            def remove_parenthese(string):
                return re.sub(ur'\([^)]*\)', u'', string).strip()
            name_matcher = map(lambda x: (int(x[0].strip()), (remove_parenthese(
                x[1]), remove_parenthese(x[2]))), splitted)
            yes_track_table = pd.DataFrame(
                name_matcher, columns=['yes_id', 'track_name'])
            self.yes_track_table = yes_track_table.set_index('yes_id')

    def match_tid_to_title(self, tid):
        return self.whole_song_info_table.loc[tid].title, self.whole_song_info_table.loc[tid].artist_name

    def match_all(self, tids):
        if self.context == "playlist":
            return list(self.yes_track_table.loc[tids].track_name)
        else:
            return map(self.match_tid_to_title, tids)


#


# add a tag_link_tid_filter into __init__ ()
# need a call_back function
# - to decide if a tid can be match with a tag () => ( match to all context & match to all tag ) (no need to match with content or not )
# - to decide a tid is untagged (adding signature to tid keys )
# : the alteration function is the ex. "np.random.choice" function , need to choose from a list
# : method : we can add a subsetting function (filtering function ) to reduce element from the "list"
# : where is the selection function ? :
# 1. playlist           => in the "neighbor_propagate"."propagate_select(select_line, decay)" function
# 2. artist             => in the "get_similar_by_artist"  function :  alter "np.random.choice" function
# 3. lyrics, listen     => in the "get_similar_by_sparse_matrix" function : alter "np.random.choice" function
# 4. webpage            => in the "get_similar_by_webpage"  : alter "np.random.choice" function
# - how to filter : select the ones in fix_tid_list (all labeled and unlabeled tid list) specifically in the labeled tid list !
#


class Similar_Track_Matcher(object):
    def __init__(self, context, playlist_source=None, fix_tid_list=None, labeled_tid_list=None, tid_matcher=lambda x: x, random_select=True, reduced_dimension=None):
        self.tid_matcher = tid_matcher
        self.context = context
        if context == 'playlist':
            assert playlist_source == 'aotm' or playlist_source == 'yes'
            self.playlist_source = playlist_source
        self.random_select = random_select
        self.labeled_tid_list = labeled_tid_list
        self.reduced_dimension = reduced_dimension
        if labeled_tid_list != None:
            # all items in labeled_tid_list are in the fix_tid_list !
            assert len(set(labeled_tid_list) - set(fix_tid_list)) == 0
        if self.context == u"artist":
            self.load_artist_info(fix_tid_list)
            # 1. self.artist_tid_table
            # 2. self.tid_artist_table
            # 3. self.tid_list
        if self.context == u"webpage":
            self.load_webpage_info(fix_tid_list, reduced_dimension)
        if self.context == u"lyrics":
            self.load_lyrics_info(fix_tid_list, reduced_dimension)
            # 1. self.tid_song_id_table
            # 2. self.tid_list
        if self.context == u"listen":
            self.load_listen_info(fix_tid_list, reduced_dimension)
            # 1. self.tid_song_id_table
            # 2. self.tid_list
        if self.context == u"playlist":
            self.load_playlist_info(fix_tid_list)
        if self.context == "content":
            self.load_audio_info(fix_tid_list)

    def index(self, array, indices):
        if len(set(indices)) == 1 and math.isnan(indices[0]):
            return indices
        else:
            return pd.DataFrame(array).loc[indices][0].tolist()

    def get_next_node(self, this_node):  # for igraph
        if type(this_node) != int and type(this_node) != np.int64:
            this_node = self.G.vs.find(name=this_node).index
        Nodes = self.G.neighbors(this_node)
        Weights = map(lambda n: self.G.es[self.G.get_eid(
            this_node, n)]['weight'], Nodes)
        Prop = np.array(Weights).astype(np.float) / np.sum(Weights)
        return np.random.choice(Nodes, p=Prop)

    def generate_random_walk(self, seed_node, length):  # same
        yield seed_node
        start_node = seed_node
        for _ in range(length - 1):
            next_node = self.get_next_node(start_node)
            # if type(this_node)!=int and type(this_node)!=np.int64:
            #    yield self.G.vs[next_node]['name']
            # else:
            yield next_node
            start_node = next_node

    def load_artist_info(self, fix_tid_list):
        try:
            f = open(u"tid_artist_pairs", u"r")
            tid_artist_pairs = json.loads(f.read())
            f.close()
        except:
            print u"start crawling artist info"
            # matching to ids
            # build an id to audio feature matcher (first build an tid to feature table in to a hickle file)
            arr = os.listdir(u'/data/jeffrey82221/MSD_Audio_Raw/')
            tids = imap(lambda x: x.split(u".")[0], arr)
            tids_key, tids = it.tee(tids)
            title_artists = title_artist_generator_from_tid(tids)
            artists = imap(lambda x: x[1], title_artists)
            tid_artist_pairs = list(izip(tids_key, artists))
            f = open(u"tid_artist_pairs", u"w")
            f.write(json.dumps(tid_artist_pairs, ensure_ascii=False))
            f.close()
        tid_artist_dict = dict(tid_artist_pairs)
        self.tid_artist_table = pd.DataFrame(
            tid_artist_dict.items(), columns=['tid', 'artist'])
        self.artist_tid_table = self.tid_artist_table.groupby(
            u'artist').aggregate(lambda x: list(x))
        self.tid_artist_table = self.tid_artist_table.set_index('tid')
        self.tid_list = list(tid_artist_dict.keys())
        self.artist_list = list(self.artist_tid_table.index)
        if fix_tid_list != None:
            self.tid_artist_table = self.tid_artist_table.loc[fix_tid_list]
            self.artist_tid_table = self.tid_artist_table.reset_index(
                'tid').groupby(u'artist').aggregate(lambda x: list(x))
            self.artist_list = list(self.artist_tid_table.index)
            print "FIX TID DONE"
        print u"artist info load"

    def load_audio_info(self, fix_tid_list):
        assert fix_tid_list != None
        self.tid_list = fix_tid_list
        content_retriever = FeatureRetriever()
        content_retriever.open()
        self.matrix = np.vstack(
            content_retriever.get(self.tid_list))
        content_retriever.close()
        self.tid_song_id_table = pd.DataFrame(
            list(enumerate(fix_tid_list)), columns=['id', 'tid'])
        self.tid_song_id_table = self.tid_song_id_table.set_index('tid')
        # tid to index table
        print u"audio info load"

    def load_webpage_info(self, fix_tid_list):
        self.load_artist_info(fix_tid_list)
        artist_webpages_pairs = ijson.items(
            open(u"artist_webpages_pairs", u"r"), "item")

        # artist_webpages_pairs = json.loads(f.read()) # read using generator
        # f.close()
        def proper_noun_page(input):
            title_terms = input[0].split(u" ")
            web_tokens = input[1].split(u" ")
            for i, term in enumerate(web_tokens):
                if term.lower() == title_terms[0].lower():
                    if (web_tokens[i - 1].lower() == u'a' or web_tokens[i - 1].lower() == u'an') and term[0].islower():
                        return False
            return True
        # print u'original web count:', len(artist_webpages_pairs)
        cleaned_artist_webpages = ifilter(lambda x: x[1] != None
                                          and x[1] != u""
                                          and proper_noun_page(x)
                                          and u"may refer to:" not in x[1]
                                          and u'From other capitalisation' not in x[1]
                                          and u'From a page move' not in x[1]
                                          and u'From an ambiguous term:' not in x[1]
                                          and u'This is a redirect' not in x[1]
                                          and u'refer to:' not in x[1], artist_webpages_pairs)
        # print u"cleaned web count:", len(cleaned_artist_webpages)
        artist_webpage_table = pd.DataFrame.from_records(
            cleaned_artist_webpages, columns=['artist', 'webpage'])
        artist_webpage_table = artist_webpage_table.set_index('artist')
        artist_webpage_table = artist_webpage_table.loc[list(
            set(artist_webpage_table.index) & set(self.artist_list))]
        self.webpage_artist_list = list(
            artist_webpage_table.index)  # matchable_with_webpage

        # self.labeled_artist_id_list => the artist should match to labeled songs
        if self.labeled_tid_list != None:
            # many artist that match with the labeled tid list is outside the
            labeled_tid_artist_table = self.tid_artist_table.loc[self.labeled_tid_list]
            artist_labeled_tid_table = labeled_tid_artist_table.reset_index(
                'tid').groupby(u'artist').aggregate(lambda x: list(x))
            labeled_artist_id_list = list(
                artist_labeled_tid_table.index)  # these are the artist list that can be match with at least one label matchable
            self.labeled_artist_id_list = list(
                set(labeled_artist_id_list) & set(self.webpage_artist_list))

        else:
            self.labeled_artist_id_list = None
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(
            1, 3), max_features=100000, stop_words=u'english')
        # this matrix is for calculating similarity
        self.matrix = tfidf_vectorizer.fit_transform(
            imap(lambda x: x[1], artist_webpage_table.reset_index('artist').itertuples(index=False)))
        if reduced_dimension != None:
            self.matrix = self.Dimension_Reduction(
                self.matrix, reduced_dimension)
        # this table help map from an artist to the id number
        self.webpage_artist_table = pd.DataFrame(
            list(enumerate(self.webpage_artist_list)), columns=['id', 'artist'])
        self.webpage_artist_table = self.webpage_artist_table.set_index(
            'artist')
        print u"finish building web tfidf matrix"

    '''
    def load_lyrics_network(self):
        f = open(u"/data/jeffrey82221/MSD_Lyrics/mxm_dataset_train.txt")
        lines = ifilter(lambda x: x[0] != u'#' and x[0] != u'%', f)
        elements = imap(lambda x: x.replace(u"\n", u"").split(u","), lines)
        pairs = imap(lambda x: (x[0], x[2:]), elements)
        processed_pairs = imap(lambda x: (x[0], dict(
            imap(lambda x: list(imap(int, x.split(u":"))), x[1]))), pairs)
        tid_lyric_pairs_train = list(processed_pairs)
        f = open(u"/data/jeffrey82221/MSD_Lyrics/mxm_dataset_test.txt")
        lines = ifilter(lambda x: x[0] != u'#' and x[0] != u'%', f)
        elements = imap(lambda x: x.replace(u"\n", u"").split(u","), lines)
        pairs = imap(lambda x: (x[0], x[2:]), elements)
        processed_pairs = imap(lambda x: (x[0], dict(
            imap(lambda x: list(imap(int, x.split(u":"))), x[1]))), pairs)
        tid_lyric_pairs_test = list(processed_pairs)
        tid_lyric_pairs = tid_lyric_pairs_train + tid_lyric_pairs_test

        f = open("/data/jeffrey82221/MSD_Lyrics/mxm_dataset_test.txt")
        self.word_list = next(it.islice(f, 17, None)).split(',')
        self.word_list[0] = 'i'
        self.tid_list = map(lambda x: x[0], tid_lyric_pairs)
        print "Lyrics Info LOAD"
        # add tid_list and word_list  into graph
        # build sparse matrix base on this tid to word count dict
        # add vertices
        self.G = Graph()
        self.G.add_vertices(len(self.tid_list + self.word_list))
        self.G.vs['name'] = self.tid_list + self.word_list
        # add edge:
        name_id_matcher = dict(
            zip(self.tid_list + self.word_list, range(len(self.tid_list + self.word_list))))

        def song_word_edgelist_generator(tid_lyric_pairs):
            for song_id in range(len(tid_lyric_pairs)):
                for word_id, term_frequency in tid_lyric_pairs[song_id][1].items():
                    yield tid_lyric_pairs[song_id][0], self.word_list[word_id - 1], term_frequency
        edgelist_generator = song_word_edgelist_generator(tid_lyric_pairs)
        graph_edgelist_generator = imap(lambda x: (
            name_id_matcher[x[0]], name_id_matcher[x[1]]), edgelist_generator)  # match tids and word_list to number
        self.G.add_edges(list(graph_edgelist_generator))
        # add weights to edges :
        self.G.es['weight'] = list(
            imap(lambda x: x[2], song_word_edgelist_generator(tid_lyric_pairs)))
        print "Graph Build"
    '''

    def load_lyrics_info(self, fix_tid_list):
        f = open(u"/data/jeffrey82221/MSD_Lyrics/mxm_dataset_train.txt")
        lines = ifilter(lambda x: x[0] != u'#' and x[0] != u'%', f)
        elements = imap(lambda x: x.replace(u"\n", u"").split(u","), lines)
        pairs = imap(lambda x: (x[0], x[2:]), elements)
        tid_lyric_pairs_train = imap(lambda x: (x[0], dict(
            imap(lambda x: list(imap(int, x.split(u":"))), x[1]))), pairs)
        #tid_lyric_pairs_train = list(processed_pairs)
        f = open(u"/data/jeffrey82221/MSD_Lyrics/mxm_dataset_test.txt")
        lines = ifilter(lambda x: x[0] != u'#' and x[0] != u'%', f)
        elements = imap(lambda x: x.replace(u"\n", u"").split(u","), lines)
        pairs = imap(lambda x: (x[0], x[2:]), elements)
        tid_lyric_pairs_test = imap(lambda x: (x[0], dict(
            imap(lambda x: list(imap(int, x.split(u":"))), x[1]))), pairs)
        #tid_lyric_pairs_test = list(processed_pairs)
        tid_lyric_pairs = it.chain(tid_lyric_pairs_train, tid_lyric_pairs_test)
        tid_lyric_table = pd.DataFrame.from_records(
            tid_lyric_pairs, columns=['tid', 'lyrics'])
        tid_lyric_table = tid_lyric_table.set_index('tid')
        print "tid lyric info load"
        if fix_tid_list != None:
            tid_lyric_table = tid_lyric_table.loc[fix_tid_list]
            print "FIX TID DONE"
        print u"lyrics file load"
        tid_lyric_pairs_left, tid_lyric_pairs_right = it.tee(
            tid_lyric_table.reset_index("tid").itertuples(index=False))
        lyrics = imap(lambda x: x[1], tid_lyric_pairs_right)
        lyric_id_list = imap(lambda x: list(
            it.chain(*list(imap(lambda k_v: [k_v[0]] * k_v[1], x.items())))), lyrics)
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(
            1, 1), preprocessor=lambda x: x, tokenizer=lambda x: x)
        self.matrix = tfidf_vectorizer.fit_transform(
            lyric_id_list)  # user word matrix
        if reduced_dimension != None:
            self.matrix = self.Dimension_Reduction(
                self.matrix, reduced_dimension)
        self.tid_list = list(imap(lambda x: x[0], tid_lyric_pairs_left))
        self.tid_song_id_table = pd.DataFrame(
            [(v, i) for i, v in enumerate(self.tid_list)], columns=['tid', 'id'])
        self.tid_song_id_table = self.tid_song_id_table.set_index('tid')
        print u"lyrics tfidf matrix build"
        # 1. tid_song_id_table
        # 2. tid_list
    '''
    def load_listen_network(self):
        self.G = Graph.Read_Ncol(
            "/data/jeffrey82221/MSD_Taste/train_triplets.txt", names=True, directed=False, weights=True)
        # rematch names to tids
        print "Graph LOAD"
        f = open("/data/jeffrey82221/MSD_Taste/unique_tracks.txt")
        elements = imap(lambda x: x.replace("\n", "").split("<SEP>"), f)
        track_id_table = pd.DataFrame.from_records(
            elements, columns=[u'tid', u'echoid', u'artist', u'title'])
        track_id_table = track_id_table.set_index('echoid')
        node_names = self.G.vs['name']
        tmp_tid_list = filter(lambda x: x[1][0] == 'S', enumerate(node_names))
        self.tid_list = list(
            track_id_table.loc[map(lambda x:x[1], tmp_tid_list)].tid)
        self.G.vs[map(lambda x:x[0], tmp_tid_list)]['name']=self.tid_list
        print "Graph Track id Replaced"
    '''

    def load_listen_info(self, fix_tid_list):
        f = open(u"/data/jeffrey82221/MSD_Taste/unique_tracks.txt")
        elements = imap(lambda x: x.replace(u"\n", u"").split(u"<SEP>"), f)
        track_id_table = pd.DataFrame.from_records(
            elements, columns=[u'tid', u'echoid', u'artist', u'title'])
        print "track info load"
        f = open(u"/data/jeffrey82221/MSD_Taste/train_triplets.txt")
        listening_log_table = pd.read_csv(
            u"/data/jeffrey82221/MSD_Taste/train_triplets.txt", sep=u"\t", header=None)
        listening_log_table.columns = [u'user', u'echoid', u'count']
        listening_log_table = listening_log_table.merge(
            track_id_table, on='echoid')  # match tid to echo id , (an echoid can have multiple tids) => user ~ echoid ~ count => user ~ echoid ~ tids ~ count =>
        # reduce using tids
        print "listening log load"
        if fix_tid_list != None:
            listening_log_table = listening_log_table.set_index(
                'tid').loc[fix_tid_list].reset_index('tid')
            print "FIX TID DONE"
        self.tid_list = list(set(listening_log_table.tid))
        user_list = list(set(listening_log_table.user))
        self.tid_song_id_table = pd.DataFrame(  # match to number id
            [(v, i) for i, v in enumerate(self.tid_list)], columns=['tid', 'id'])
        listening_log_table = listening_log_table.merge(
            self.tid_song_id_table, on='tid')  # merge number id ~ tid to listening log table
        self.tid_song_id_table = self.tid_song_id_table.set_index('tid')
        listening_log_table = listening_log_table.merge(pd.DataFrame(
            list(enumerate(user_list)), columns=['user_id', 'user']), on='user')
        self.matrix = scipy.sparse.csr_matrix((np.array(
            listening_log_table[u'count']), (listening_log_table[u'id'], listening_log_table[u'user_id'])))
        if reduced_dimension != None:
            self.matrix = self.Dimension_Reduction(
                self.matrix, reduced_dimension)
            print 'finished processing dimension reduction'
        print u"user tid matrix build"

    def load_playlist_song_info(self, fix_tid_list):
        if self.playlist_source == 'yes':
            num_lines = sum(1 for line in open(
                u'/data/jeffrey82221/YES/song_hash.txt'))
            f = open(u'/data/jeffrey82221/YES/song_hash.txt', u'rt')
            splitted = csv.reader(f, delimiter='\t', quotechar='|')

            def remove_parenthese(string):
                return re.sub(ur'\([^)]*\)', u'', string).strip()
            name_matcher = imap(lambda x: (
                int(x[0].strip()), (remove_parenthese(x[1]), remove_parenthese(x[2]))), splitted)
            # playlist matching
            playlist_key, name_matcher_ = it.tee(name_matcher)
            name_table = pd.DataFrame(
                list(name_matcher), columns=['song_id', 'name'])
            print "playlist info LOAD"
            f = open("/data/jeffrey82221/MSD_Taste/unique_tracks.txt")
            elements = imap(lambda x: x.replace("\n", "").split("<SEP>"), f)
            track_id_table = pd.DataFrame.from_records(
                elements, columns=[u'tid', u'echoid', u'artist', u'title'])

            track_id_table['name'] = zip(
                track_id_table['title'], track_id_table['artist'])
            print "track_id_table info LOAD"
            # a yes.com id may match to multiple tid , random choose one !

            matched_table = track_id_table.merge(name_table, on='name', how='inner')[
                ['tid', 'song_id']]

            id_nid_dict = dict()
            for i, id in enumerate(matched_table.song_id):
                if id in id_nid_dict:
                    id_nid_dict[id].append(i)
                else:
                    id_nid_dict[id] = [i]
            nids = map(lambda x: random.choice(x[1]), id_nid_dict.items())
            self.tid_song_id_table = matched_table.iloc[nids]
            self.tid_song_id_table = self.tid_song_id_table.set_index('tid')

        elif self.playlist_source == 'aotm':
            # tid_song_id_dict :::
            # load unique song table (track info table )
            # load the important from neighbor_dict keys
            # match to tids
            f = open("/data/jeffrey82221/MSD_Taste/unique_tracks.txt")
            elements = imap(lambda x: x.replace("\n", "").split("<SEP>"), f)
            track_info_table = pd.DataFrame.from_records(
                elements, columns=[u'tid', u'echoid', u'artist', u'title'])
            track_info_table = track_info_table.set_index("echoid")
            try:
                f = open(u"playlist_song_ids", u"r")
                playlist_song_ids = json.loads(f.read())
                f.close()
            except:
                f = open(
                    '/data/jeffrey82221/playlist_aotm/aotm2011_playlists.json')
                playlist_generator = imap(lambda x: list(
                    it.chain(*x)), ijson.items(f, "item.filtered_lists"))
                playlist_generator = Count_Class(
                    playlist_generator, 10000, 'playlist').passing()
                playlist_song_ids = list(
                    set(list(it.chain.from_iterable(playlist_generator))))
                f = open(u"playlist_song_ids", u"w")
                f.write(json.dumps(playlist_song_ids, ensure_ascii=False))
                f.close()
            self.tid_song_id_table = track_info_table.loc[playlist_song_ids][[
                'tid']]
            self.tid_song_id_table = self.tid_song_id_table.reset_index(
                'echoid').set_index('tid')
            self.tid_song_id_table.columns = ['song_id']
        if fix_tid_list != None:
            self.tid_song_id_table = self.tid_song_id_table.loc[fix_tid_list]
        self.song_id_tid_table = self.tid_song_id_table.reset_index(
            'tid').set_index('song_id')
        # reduce the possibility that a song_id may match to multiple tid
        self.tid_list = list(self.tid_song_id_table.index)
        song_id_list = list(self.song_id_tid_table.index)
        song_id_set = list(set(song_id_list))
        nid_list = map(lambda id: song_id_list.index(id), song_id_set)
        self.song_id_tid_table = self.song_id_tid_table.iloc[nid_list]

    def load_playlist_info(self, fix_tid_list):
        self.load_playlist_song_info(fix_tid_list)
        # playlist song count : 119894
        # if save neighbor dict : playlist source dependent and fix_tid_list dependent !
        # imap filter out the irrelevent element in array
        try:
            if fix_tid_list != None:
                f = open(self.playlist_source +
                         str(len(fix_tid_list)) + ".neighbor_dict", u"r")
                neighbor_dict = json.loads(f.read())
                f.close()
            else:
                f = open(self.playlist_source + ".neighbor_dict", u"r")
                neighbor_dict = json.loads(f.read())
                f.close()
            print u"neighbor_dict LOAD"
        except:
            if self.playlist_source == "yes":
                playlist_file = open(u'/data/jeffrey82221/YES/train.txt')
                playlists_train = it.islice(
                    imap(lambda x: list(imap(int, x.split(u' ')[:-1])), playlist_file), 2, None)
                playlist_file = open(u'/data/jeffrey82221/YES/test.txt')
                playlists_test = it.islice(
                    imap(lambda x: list(imap(int, x.split(u' ')[:-1])), playlist_file), 2, None)
                playlists = it.chain(playlists_train, playlists_test)
                # remove not matchable songs from playlist
                # remove
            elif self.playlist_source == "aotm":
                f = open(
                    '/data/jeffrey82221/playlist_aotm/aotm2011_playlists.json')
                #playlist_generator = ijson.items(f, "item")
                playlists = imap(lambda x: list(it.chain(*x)),
                                 ijson.items(f, "item.filtered_lists"))
                #playlists = cleaned_playlist_generator(playlist_generator)
            if fix_tid_list != None:
                fix_song_id_list = list(
                    self.tid_song_id_table.loc[fix_tid_list].song_id)
                p = multiprocessing.Pool()
                playlists = ifilter(lambda x: len(x) > 0, p.imap_unordered(
                    partial(filtering, fix_song_id_list=fix_song_id_list), playlists))

            playlists = Count_Class(playlists, 100, 'playlist').passing()

            # the begining song of itself should remain in the neighbor list
            def neighbor_pair_generator(playlist):
                for i in xrange(len(playlist)):
                    left = playlist[:i][::-1]
                    right = playlist[i + 1:]
                    if len(left) != 0 and len(right) != 0:
                        yield [playlist[i], [left, right]]
                    elif len(left) == 0:
                        yield [playlist[i], [right]]
                    elif len(right) == 0:
                        yield [playlist[i], [left]]
            neighbor_pairs = it.chain.from_iterable(
                imap(neighbor_pair_generator, playlists))

            def clean_neighbor_pair(input):
                return input[0], filter(lambda x: len(x) != 0, input[1])
            neighbor_pairs = ifilter(lambda x: len(x[1]) > 0, imap(
                clean_neighbor_pair, neighbor_pairs))  # the pair with song occur alone in the playlist after filtering is also filter out ! (they should be remain in the list)
            neighbor_dict = dict()
            for neighbor_pair in neighbor_pairs:
                if neighbor_pair[0] in neighbor_dict:
                    neighbor_dict[neighbor_pair[0]].extend(neighbor_pair[1])
                else:
                    neighbor_dict[neighbor_pair[0]] = neighbor_pair[1]

            if fix_tid_list != None:
                f = open(self.playlist_source +
                         str(len(fix_tid_list)) + ".neighbor_dict", u"w")
                f.write(json.dumps(neighbor_dict, ensure_ascii=False))
                f.close()
            else:
                f = open(self.playlist_source + ".neighbor_dict", u"w")
                f.write(json.dumps(neighbor_dict, ensure_ascii=False))
                f.close()
            print u"neighbor_dict BUILD and SAVE"
        self.neighbor_table = pd.DataFrame(neighbor_dict.items(), columns=[
                                           'song_id', 'neighbor_list'])
        self.neighbor_table = self.neighbor_table.set_index('song_id')

    def random_select_an_id_from_list(self, input_id_list, weights=None, labeled_id_list=None):
        # print type(input_id_list),len(input_id_list),type(weights),len(weights),type(labeled_id_list)
        # print input_id_list
        if type(input_id_list) == list or type(input_id_list) == np.ndarray:
            if len(input_id_list) != 0:
                if type(labeled_id_list) != type(None):
                    if type(weights) != type(None):
                        id_list_weight_pairs = filter(
                            lambda x: x[0] in labeled_id_list, zip(input_id_list, weights))
                        filtered_id_list = map(
                            lambda x: x[0], id_list_weight_pairs)
                        filtered_weights = map(
                            lambda x: x[1], id_list_weight_pairs)
                    else:
                        filtered_id_list = list(
                            set(labeled_id_list) & set(input_id_list))
                        filtered_weights = None
                    # filter with weights
                    if len(filtered_id_list) != 0:
                        return np.random.choice(filtered_id_list, p=filtered_weights)
                    else:
                        return np.nan
                else:
                    return np.random.choice(input_id_list, p=weights)
            else:
                return np.nan
        else:
            return np.nan

    def Dimension_Reduction(self, matrix, dimension):
        # save if the matrix is already processed
        svd = TruncatedSVD(n_components=dimension, n_iter=10)
        return svd.fit_transform(matrix)

    def get_similar_by_artist(self, id):
        if self.random_select:
            result_tids = list(self.artist_tid_table.loc[list(
                self.tid_artist_table.loc[id].artist)].tid)
            return map(lambda x: self.random_select_an_id_from_list(x, labeled_id_list=self.labeled_tid_list), result_tids)
        else:
            result_tids = list(self.artist_tid_table.loc[list(
                self.tid_artist_table.loc[id].artist)].tid)
            [result_tids[i].insert(0, result_tids[i].pop(
                result_tids[i].index(ind))) for i, ind in enumerate(id)]
            return result_tids

    def get_similar_artist_by_webpage(self, artists, K=3):
        # some artist is not in the list
        matchable_artists = filter(
            lambda x: x in self.webpage_artist_list, artists)
        if len(matchable_artists) != 0:
            if self.reduced_dimension == None:
                matched_result_artist = self.get_similar_by_sparse_matrix(
                    matchable_artists, self.webpage_artist_table, self.webpage_artist_list, K=K, labeled_id_list=self.labeled_artist_id_list)  # only 841 songs
            else:
                matched_result_artist = self.get_similar_by_dense_matrix(
                    matchable_artists, self.webpage_artist_table, self.webpage_artist_list)  # only 841 songs
            result_artists = []
            for artist in artists:
                if artist in matchable_artists:  # if the return artist is None
                    result_artists.append(matched_result_artist.pop(0))
                else:
                    if self.random_select:
                        result_artists.append(artist)
                    else:
                        result_artists.append([artist])
            return result_artists
        else:
            if self.random_select:
                return artists  # all artist is not matchable
            else:
                return map(lambda x: [x], artists)

    def get_similar_by_webpage(self, id, K=3):
        artists = list(self.tid_artist_table.loc[id].artist)
        similar_artists = self.get_similar_artist_by_webpage(artists, K)
        if self.random_select:
            if len(set(similar_artists)) == 0 and similar_artists[0] == None:
                result_tids = [[]] * len(similar_artists)
            else:
                result_tids = self.artist_tid_table.loc[similar_artists].tid
            return map(lambda x: self.random_select_an_id_from_list(x, labeled_id_list=self.labeled_tid_list), result_tids)
        else:
            result_tids = [list(it.chain(*self.artist_tid_table.loc[a].tid))
                           for a in similar_artists]
            [result_tids[i].insert(0, result_tids[i].pop(
                result_tids[i].index(ind))) for i, ind in enumerate(id)]
            return result_tids

    def get_similar_by_dense_matrix(self, input, id_convert_table, id_convert_list):
        song_length = len(id_convert_list)
        similarities = rbf_kernel(
            self.matrix[id_convert_table.loc[input].id], self.matrix)
        similarity_normalized = normalize(
            similarities, norm='l1', axis=1)  # 16.1ms
        neighbor_ids = [self.random_select_an_id_from_list(
            range(song_length), similarity_normalized[i]) for i in range(len(input))]
        return self.index(id_convert_list, neighbor_ids)

    def get_similar_by_sparse_matrix(self, input, id_convert_table, id_convert_list, K=5, labeled_id_list=None):
        start_time = time.clock()
        if labeled_id_list != None:
            assert len(set(labeled_id_list) - set(id_convert_table.index)) == 0
            # if all labeled_id_list of artist in the id convert table ?
            labeled_nid_list = np.array(
                id_convert_table.loc[labeled_id_list].id)
            # remove the no lable rows
            labeled_matrix = self.matrix[labeled_nid_list, :]
            dist_out = cosine_similarity(
                self.matrix[id_convert_table.loc[input].id, :], labeled_matrix, dense_output=False)

        else:

            dist_out = cosine_similarity(
                self.matrix[id_convert_table.loc[input].id, :], self.matrix, dense_output=False)
        print "1.", time.clock() - start_time, 'second'
        start_time = time.clock()
        if self.random_select:
            cos_result_normalized = normalize(
                dist_out, norm='l1', axis=1)  # 16.1ms
            print "2.", time.clock() - start_time, 'second'
            start_time = time.clock()

            def similar_id_select(id):
                return self.random_select_an_id_from_list(cos_result_normalized[id, :].indices, weights=cos_result_normalized[id, :].data)
            neighbor_ids = [similar_id_select(id) for id in range(len(input))]
            print "4.", time.clock() - start_time, 'second'
            if labeled_id_list != None:
                # print neighbor_ids
                # print np.array(labeled_nid_list)[neighbor_ids] # <EMPTY> occur in the neighbor_tids
                # return np.array(id_convert_list)[np.array(labeled_nid_list)[neighbor_ids]].tolist()
                return self.index(id_convert_list, self.index(labeled_nid_list, neighbor_ids))
            else:
                # np.array(id_convert_list)[neighbor_ids].tolist()
                return self.index(id_convert_list, neighbor_ids)
        else:
            result_tids = [self.index(id_convert_list,
                                      np.squeeze(np.argsort(np.array(dist_out[i, :].todense())))[
                                          ::-1][:K]) for i in range(len(input))]
            return result_tids

    def get_similar_by_playlist(self, id, K=5, decay=0.8):
        # match multiple id to multiple yes id
        # match to multiple neighbor clusters
        #
        # do not match one tid to two song_id !
        song_ids = list(self.tid_song_id_table.loc[id].song_id)
        if self.random_select:
            def propagate_select(select_line, decay):
                while True:
                    for line in select_line:
                        if random.uniform(0, 1) < decay:
                            None  # skip to next song
                        else:
                            return line  # hit songs
                            break

            def neighbor_propagate(id_lists, decay):
                # random select a side for entry
                select_line_id = randint(0, len(id_lists) - 1)
                select_line = id_lists[select_line_id]
                return propagate_select(select_line, decay)

            def select_neighbor(song_id, decay):
                try:
                    while True:
                        if random.uniform(0, 1) < decay:
                            return neighbor_propagate(self.neighbor_table.loc[song_id].neighbor_list, decay)
                            None  # skip to next song
                        else:
                            return song_id  # hit songs
                            break
                except:  # except if the "song_id" cannot be match with any neighbors in the "neighbor_table"
                    return song_id
                # if the song_id is not in the dataset
                # map(lambda id:select_neighbor(id,0.8),song_ids)
                # map(lambda id:select_neighbor(id,0.8),['SOKQMAI12A6D4FAC0D','SOJMQNP12A8AE465CC'])
                # neighbor_table.loc['SOJMQNP12A8AE465CC'].neighbor_list
                # neighbor_propagate(neighbor_table.loc['SOJMQNP12A8AE465CC'].neighbor_list,0.8)

            # many match to NaN !! in "self.neighbor_table.loc[song_ids].neighbor_list"
            if self.playlist_source == 'yes':
                return map(lambda x: select_neighbor(x, decay=decay), song_ids)
            else:
                # match back to tids
                assert len(map(lambda x: select_neighbor(x, decay=decay), song_ids)) == len(
                    self.song_id_tid_table.loc[map(lambda x: select_neighbor(x, decay=decay), song_ids)].tid)
                return self.song_id_tid_table.loc[map(lambda x: select_neighbor(x, decay=decay), song_ids)].tid
        else:
            def get_neighbors(sections, K=K):
                results = []
                for section in sections:
                    if len(section) >= K:
                        results.extend(section[:K])
                    else:
                        results.extend(section)
                return results
            # adding itself!
            result_lists = list(
                self.neighbor_table.loc[song_ids].neighbor_list.map(get_neighbors))
            if self.playlist_source == 'yes':
                return map(lambda x: [x[0]] + x[1], zip(song_ids, result_lists))
            else:
                # match back to tids
                return map(lambda x: [x[0]] + x[1], zip(song_ids, result_lists))

    def set_tid_matcher(self, func):
        self.tid_matcher = func

    def get_similar_track(self, id, K=2, decay=0.8, rand=False, merge_same_tid=False):
        if self.context == u"artist":
            result_tracks = self.get_similar_by_artist(id)
        elif self.context == u'webpage':
            result_tracks = self.get_similar_by_webpage(id, K + 1)
        elif self.context == u"lyrics":
            if self.reduced_dimension == None:
                result_tracks = self.get_similar_by_sparse_matrix(
                    id, self.tid_song_id_table, self.tid_list, K=K + 1, labeled_id_list=self.labeled_tid_list)
            else:
                result_tracks = self.get_similar_by_dense_matrix(
                    id, self.tid_song_id_table, self.tid_list)
        elif self.context == u"listen":
            if self.reduced_dimension == None:
                result_tracks = self.get_similar_by_sparse_matrix(
                    id, self.tid_song_id_table, self.tid_list, K=K + 1, labeled_id_list=self.labeled_tid_list)
            else:
                result_tracks = self.get_similar_by_dense_matrix(
                    id, self.tid_song_id_table, self.tid_list)
        elif self.context == u"playlist":
            result_tracks = self.get_similar_by_playlist(id, K=K, decay=decay)
        elif self.context == "content":
            result_tracks = self.get_similar_by_dense_matrix(
                id, self.tid_song_id_table, self.tid_list)
        else:
            result_tracks = map(lambda x: [x], id)  # return itself!!
        if rand:
            result_tracks = map(lambda x: list(map(lambda _: random.choice(
                self.tid_list), range(len(x)))), result_tracks)  # randomly choose selection !
        # return map(self.tid_matcher,result_tracks)
        if merge_same_tid:
            return map(lambda x: self.tid_matcher(list(set(x))), result_tracks)
        else:
            # return map(self.tid_matcher,result_tracks)
            return self.tid_matcher(result_tracks)
      # TODO: audio similarity based
      # 1. get audio info for all matchable tids !
      # 2. calculate similarity by audio !


'''
Accessing Script : 
from Source_Matcher import * 
stm = Similar_Track_Matcher('playlist',playlist_source='aotm') 
stm.get_similar_track(stm.tid_list[:100]) 
stm = Similar_Track_Matcher('playlist',playlist_source='yes') 
stm.get_similar_track(stm.tid_list[:100]) 
stm = Similar_Track_Matcher('artist') 
stm.get_similar_track(stm.tid_list[:100]) 
stm = Similar_Track_Matcher('webpage') 
stm.get_similar_track(stm.tid_list[:100]) 
stm = Similar_Track_Matcher('lyrics') 
stm.get_similar_track(stm.tid_list[:100]) 
stm = Similar_Track_Matcher('listen') 
stm.get_similar_track(stm.tid_list[:100]) 

OVERLAP
from Source_Matcher import * 
for i,CONTEXT in enumerate(['artist','webpage','lyrics','listen','playlist']): 
    stm = Similar_Track_Matcher(CONTEXT,playlist_source='aotm') 
    print "Matched Length" , CONTEXT, len(stm.tid_list) 
    if i == 0:
        merge_tids = set(stm.tid_list)
    else:
        merge_tids = list(set(merge_tids)&set(stm.tid_list)) 
    print "Overlap Length:", len(merge_tids) 

CHECK FIX TID REDUCTION WORK
from Batch_Generation_Class import *
context_tids = context_match_tids(['artist','webpage','lyrics','listen','playlist'])
for i,CONTEXT in enumerate(['playlist','artist','webpage','lyrics','listen']): 
    stm = Similar_Track_Matcher(CONTEXT,playlist_source='aotm',fix_tid_list = context_tids) 
    print "Matched Length" , CONTEXT, len(stm.tid_list) 
    if i == 0:
        merge_tids = set(stm.tid_list)
    else:
        merge_tids = list(set(merge_tids)&set(stm.tid_list)) 
    print "Overlap Length:", len(merge_tids) 
    print "===================================================="

'''
