# -*- coding: utf-8 -*-
from public_functions import *
from word2vec import word2vector, load_word2vec300withoutLOOP

__author__ = 'benywon'


class dataPreprocess:
    TRAIN = []
    TEST = []
    DEV = []

    def __init__(self,
                 padding_data=False,
                 Max_length=50,
                 batch_training=False,
                 reload=False,
                 max_batch_size=128,
                 load_embedding=True,  # whether load embedding from file
                 EmbeddingSize=100,
                 **kwargs):
        self.max_batch_size = max_batch_size
        self.batch_training = batch_training
        self.reload = reload
        self.load_embedding = load_embedding
        self.Max_length = Max_length
        self.EmbeddingSize = EmbeddingSize
        self.word2id = {'_NULL_': 0}
        self.vocabularySize = 0
        self.padding_data = padding_data
        self.wordEmbedding = []
        self.data_pickle_path = ''
        self.train_number = 0
        self.dev_number = 0
        self.test_number = 0
        self.dataset_name = ''
        self.path_base = './data/'
        self.transfun = lambda x, y: np.asmatrix(x, dtype=y) if self.batch_training else np.asarray(x, dtype=y)

    def calc_data_stat(self):
        self.train_number = len(self.TRAIN) if self.batch_training else len(self.TRAIN[0])
        if len(self.DEV) > 0:
            self.dev_number = len(self.DEV) if self.batch_training else len(self.DEV[0])
        else:
            self.dev_number = 0
        self.test_number = len(self.TEST)
        self.vocabularySize = len(self.word2id)

    def get_sentence_id_list(self, sentence, add_vacabulary=True, divider=' ', max_length=None):
        if sentence.endswith('?'):
            if not sentence.replace('?', '').endswith(' '):
                sentence = sentence.replace('?', ' ?')
        return [self.get_word_id(word, add_vacabulary) for word in sentence.split(divider)][0:max_length]

    def get_word_id(self, word, add_vacabulary):
        if word.endswith(','):
            word = word.replace(',', '')
        word = str(word)
        if word.endswith('?'):
            word = word.replace('?', '')
        word = clean_word(word)
        if word in self.word2id:
            return self.word2id[word]
        else:
            if add_vacabulary:
                self.word2id[word] = len(self.word2id)
                return self.word2id[word]
            else:
                return 0

    def transferTest(self):
        transfun_default = lambda z: np.asmatrix(z, dtype='int32') if self.batch_training else np.asarray(z,
                                                                                                          dtype='int32')
        self.TEST = [map(transfun_default, x) for x in self.TEST]

    def build_word2vec(self):
        self.vocabularySize = len(self.word2id)
        assert len(self.word2id) > 0, 'you have not load word2id!!'
        self.wordEmbedding = np.zeros(shape=(self.vocabularySize, self.EmbeddingSize), dtype='float32')
        if not (self.EmbeddingSize == 300):
            word2vec = word2vector(self.EmbeddingSize)
            for (word, word_id) in self.word2id.items():
                vec = word2vec.returnWordVec(clean_str_remove(word))
                self.wordEmbedding[word_id] = vec
        else:
            self.wordEmbedding = load_word2vec300withoutLOOP(self.word2id)

    def save_data(self):
        print 'start saveing data to..' + self.data_pickle_path
        obj = {'train': self.TRAIN, 'test': self.TEST, 'dev': self.DEV, 'word2id': self.word2id,
               'word2vec': self.wordEmbedding}

        with open(self.data_pickle_path, 'wb') as f:
            cPickle.dump(obj, f, cPickle.HIGHEST_PROTOCOL)

    def load_data(self):
        print 'start load data from ' + self.data_pickle_path
        with open(self.data_pickle_path, 'rb') as f:
            obj = cPickle.load(f)
        self.TRAIN = obj['train']
        self.TEST = obj['test']
        self.DEV = obj['dev']
        self.word2id = obj['word2id']
        self.wordEmbedding = obj['word2vec']
        print 'data loaded'

    @staticmethod
    def _transfer2batch(list_obj, max_batch_size):
        layer = len(list_obj)
        length = len(list_obj[0])
        length_tuple = [[len(list_obj[i][j]) for i in xrange(layer)] for j in xrange(length)]
        [length_tuple[m].append(m) for m in xrange(length)]
        sorted_tuple = sorted(length_tuple, key=lambda x: [x[m] for m in xrange(layer)])
        sorted_list_tuple = []
        i = 1
        one_patch = [sorted_tuple[0][layer]]
        while i < length:
            if sorted_tuple[i][0:layer] == sorted_tuple[i - 1][0:layer] and len(
                    one_patch) < max_batch_size:
                one_patch.append(sorted_tuple[i][layer])
            else:
                sorted_list_tuple.append(one_patch)
                one_patch = [sorted_tuple[i][layer]]
            i += 1
        sorted_list_tuple.append(one_patch)
        after = []
        for tuple_obj in sorted_list_tuple:
            one_batch = []
            for i in xrange(layer):
                one_batch.append([list_obj[i][item] for item in tuple_obj])
            after.append(one_batch)
        return after

    @staticmethod
    def __transfer2numpy__(obj, batch_training):
        print 'transfer to numpy objects '
        if batch_training:
            layer = len(obj[0])
            length = len(obj)
            for i in xrange(length):
                for j in xrange(layer):
                    obj[i][j] = np.asmatrix(obj[i][j], dtype='int32')
        else:
            layer = len(obj)
            for i in range(layer):
                for j in range(len(obj[i])):
                    obj[i][j] = np.asarray(obj[i][j], dtype='int32')

    def transfer_data(self, add_dev=True):
        if self.batch_training:
            self.TRAIN = self._transfer2batch(self.TRAIN, self.max_batch_size)
            if add_dev:
                self.DEV = self._transfer2batch(self.DEV, self.max_batch_size)

        if self.load_embedding:
            self.build_word2vec()
        if add_dev:
            self.__transfer2numpy__(self.DEV, self.batch_training)

        self.__transfer2numpy__(self.TRAIN, self.batch_training)
        self.save_data()
