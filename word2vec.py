# -*- coding: utf-8 -*-
import cPickle
import numpy as np
import os

from public_functions import dump_file

__author__ = 'beny'


class word2vector():
    size = 0

    def __init__(self,
                 size=100):
        self.OOV_vec = np.random.normal(0, 0.2, size)
        self.size = size
        self.word2vec = {}
        basepath = './data/word2vec/'
        if size == 50:
            self.path = basepath + 'cbow_all_shuffle_v50_w2_ns5_100'
            self.objpath = basepath + 'word2v50.pickle'
        elif size == 100:
            self.path = basepath + 'vec_wbn.txt_3'
            self.objpath = basepath + 'word2v100.pickle'
        elif size == 300:
            self.path = basepath + 'GoogleNews-vectors-negative300.bin'
            self.objpath = basepath + 'word2v300.pickle'
        if os.path.exists(self.objpath):
            self.load_obj()
        else:
            self.load_data()

    def load_obj(self):
        print 'load word2vec functin'
        f = file(self.objpath, 'rb')
        self.word2vec = cPickle.load(f)
        f.close()

    def load_data(self):
        if self.size == 50 or self.size == 100:
            with open(self.path, 'rb') as f:
                for line in f:
                    tokens = line.split()
                    word = tokens[0]
                    w_vecstr = np.array(tokens[1:], dtype='float32')
                    self.word2vec[word] = w_vecstr
            dump_file(self.word2vec, self.objpath)
        elif self.size == 300:
            with open(self.path, "rb") as f:
                header = f.readline()
                vocab_size, layer1_size = map(int, header.split())
                binary_len = np.dtype('float32').itemsize * layer1_size
                for line in xrange(vocab_size):
                    word = []
                    while True:
                        ch = f.read(1)
                        if ch == ' ':
                            word = ''.join(word)
                            break
                        if ch != '\n':
                            word.append(ch)
                    self.word2vec[word] = np.fromstring(f.read(binary_len), dtype='float32')
            dump_file(self.word2vec, self.objpath)

    def returnVec(self, word):
        if isinstance(word, list) and len(word) > 1:
            res = []
            for x in word:
                if x in self.word2vec:
                    res.append(self.word2vec[x])
                else:
                    res.append(self.OOV_vec)
            return res
        else:
            if (isinstance(word, list)):
                word = word[0]
            if word in self.word2vec:
                vec = self.word2vec[word]
            else:
                vec = self.OOV_vec
            return [vec]

    def returnWordVec(self, word):
        if word in self.word2vec:
            vec = self.word2vec[word]
        else:
            vec = self.OOV_vec
        return vec


def load_word2vec300withoutLOOP(word2id,):
    wordEmbedding=[]
    print 'load word2vec from miklov file:size 300'
    fname = './data/word2vec/GoogleNews-vectors-negative300.bin'
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in word2id:
                wordEmbedding[word2id[word]] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    print 'load word2vec done'
    return wordEmbedding
