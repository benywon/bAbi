# -*- coding: utf-8 -*-
import os

from public_functions import *

__author__ = 'benywon'


def add_word2id(word, word2id):
    if word not in word2id:
        word2id[word] = len(word2id)


class bAbiDataPreprocess:
    def __init__(self, path=None, number_int=1000, re_load=False):
        if number_int == 1000:
            number = ''
            self.number = '1k'
        else:
            number = '-10k'
            self.number = '10k'
        self.objFilepath = '../data/bAbi/data_' + self.number + '.pickle'
        if path is not None:
            self.filePath = path
        else:
            self.filePath = '../data/bAbi'
        self.filePath += '/en' + number + '/'
        self.word2id = {'_NONE_': 0}
        self.data = {}
        self.load_data(re_load=re_load)

    def load_data(self, re_load=False):
        if re_load:
            self.reload_file()
        else:
            [self.word2id, self.data] = load_file(self.objFilepath)

    def reload_file(self):
        print('start process data')
        file_map = {}
        for parent, dirnames, filenames in os.walk(self.filePath):
            for filename in filenames:
                number = find_number(filename)
                if number not in file_map:
                    file_map[number] = []
                file_map[number].append(self.filePath + filename)
        for fileID in file_map:
            self.__load_file__(fileID, file_map[fileID])
        print('process data done')
        dump_file([self.word2id, self.data], self.objFilepath)

    def __line2id(self, line):
        question_word = line.split(' ')
        question_list = []
        for word in question_word:
            add_word2id(word, self.word2id)
            question_list.append(self.word2id[word])
        return question_list

    def __load_file__(self, id, path_list):
        def parseQuestion(line):
            pattern = re.compile(r'(\d+) (.*)\?\s(.*?)\s(.+)')
            res = re.findall(pattern, line)[0]
            question = res[1]
            answer = res[2].replace('\t', '')
            support_id = res[3].split(' ')
            question_id = self.__line2id(question)
            answer_id = self.__line2id(answer)
            return [question_id, answer_id, map(lambda x: int(x), support_id[0:-1])]

        one_data = {}
        for path in path_list:
            with open(path, 'r') as f:
                one_data_set = []
                doc_id = 0
                sentences = []
                pre_id = 0
                if r'test' in path:
                    purpose = 'test'
                else:
                    purpose = 'train'
                for line in f.readlines():
                    sent_id = find_number(line)
                    if sent_id < pre_id:
                        sentences = []
                        doc_id += 1
                    line = clean_string(line)
                    if r'?' in line:
                        question = parseQuestion(line)
                        one_data_set.append([doc_id, list(sentences), question])
                    else:
                        sent = line.split(' ')[1:]
                        # sentences.append(sent)
                        for word in sent:
                            add_word2id(word, self.word2id)
                        sentences.append(map(lambda x: self.word2id[x], sent))

                    pre_id = sent_id
                one_data[purpose] = one_data_set
                print('load:' + path + 'done!')
        self.data[id] = one_data


if __name__ == '__main__':
    bAbiDataPre = bAbiDataPreprocess(number_int=10000, re_load=True)
    print 'done'
