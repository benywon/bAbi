# -*- coding: utf-8 -*-
from collections import OrderedDict

from dataPreprocess import dataPreprocess
from lxml import html

from public_functions import *

__author__ = 'benywon'


class QASentdataPreprocess(dataPreprocess):
    def __init__(self,
                 use_clean=False,
                 **kwargs):
        dataPreprocess.__init__(self, **kwargs)
        self.use_clean = use_clean
        self.path = './data/QAsent/'
        append_str = '_batch' if self.batch_training else ''
        append_str += '_clean' if self.use_clean else ''
        self.data_pickle_path = self.path + 'QAsent' + append_str + '.pickle'
        if self.reload:
            self.build_data_set()
        else:
            self.load_data()
        self.calc_data_stat()
        self.dataset_name = 'QASent'

    def build_data_set(self):
        print 'start loading data from original file'
        if self.use_clean:
            trainfilepath = self.path + 'train-less-than-40.manual-edit.xml'
        else:
            trainfilepath = self.path + 'train2393.cleanup.xml'
        testfilepath = self.path + 'test-less-than-40.manual-edit.xml'
        devfilepath = self.path + 'dev-less-than-40.manual-edit.xml'

        def get_one_set(filepath, train=True):
            print 'process:' + filepath
            all_the_text = open(filepath).read()
            doc = html.fromstring(all_the_text)
            target = []
            data = []
            for ele in doc:
                one_question = {'right': [], 'wrong': []}
                for pos_neg in ele:
                    tag = pos_neg.tag
                    content = pos_neg.text
                    content = content.split('\n')[1]
                    content = clean_string(content)
                    if tag == 'question':
                        one_question['question'] = self.get_sentence_id_list(content)
                    elif tag == 'positive':
                        one_question['right'].append(self.get_sentence_id_list(content))
                    elif tag == 'negative':
                        one_question['wrong'].append(self.get_sentence_id_list(content))
                if len(one_question['right']) ==0 or len(one_question['wrong'])==0:
                    continue
                data.append(one_question)
            if train:
                q = []
                yes = []
                no = []
                for one_question in data:
                    question = one_question['question']
                    pos = one_question['right']
                    neg = one_question['wrong']
                    for po in pos:
                        for ne in neg:
                            q.append(question)
                            yes.append(po)
                            no.append(ne)
                target.append([x[0:self.Max_length] for x in q])
                target.append([x[0:self.Max_length] for x in yes])
                target.append([x[0:self.Max_length] for x in no])
                return target
            else:
                for one_question in data:
                    one_patch = []
                    question = one_question['question']
                    pos = one_question['right']
                    neg = one_question['wrong']
                    for po in pos:
                        one_patch.append([question, po, 1])
                    for ne in neg:
                        one_patch.append([question, ne, 0])
                    target.append(one_patch)
                return [[[np.asarray(x[0], dtype='int32'), np.asarray(x[1], dtype='int32'), x[2]] for x in t] for t
                        in target]

        self.TRAIN = get_one_set(trainfilepath, train=True)
        self.DEV = get_one_set(devfilepath, train=True)
        self.TEST = get_one_set(testfilepath, train=False)
        self.transfer_data()
        print 'load data done'


if __name__ == '__main__':
    c = QASentdataPreprocess(reload=True, batch_training=False, use_clean=False)
