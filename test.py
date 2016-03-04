# -*- coding: utf-8 -*-
import re

__author__ = 'benywon'

from lxml import html

filename = './data/QAsent/test-less-than-40.manual-edit.xml'
all_the_text = open(filename).read()

doc = html.fromstring(all_the_text)

for ele in doc:
    id_number = ele.attrib['id']
    for pos_neg in ele:
        print pos_neg

