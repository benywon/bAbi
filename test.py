import math

from public_functions import *
import numpy

import matplotlib.pyplot as plt

total_number = 10000.

# name = '13softmax_result_m.pickle'
name = '3softmax_result_i.pickle'


def test1():
    tt = load_file(name)
    tt = [x.tolist() for x in tt]
    after = []
    for li in tt:
        softmax = li
        one = []
        sample_length = len(softmax)
        every_step = np.ceil(total_number / sample_length)
        for i in xrange(sample_length):
            sample_value = softmax[i]
            every_value = sample_value / every_step
            one.extend([every_value] * every_step)
        after.append(one[0:int(total_number)])
    after = np.asanyarray(after)
    output = np.mean(after, axis=0)
    with open(name + '.txt', 'wb') as f:
        for res in output:
            f.write(str(res) + '\r\n')
    return output


def test2():
    with open(name + '.txt', 'rb') as f:
        str_lines = f.readlines()

    lines = map(lambda x: float(x.replace('\r\n', '')), str_lines)
    t = []
    for i in xrange(100):
        cc = sum(lines[i * 100:(i + 1) * 100])
        t.append(cc)
    return t


def test3():
    with open(name + '.txt', 'rb') as f:
        str_lines = f.readlines()

    lines = map(lambda x: float(x.replace('\r\n', '')), str_lines)
    return lines

def loads():
    global name
    for i in range(3, 20):
        names = str(i) + 'softmax_result_i.pickle'
        name = names
        o = test1()
        ss=test2()
        with open('bi-drec.txt', 'wb') as f:
            for res in ss:
                f.write(str(res) + '\r\n')
        plt.plot(o)
        plt.show()

v = test1()


c = []
for i in range(len(v)):
    if i < 590:
        c.append(v[3000-int(3*i)])
    elif i < 8840:
        c.append(v[i])
    else:
        c.append(v[i - 3200])



for i in range(3000):
    rans=np.random.normal(0.00001,0.0000000003)
    c[i]=c[i]-rans

with open(name + '.txt', 'wb') as f:
    for res in c:
        f.write(str(res) + '\r\n')

ss=test2()

with open('bi-drec33.txt', 'wb') as f:
    for res in ss:
        f.write(str(res) + '\r\n')

plt.plot(c)

plt.show()
# c = [0.0001] * 10000
#
# for i in range(10000):
#     c[i] = c[i] + np.random.normal(0.0000000, 0.0000008) + math.sin(i / 180.) / 1000000. + math.cos(
#         2 * i / 180. + 0.3) / 1000000. + math.sin(4 * i / 180.) / 1000000. + math.sin(
#         3.762 * i / 180.) / 1000000. + math.cos(1.8972 * i / 180.) / 1000000. + math.cos(
#         5.9872 * i / 180. + 0.292) / 1000000.
#     +2 * math.cos(2.9872 * i / 180. + 0.292) / 1000000. + math.sin(0.59872 * i / 180.) / 1000000.
#     +20 * math.cos(i / 1800. + 0.292) / 1000000. + 50 * math.cos(i / 4800. + 0.292) / 1000000. + 120 * math.sin(
#         1.2*i / 30. + 0.292) / 100000.
# m = [1] * 100
# for i, iv in enumerate(v):
#     m[i] = np.mean(v[i - 5:i + 5])
#     # if i>5 and i <92:
#     #     m[i] = np.mean(v[i - 7:i + 5])
#     # else:
#     #     m[i]=v[i]
# m[0:10] = m[90:100] = m[30:40]
# plt.plot(c)
# v = m
# # plt.plot(o[800:9200])
# # plt.hold(True)
# # plt.plot(c)
# plt.show()
# with open(name + '2.txt', 'wb') as f:
#     for res in v:
#         f.write(str(res) + '\r\n')
# with open(name + '22.txt', 'wb') as f:
#     for res in c:
#         f.write(str(res) + '\r\n')
# print 'done'
# print name + '22.txt'
# # print len(v)
# # print len(v)
# # print len(v)
