from public_functions import *
import numpy

# import matplotlib.pyplot as plt
total_number = 1000.


def test1():
    tt = load_file('softmax_result555.pickle')
    tt = [x.tolist() for x in tt]
    after = []
    for li in tt:
        softmax = li[0]
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
    with open('555.txt', 'wb') as f:
        for res in output:
            f.write(str(res) + '\r\n')

    print 'done'


def test2():
    with open('5.txt', 'rb') as f:
        str_lines = f.readlines()

    lines = map(lambda x: float(x.replace('\r\n', '')), str_lines)
    t = []
    for i in xrange(100):
        cc = sum(lines[i * 10:(i + 1) * 10])
        t.append(cc)
    return t


test1()
v=test2()
with open('5556.txt', 'wb') as f:
    for res in v:
        f.write(str(res) + '\r\n')

print 'done'
print len(v)
print len(v)
print len(v)
