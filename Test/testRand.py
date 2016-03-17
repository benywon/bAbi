# -*- coding: utf-8 -*- 
__author__ = 'benywon'

import numpy as np
import matplotlib.pyplot as plt
low = 1 / 20000.
high = 1 * 4 / 20000.
step =(high - low) / 10000.
c = []
for i in range(10000):
    base = low + i * step
    rans =np.random.normal(loc=0,scale=0.2*low)
    c.append(base + rans)
plt.plot(c)
plt.show()