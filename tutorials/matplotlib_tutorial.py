#!/usr/bin/env python
# ******************************************************
# Author       : Hanfei Ye
# Last modified: 2020-12-30 14:32
# Email        : hanfei.ye@cern.ch
# Filename     : matplotlib_tutorial.py
# Description  : 
# ******************************************************
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 3*np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.figure(0)
plt.plot(x, y_sin)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Sine and Cosine")
plt.savefig("./matplotlib_sin.png")

plt.figure(1)
plt.plot(x, y_cos)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Sine and Cosine")
plt.savefig("./matplotlib_cos.png")
#plt.show()  
