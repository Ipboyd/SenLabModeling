import time
import pythranTestingCPP as pyt
import numpy as np

v = np.ones((10000,))
A = np.ones((10000,100,1,2))*10
out = np.zeros((1000,100,1))

#CPP Version

start = time.perf_counter()

for k in range(1000):
    B = pyt.mul_broadcast(v,A) #CPP verison

elapsed = time.perf_counter() - start
print(f"{elapsed*1000:.2f} ms")


#Python only version

start = time.perf_counter()

for k in range(1000):
    n = v.shape[0]
    B = v.reshape((n, 1, 1)) * A[:, :, :, -1]

elapsed = time.perf_counter() - start
print(f"{elapsed*1000:.2f} ms")