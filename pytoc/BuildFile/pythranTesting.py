import numpy as np
#pythran export mul_broadcast(float64[:], float64[:,:,:,:])
def mul_broadcast(v, a):
    n, m, k, t = a.shape
    out = np.empty((n, m, k), dtype=a.dtype)
    tlast = t - 1

    for i in range(n):
        out[i, :, :] = v[i] * a[i, :, :, tlast]
    return out




# Very slow?
# def mul_broadcast(v, A, out):
#     N, M, K, T = A.shape
#     for i in range(N):
#         vi = v[i]
#         for j in range(M):
#             for k in range(K):
#                 out[i, j, k] = vi * A[i, j, k, T - 1]
#     return out



#Approximnately the same speed as python
# def mul_broadcast(v, a):
#     # v: (n,), a: (n, m, k, t)
#     n = v.shape[0]
#     return v.reshape((n, 1, 1)) * a[:, :, :, -1]
