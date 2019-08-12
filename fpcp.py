import numpy as np
from scipy.sparse.linalg import svds
import cv2 as cv
import math

def shrink(v, lamb):
    return np.sign(v) * np.maximum(0, np.absolute(v) - lamb)

def fast_pcp(V, lamb, loops=2, rank0=1, rank_threshold=0.01, lambda_factor=1.0):
    # h, w = V.shape
    inc_rank = True

    rank = rank0
    ranks = []
    rhos = []
    ranks.append(rank)
    rhos.append(0)

    # Partial SVD
    # Ulan, Slan, Vlan = lansvd(V, rank, 'L')
    # Ulan, Slan, Vlan = svds(V, rank)
    Ulan, Slan, Vlan = svds(V, rank)
    
    # Current low-rank approximation
    L1 = Ulan @ np.diag(Slan) @ Vlan

    # Shrinkage
    S1 = shrink(V - L1, lamb)

    for k in range(1, loops):
        if inc_rank:
            lamb = lamb * lambda_factor # modify Lambda at each iteration
            rank = rank + 1 # increase rank

        # low rank (partial SVD)
        Ulan, Slan, Vlan = svds(V - S1, rank) # fastest
    
        # current_evals = np.diag(Slan) # extract current evals
        ranks.append(len(Slan)) # save current rank
        rhos.append(Slan[-1] / np.sum(Slan[:-1])) # relative contribution of the last evec
    
        # simple rule to keep or increase the current rank's value
        if rhos[k - 1] < rank_threshold: 
            inc_rank = False
        else:
            inc_rank = True
    
        # Current low-rank approximation
        L1 = Ulan @ np.diag(Slan) @ Vlan
        
        # Shrinkage
        S1 = shrink(V - L1, lamb)
    
    return L1, S1, ranks, rhos

def v3d_to_v2d(V):
    m, n, p = V.shape

    M = np.zeros((m * n, p), dtype=np.float32)
    for i in range(p):
        M[:, i] = V[:, :, i].ravel()

    return M, m, n

def v2d_to_v3d(I, m, n):
    s, p = I.shape
    assert(s == m * n)

    V = np.zeros((m, n, p), dtype=np.float32)
    for i in range(p):
        V[:, :, i] = np.reshape(I[:, i], (m, n))

    return V

def v3d_to_mov(V, fn):
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(fn, fourcc, 30.0, (640, 480))

    for i in range(V.shape[2]):
        out.write(V[:, :, i])

    out.release()
    
if __name__ == "__main__":
    cap = cv.VideoCapture('input/quick.mov')

    # get first 100 frames
    V = []
    for i in range(100):
        ret, frame = cap.read()
        if ret == False:
            break
        V.append(frame)
    V = np.dstack(V)
    
    M, m, n = v3d_to_v2d(V)
    lamb = 1 / math.sqrt(max(M.shape))
    L, S, ranks, rhos = fast_pcp(M, lamb)

    L_3d = v2d_to_v3d(L, m, n)
    S_3d = v2d_to_v3d(S, m, n)

    v3d_to_mov(L_3d.astype(np.uint8), 'low.mov')
    v3d_to_mov(S_3d.astype(np.uint8), 'sparse.mov')