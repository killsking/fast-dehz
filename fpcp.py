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

def flatten(V):
    # 3d to 2d, 0-255 to 0-1
    m, n, p = V.shape
    
    M = np.zeros((m * n, p), dtype=np.float32)
    for i in range(p):
        frame = V[:, :, i]
        frame = cv.normalize(frame.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
        M[:, i] = frame.ravel()

    return M, m, n

def restore(I, m, n):
    # 2d to 3d, 0-1 to 0-255
    s, p = I.shape
    assert(s == m * n)

    V = np.zeros((m, n, p), dtype=np.uint8)
    for i in range(p):
        frame = np.reshape(I[:, i], (m, n))
        frame = cv.normalize(frame, None, 0.0, 255.0, cv.NORM_MINMAX).astype(np.uint8)
        V[:, :, i] = frame

    return V

def hard_threshold(S):
    value = 0.5 * (np.std(S) ** 2)
    S2 = 0.5 * np.square(S)
    ret, O = cv.threshold(S2, value, 1.0, cv.THRESH_BINARY)
    return O

def save_mov(V, fn):
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(fn, fourcc, 30.0, (640, 480), 0)

    for i in range(V.shape[2]):
        # im = V[:, :, i]
        # cv.imshow(fn, im)
        # if cv.waitKey(1) & 0xFF == ord('q'):
        #     break
        out.write(V[:, :, i])

    out.release()
    
if __name__ == "__main__":
    cap = cv.VideoCapture('input/quick.mov')

    # get first 100 frames
    V = []
    # while (cap.isOpened()):
    for i in range(100):
        ret, frame = cap.read(0)
        if ret == False:
            break
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        V.append(frame)
    V = np.dstack(V)
    
    M, m, n = flatten(V)
    lamb = 1 / math.sqrt(max(M.shape))
    L, S, ranks, rhos = fast_pcp(M, lamb)

    # outlier
    O = hard_threshold(S)

    L_3d = restore(L, m, n)
    S_3d = restore(S, m, n)
    O_3d = restore(O, m, n)

    save_mov(L_3d, 'low.mov')
    save_mov(S_3d, 'sparse.mov')
    save_mov(O_3d, 'outlier.mov')