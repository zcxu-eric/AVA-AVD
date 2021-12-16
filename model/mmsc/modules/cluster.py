import torch
import numpy as np


class AVDCluster():
    '''
    AHC clustering
    '''
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold
    
    def fit_predict(self, similarity):
        if isinstance(similarity, torch.Tensor):
            similarity = similarity.cpu().numpy()
        dist = -similarity
        return self.__AHC__(dist, threshold=self.threshold)

    def __AHC__(self, dist, threshold=0.3):
        dist[np.diag_indices_from(dist)] = np.inf
        clsts = [[i] for i in range(len(dist))]
        while True:
            mi, mj = np.sort(np.unravel_index(dist.argmin(), dist.shape))
            if dist[mi, mj] > -threshold:
                break
            dist[:, mi] = dist[mi,:] = (dist[mi,:]*len(clsts[mi])+dist[mj,:] \
                                        *len(clsts[mj]))/(len(clsts[mi])+len(clsts[mj]))
            dist[:, mj] = dist[mj,:] = np.inf
            clsts[mi].extend(clsts[mj])
            clsts[mj] = None
        labs= np.empty(len(dist), dtype=int)
        for i, c in enumerate([e for e in clsts if e]):
            labs[c] = i
        return labs