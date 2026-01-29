import numpy as np
import hnswlib
from numpy import linalg as LA

def cosine_sim(x,y):
    return np.dot(x,y)/LA.norm(x)/LA.norm(y)

class mol:
    def __init__(self,smiles0,smiles1,feat):
        self.smiles0=smiles0
        self.smiles1=smiles1
        self.feat=feat

class infogain:
    def __init__(self, initial_points=None, n=3000000, keep_seed=False, submodular_k=4,dim=128):
        self.clusters = None
        self.target_n = n
        self.current_n = 0
        self.keep_seed = keep_seed
        self.dim = dim
        self.submodular_k = submodular_k

        if initial_points: 
            self.data = initial_points.copy()
            self.current_n = len(initial_points)
            if keep_seed:
                self.seeded_num = len(initial_points)
            self.dim = len(initial_points[0].feat)
        else:
            self.data = []
        self.submodular_gain = [(1)]*len(self.data)
        # initialize HNSW index
        self.knn_graph = hnswlib.Index(space='cosine', dim=self.dim)
        self.knn_graph.init_index(max_elements=n, ef_construction=100, M=48, allow_replace_deleted = False)
        self.precluster(initial_points)
        self.knn_graph.set_ef(32)

    def precluster(self, initial_points):
    # Starting from some initial points (the cleaner the better) to do online selection
        if initial_points is None or initial_points==[]: return
        for idx,data in enumerate(self.data):
            data.index = idx

        for idx,data in enumerate(self.data):
            self.submodular_gain[idx] = self.submodular_func(data, True)
            self.knn_graph.add_items(data.feat, idx)

    def submodular_func(self, data, skip_one=False):
        if self.knn_graph.get_current_count()==0:
            return (1.)
        k = min(self.knn_graph.get_current_count(), self.submodular_k)
        near_label,near_distances = self.knn_graph.knn_query(data.feat, k)
        return np.mean(near_distances)

    def add_item(self, data):
        data.index = self.current_n
        self.data.append(data)
        self.knn_graph.add_items(data.feat, self.current_n)
        self.current_n+=1

    def replace_item(self, data, index):
        # Not used in current work but provide for future extension on replacing samples
        data_to_rep = self.data[index]
        n_index = data_to_rep.index
        data.index = self.current_n
        self.knn_graph.mark_deleted(n_index)
        self.knn_graph.add_items(data.I_feat, self.current_n, replace_deleted = True)
        self.data[index] = data
        self.current_n+=1

    def process_item(self, data,):
        # find near clusters
        # go into nearest clusters to search near neighbour
        # calculate corresponding threshold to decide if try to add or not
        gain = self.submodular_func(data)
        self.add_item(data)
        self.submodular_gain.append(gain)

    def final_gains(self):
        return self.submodular_gain
