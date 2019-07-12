import numpy as np
from graph import readEdgelist
from utils import saveBinary

import argparse
import os


def ldp_features(g):
    """
    Generate feature matrix according to following attributes, per node:
        - Node degree
        - Min, max, mean, and standard deviation of immediate neighborhood

    See https://github.com/Chen-Cai-OSU/LDP
    """

    def normalize(features):
        """ 
        Normalizes each feature by z-score transform 
        Other transformations possible
        """
        feat_std = np.std(features, axis=0) 
        feat_mean = np.mean(features, axis=0)
        features = (features-feat_mean)/feat_std
        return features

    ldp_stats = []
    nodes = list(sorted(g.nodes))
    for u in nodes:
        deg_u = g.degree[u]
        neighbors = [g.degree[v] for v in g.neighbors(u)]
        if deg_u:
            neigh_mean = np.mean(neighbors)
            neigh_min = np.amin(neighbors)
            neigh_max = np.max(neighbors)
            neigh_std = np.std(neighbors)
        else:
            neigh_mean, neigh_min, neigh_max, neigh_std = 0,0,0,0
        ldp_stats.append((deg_u, neigh_min, neigh_max, neigh_mean, neigh_std))
    ldp_stats = np.array(ldp_stats)
    data = normalize(ldp_stats)

    return data 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates feature vectors for graphs')

    parser.add_argument('edgelist', help='Graph edgelist (0 indexed) to generate vectors from')
    parser.add_argument('--save', dest='save_dir', default=None, help='If enabled, save features to directory as binary')

    args = parser.parse_args()

    g = readEdgelist(args.edgelist)

    features = ldp_features(g)
    if args.save_dir:
        graph_parse = args.edgelist.split("/")[-1]
        graph_name = graph_parse.split(".")[0].strip()
        #print(graph_name)
        saveBinary(features, graph_name, "feats", args.save_dir) 
    
