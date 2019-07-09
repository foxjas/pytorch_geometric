#!/usr/bin/env python3

import networkx 
import numpy as np

from sklearn import preprocessing

import argparse

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
    nodes = list(g.nodes) # ordering?
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

def save_out(m, o):
    with open(o, 'w') as f:
        for row in m:
            f.write(','.join(map(lambda x: str(x), row)))
            f.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates feature vectors for graphs')

    parser.add_argument('edgelist', help='Graph edgelist (0 indexed) to generate vectors from')
    parser.add_argument('--save_features', default=False, help='If enabled save as ? files', action='store_true')

    args = parser.parse_args()

    g = networkx.read_edgelist(args.edgelist)

    features = ldp_features(g)
    #X, Y = get_xy(g, args.walks_per, args.balance, args.alg)
    
    """
    if args.save_text:
        save_out(X, '%s.walks' % args.out)
        save_out(Y, '%s.pr' % args.out)
    else:
        np.save('%s_X' % args.out, X)
        np.save('%s_Y' % args.out, Y)
    """
