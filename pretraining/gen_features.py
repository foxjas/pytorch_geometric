#!/usr/bin/env python3

import networkx 
import numpy as np

from sklearn import preprocessing

import argparse

def ldp_features(g):
    pass

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

    #X, Y = get_xy(g, args.walks_per, args.balance, args.alg)
    
    """
    if args.save_text:
        save_out(X, '%s.walks' % args.out)
        save_out(Y, '%s.pr' % args.out)
    else:
        np.save('%s_X' % args.out, X)
        np.save('%s_Y' % args.out, Y)
    """
