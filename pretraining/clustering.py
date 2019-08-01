import os.path
import sys
sys.path = [os.path.expanduser("~/pytorch_geometric")] + sys.path
import argparse
from pretraining.airports import Airport
from pretraining.models import MLP2,MLP3,GIN,train_step,test_step
import numpy as np
import torch
from sklearn.cluster import KMeans
from pprint import pprint

def cluster_kmeans(X, nc):
    """
    Set clusters as ratio of number of nodes?
    """
    if type(X) is not np.ndarray:
        X = X.numpy()
    kmeans = KMeans(nc, max_iter=100).fit(X)
    labels = torch.Tensor(kmeans.labels_).long()

    #unique, counts = np.unique(labels, return_counts=True)
    #class_counts = dict(zip(unique, counts))
    #print("total class counts: {}".format(class_counts))
    return labels


def clusterDegree(X, nc):
    """
    Label based on degree percentile. Percentiles
        correspond to number of clusters.
    - X: normalized features
    - nc: number of clusters
    """
    if type(X) is not np.ndarray:
        X = X.numpy()
    
    n = X.shape[0] 
    sorted_indices = [i for i, x in sorted(enumerate(X), key=lambda v: v[1][0])]
    classes = np.full(n, -1)
    steps = int(n/(nc-1))
    for i in range(nc-1):
        classes[i*steps:(i+1)*steps] = i
    classes[(i+1)*steps:] = nc-1

    labels = np.full(n, -1)
    for j in range(n):
        v_id = sorted_indices[j]
        c = classes[j]
        labels[v_id] = c
    
    labels = torch.Tensor(labels).long()
    return labels


def clusterTwoLevel(X, nc1, nc2):
    pass 

def clusterRandom(X, nc):
    """
    Randomly assigns labels, in range of number of clusters.
    Does not guarantee uniform class distribution
    """
    labels = np.random.randint(0, nc, size=X.shape[0])
    unique, counts = np.unique(labels, return_counts=True)
    class_counts = dict(zip(unique, counts))
    labels = torch.Tensor(labels).long()
    return labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('data_dir', help='Data directory')
    parser.add_argument('data_name', help='Dataset name')
    parser.add_argument('n_clusters', type=int, help='Number of clusters')
    parser.add_argument('type', help='Type of cluster labels to generate')
    parser.add_argument('--verbose', default=False, action='store_true', help='Print additional training information')
    args = parser.parse_args()

    dataset = Airport(args.data_dir, args.data_name, "LDP", load_data=False)
    if args.type == "degree":
        clusterDegree(dataset.x, args.n_clusters)
    elif args.type == "kmeans":
        cluster_kmeans(dataset.x, args.n_clusters)
    elif args.type == "random":
        clusterRandom(dataset.x, args.n_clusters)

