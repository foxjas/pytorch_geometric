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

def cluster_kmeans(X, cr):
    """
    Set clusters as ratio of number of uniqued data points in X 
    """
    if type(X) is not np.ndarray:
        X = X.numpy()

    n = len(np.unique(X, axis=0))
    nc = int(cr*n)
    kmeans = KMeans(nc, max_iter=100, random_state=0).fit(X)
    preds = kmeans.predict(X)
    labels = torch.Tensor(preds).long()
    #unique, counts = np.unique(labels, return_counts=True)
    #class_counts = dict(zip(unique, counts))
    #print("total class counts: {}".format(class_counts))
    return labels


def clusterDegree(X, cr):
    """
    Label based on degree percentile. Percentiles
        correspond to number of clusters.
    - X: normalized features
    - cr: ratio of clusters with respect to unique degrees 
    """
    if type(X) is not np.ndarray:
        X = X.numpy()
    
    n_degs_unique = len(np.unique(X[:,0]))
    #print("n_deg_unique: {}".format(n_degs_unique))
    n = X.shape[0] 
    nc = int(n_degs_unique*cr)
    #print("nc: {}".format(nc))
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

def clusterRandom(X, cr):
    """
    Randomly assigns labels, in range of number of clusters.
    Does not guarantee uniform class distribution
    """
    if type(X) is not np.ndarray:
        X = X.numpy()
    nc = int(cr*X.shape[0])
    labels = np.random.randint(0, nc, size=X.shape[0])
    unique, counts = np.unique(labels, return_counts=True)
    class_counts = dict(zip(unique, counts))
    labels = torch.Tensor(labels).long()
    return labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('data_dir', help='Data directory')
    parser.add_argument('data_name', help='Dataset name')
    #parser.add_argument('n_clusters', type=int, help='Number of clusters')
    parser.add_argument('clusters_ratio', type=float, help='Number of clusters')
    parser.add_argument('type', help='Type of cluster labels to generate')
    parser.add_argument('--verbose', default=False, action='store_true', help='Print additional training information')
    args = parser.parse_args()

    dataset = Airport(args.data_dir, args.data_name, "LDP", load_data=False)
    if args.type == "degree":
        clusterDegreeByRatio(dataset.x, args.clusters_ratio)
    elif args.type == "kmeans":
        cluster_kmeans(dataset.x, args.clusters_ratio)
    elif args.type == "random":
        clusterRandom(dataset.x, args.clusters_ratio)

