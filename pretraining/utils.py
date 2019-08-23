import os
import sys
from pprint import pprint

try:
    import cPickle as pickle
except ImportError:
    import pickle

def saveBinary(m, data_name, data_type, outdir):
    """
    m: object to pickle
    data_name: name of dataset
    data_type: type of data
    outdir: directory path
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outPath = os.path.join(outdir, "{}_{}.dat".format(data_name, data_type))
    print(outPath)
    with open(outPath, 'wb') as outf:
        pickle.dump(m, outf)
 

def readBinary(fPath):
    """
    Reads binary (pickled) and returns object

    """
    with open(fPath, 'rb') as f:
        if sys.version_info > (3, 0):
            out = pickle.load(f, encoding='latin1')
        else:
            out = pickle.load(f)

    return out


def readLabels(fPath, skip=1):
    """
    Reads node labels from file, of (node, label) format.
    Returns list of (node, label) pairs. 
    """
    node_labels = []
    with open(fPath, 'r') as f:
        for _ in range(skip):
            f.readline()

        for l in f:
            node_id, label = map(int, l.strip().split(" "))    
            node_labels.append((node_id, label))

    return node_labels


def reorderLabels(old_new_ids, node_labels):
    """
    Reorder labels according to node id mapping. 
    Returns list of labels, with indices
        corresponding to ids. 
    """
    new_labels = [-1 for _ in range(len(node_labels))]
    id_map = {old_id: new_id for old_id, new_id in old_new_ids}
    for v, l in node_labels:
        new_labels[id_map[v]] = l

    return new_labels


def compressLabels(labels):
    """ 
    Compresses label range to be in [0, C-1]
    where C is the number of classes
    """
    unique = list(set(labels))
    C = len(unique)
    old_new_map = {val: key for key, val in enumerate(sorted(unique))}
    new_labels = [old_new_map[l] for l in labels]
    return new_labels

