import os
import sys

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
