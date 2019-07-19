import os.path
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from pretraining.gen_features import *
from utils import * 
from graph import * 
from sklearn.model_selection import train_test_split 

class Airport(InMemoryDataset):
    r"""The airport network datasets "europe", "brazil" and "usa" from the
    `"Learning Node Representations from Structural Identity" paper.
    Training, validation and test splits are given by binary masks.

    Args:
        root (string): Root directory where the dataset should be saved.
        name of the dataset 
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    Functionality:
        - set train/validation/test split ratio, and new masks
        - ignore .dat in processed/
        - set new labels (e.g. from pretrain)
    """

    def __init__(self, root, name, feature_type, load_data=False, transform=None, pre_transform=None):
        self.name = name
        self.feature_type = feature_type
        self.data_mutable = None
        super(Airport, self).__init__(root, transform, pre_transform)
        if load_data:
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.process()  
           
    @property
    def raw_file_names(self):
        return ['']

    @property
    def processed_file_names(self):
        """ 
        Checks if data file exists. If so, skip process 
        """
        processed_name = "{}_{}_data.pt".format(self.name, self.feature_type)
        return processed_name

    def download(self):
        pass

    def process(self):
        self.data_mutable = read_airport_data(self.root, self.name, self.feature_type)
        self.data, self.slices = self.collate([self.data_mutable])
        torch.save((self.data, self.slices), self.processed_paths[0])

    def update_data():
        """ 
        self.mutable_data = ?
        self.mutable_data.y = new_y 
        """
        pass

    def __repr__(self):
        return '{}()'.format(self.name)


def read_airport_data(folder, data_name, feature_type):
    """
    Reads and processes airport data 
    Returns graph, feature, label tensors,
        train/validation/test masks
    """
    graph_bin_path = os.path.join(folder, "{}_edges.dat".format(data_name))
    labels_bin_path = os.path.join(folder, "{}_labels.dat".format(data_name))
    feats_path = os.path.join(folder, "{}-{}_feats.dat".format(data_name, feature_type))
 
    data_ready = os.path.isfile(graph_bin_path) and \
            os.path.isfile(feats_path) and \
            os.path.isfile(labels_bin_path)
    if not data_ready:
        prepare_airport_data(folder, data_name, feature_type)

    # read graph and create graph tensor
    coo = readBinary(graph_bin_path)
    graph_tensor = torch.Tensor(coo).t().int()
    print("graph_tensor: {}".format(graph_tensor.size()))

    # create features tensor
    feats_data = np.array(readBinary(feats_path))
    print("feats_data: {}".format(feats_data.shape))
  
    # process labels 
    labels_data = np.array(readBinary(labels_bin_path), dtype=np.uint8)
    print("labels: {}".format(len(labels_data)))
    train_index, val_index, test_index = train_validation_test_split(feats_data, labels_data, 0.6, 0.2)
    
    x = torch.Tensor(feats_data)
    y = torch.Tensor(labels_data).long()
    edge_index = graph_tensor.long() 
    train_mask = sample_mask(train_index, num_nodes=y.size(0))
    val_mask = sample_mask(val_index, num_nodes=y.size(0))
    test_mask = sample_mask(test_index, num_nodes=y.size(0))
        
    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


def prepare_airport_data(folder, data_name, feature_type):
    """
    Read and process raw files, and save to binary:
        - COO edge list
        - Generated features
        - Labels
    """
    graph_path = os.path.join(folder, "{}.edgelist".format(data_name))
    graph = readEdgelist(graph_path) # singleton case?
    graph, old_new_node_ids = relabelGraph(graph) 
    coo = graphToCOO(graph) 
    saveBinary(coo, data_name, "edges", folder) # TODO: replace with single path argument

    if feature_type == "LDP":
       feats_data = ldp_features(graph)
    elif feature_type == "degree":
       feats_data = degreeOnlyFeatures(graph) 
    base_name = data_name + "-{}".format(feature_type)
    saveBinary(feats_data, base_name, "feats", folder)

    labels_path = os.path.join(folder, "labels-{}.txt".format(data_name))
    node_labels = readLabels(labels_path)
    # reorder labels according to new graph ordering
    node_labels = reorderLabels(old_new_node_ids, node_labels)
    saveBinary(node_labels, data_name, "labels", folder)


def train_validation_test_split(X, y, train_ratio, valid_ratio):
    """
    Return indices corresponding to stratified train, validation, and
    test splits.
    Note: no random seed is set, so splits may differ on different calls
        to this function for the same data.
    """
    
    # TODO: print total class counts
    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique, counts))
    print("total class counts: {}".format(class_counts))

    rest_ratio = 1-train_ratio
    indices = np.arange(len(y))
    X_train, X_rest, y_train, y_rest, ind_train, ind_rest = \
            train_test_split(X, y, indices, test_size=rest_ratio, stratify=y)

    test_ratio = 1-(valid_ratio/rest_ratio) 
    X_valid, X_test, y_valid, y_test, ind_valid, ind_test = \
            train_test_split(X_rest, y_rest, ind_rest, test_size=test_ratio, stratify=y_rest)

    """
    unique, counts = np.unique(y_valid, return_counts=True)
    class_counts = dict(zip(unique, counts))
    print("validation class counts: {}".format(class_counts))

    unique, counts = np.unique(y_test, return_counts=True)
    class_counts = dict(zip(unique, counts))
    print("test class counts: {}".format(class_counts))
    """

    ind_reconstr = list(ind_train) + list(ind_valid) + list(ind_test)
    assert len(set(ind_reconstr)) == len(indices)
    return ind_train, ind_valid, ind_test


def sample_mask(index, num_nodes):
    """
    Converts list of indices into boolean mask (over num_nodes)
    """
    mask = torch.zeros((num_nodes, ), dtype=torch.uint8)
    mask[index] = 1
    return mask
