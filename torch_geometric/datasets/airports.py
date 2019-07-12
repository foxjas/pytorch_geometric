import os.path
import torch
from torch_geometric.data import InMemoryDataset
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
        name of the dataset (:obj:`"Cora"`,
            :obj:`"CiteSeer"`, :obj:`"PubMed"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        super(Airport, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['']

    @property
    def processed_file_names(self):
        processed_name = "{}_data.pt".format(self.name)
        return processed_name

    def download(self):
        pass

    def process(self):
        data = read_airport_data(self.root, self.name)
        #data = data if self.pre_transform is None else self.pre_transform(data)
        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


def read_airport_data(folder, data_name):
    """
    Reads and processes airport data 
    Returns graph, feature, label tensors,
        train/validation/test masks
    """
    graph_bin_path = os.path.join(folder, "{}_edges.dat".format(data_name))
    feats_path = os.path.join(folder, "{}_feats.dat".format(data_name))
    labels_bin_path = os.path.join(folder, "{}_labels.dat".format(data_name))
 
    data_ready = os.path.isfile(graph_bin_path) and \
            os.path.isfile(feats_path) and \
            os.path.isfile(labels_bin_path)
    if not data_ready:
        prepare_airport_data(folder, data_name)

    # read graph and create graph tensor
    coo = readBinary(graph_bin_path)
    graph_tensor = torch.Tensor(coo).t()
    print("graph_tensor: {}".format(graph_tensor.size()))

    # create features tensor
    feats_data = np.array(readBinary(feats_path))
    #feats_tensor = torch.Tensor(feats_data)
    print("feats_data: {}".format(feats_data.shape))
  
    # process labels 
    #labels_data = torch.Tensor(readBinary(labels_bin_path))
    labels_data = np.array(readBinary(labels_bin_path))
    print("labels: {}".format(len(labels_data)))
    train_validation_test_split(feats_data, labels_data, 0.6, 0.2)


    """
    x = torch.cat([allx, tx], dim=0)
    y = torch.cat([ally, ty], dim=0).max(dim=1)[1]

    train_mask = sample_mask(train_index, num_nodes=y.size(0))
    val_mask = sample_mask(val_index, num_nodes=y.size(0))
    test_mask = sample_mask(test_index, num_nodes=y.size(0))

    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    """
    data = None
    return data


def prepare_airport_data(folder, data_name):
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

    feats_data = ldp_features(graph)
    saveBinary(feats_data, data_name, "feats", folder)

    labels_path = os.path.join(folder, "labels-{}.txt".format(data_name))
    node_labels = readLabels(labels_path)
    # reorder labels according to new graph ordering
    node_labels = reorderLabels(old_new_node_ids, node_labels)
    saveBinary(node_labels, data_name, "labels", folder)


def train_validation_test_split(X, y, train_ratio, valid_ratio):
    """
    Return indices corresponding to stratified train, validation, and
    test splits.
    """

    rest_ratio = 1-train_ratio
    indices = np.arange(len(y))
    X_train, X_rest, y_train, y_rest, ind_train, ind_rest = \
            train_test_split(X, y, indices, test_size=rest_ratio, random_state=24, stratify=y)

    test_ratio = 1-(valid_ratio/rest_ratio) 
    X_valid, X_test, y_valid, y_test, ind_valid, ind_test = \
            train_test_split(X_rest, y_rest, ind_rest, test_size=test_ratio, random_state=24, stratify=y_rest)

    ind_reconstr = list(ind_train) + list(ind_valid) + list(ind_test)
    assert len(set(ind_reconstr)) == len(indices)
    return ind_train, ind_valid, ind_test



def sample_mask(index, num_nodes):
    mask = torch.zeros((num_nodes, ), dtype=torch.uint8)
    mask[index] = 1
    return mask
