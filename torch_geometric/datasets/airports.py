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

    Functionality:
        - set train/validation/test split ratio, and new masks
        - ignore .dat in processed/
        - set new labels (e.g. from pretrain)
    """

    def __init__(self, root, name, feature_type, load_data=False):
        self.name = name
        self.feature_type = feature_type
        self.graph_bin_path = os.path.join(root, "{}_edges.dat".format(name))
        self.labels_bin_path = os.path.join(root, "{}_labels.dat".format(name))
        self.feats_bin_path = os.path.join(root, "{}-{}_feats.dat".format(name, feature_type))
        self.data_mutable = Data() 
        self.x = None
        self.y = None
        self.edge_index = None
        self.train_mask = None
        self.val_mask = None
        self.test_mask = None
        super(Airport, self).__init__(root)
        #if load_data:
        #    self.data, self.slices = torch.load(self.processed_paths[0])
        #else:
        #    self.process()  
        self.read_airport_data()
        self.set_label_split(0.6, 0.2)
        self.update_data()
           
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
        #self.read_airport_data()
        #self.set_label_split()
        #self.update_data()
        #torch.save((self.data, self.slices), self.processed_paths[0])
        pass

    def set_label_split(self, train_ratio=0.6, validation_ratio=0.2): 
        """ 
        Sets training, validation, and test masks over data based on
        specified train and validation ratios (with respect to 
        original data). Test ratio is inferred as 1 - train_ratio - 
        validation_ratio. 

        Does NOT check for invalid ratio combinations (e.g. sum > 1)
        """
        
        y = self.y
        print("y: {}".format(y.size(0)))
        if train_ratio < 1:
            train_index, val_index, test_index = train_validation_test_split(y, train_ratio, validation_ratio)
        else:
            train_index = np.arange(len(y))
            val_index = np.empty(0)
            test_index = np.empty(0)
        self.train_mask = sample_mask(train_index, num_nodes=y.size(0))
        self.val_mask = sample_mask(val_index, num_nodes=y.size(0))
        self.test_mask = sample_mask(test_index, num_nodes=y.size(0))


    def read_airport_data(self):
        """
        Reads airport data binaries (edges, labels, features)
            and stores as Tensors.
        """
        coo = readBinary(self.graph_bin_path)
        graph_tensor = torch.Tensor(coo).t()
        self.edge_index = graph_tensor.long() 
        print("edge_index: {}".format(self.edge_index.size()))

        # create features tensor
        feats_data = np.array(readBinary(self.feats_bin_path))
        self.x = torch.Tensor(feats_data) 
        print("feats_data: {}".format(self.x.size()))

        # process labels 
        labels_data = np.array(readBinary(self.labels_bin_path), dtype=np.uint8)
        self.y = torch.Tensor(labels_data).long()


    def update_data(self):
        """ 
        Updates data object using latest value assignments
        """
        self.data_mutable.x = self.x
        self.data_mutable.y = self.y
        self.data_mutable.edge_index = self.edge_index
        self.data_mutable.train_mask = self.train_mask
        self.data_mutable.val_mask = self.val_mask
        self.data_mutable.test_mask = self.test_mask

        self.data, self.slices = self.collate([self.data_mutable])


    def __repr__(self):
        return '{}()'.format(self.name)

# TODO: this should be called by user, instead of Airports class 
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


def train_validation_test_split(y, train_ratio, valid_ratio):
    """
    Return indices corresponding to stratified train, validation, and
    test splits.
    Note: no random seed is set, so splits may differ on different calls
        to this function for the same data.
    """
    
    y = y.numpy()
    # print total class counts
    #unique, counts = np.unique(y, return_counts=True)
    #class_counts = dict(zip(unique, counts))
    #print("total class counts: {}".format(class_counts))

    rest_ratio = 1-train_ratio
    indices = np.arange(len(y))
    y_train, y_rest, ind_train, ind_rest = \
            train_test_split(y, indices, test_size=rest_ratio, stratify=y)

    test_ratio = 1-(valid_ratio/rest_ratio) 
    y_valid, y_test, ind_valid, ind_test = \
            train_test_split(y_rest, ind_rest, test_size=test_ratio, stratify=y_rest)

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

    print("({},{},{})".format(len(ind_train), len(ind_valid), len(ind_test)))
    return ind_train, ind_valid, ind_test


def sample_mask(index, num_nodes):
    """
    Converts list of indices into boolean mask (over num_nodes)
    """
    mask = torch.zeros((num_nodes, ), dtype=torch.uint8)
    mask[index] = 1
    return mask
