import os.path
import sys
sys.path = [os.path.expanduser("~/pytorch_geometric")] + sys.path
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from pretraining.gen_features import *
from utils import * 
from graph import * 
from generator.shapes import *
from generator.build_graph import *
from gen_features import *
from sklearn.model_selection import train_test_split 
from collections import defaultdict
from pprint import pprint

class Structure(InMemoryDataset):
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
        super(Structure, self).__init__(root)

        self.read_data()
        #self.shuffle_labels()
        self.set_label_samples(20, 200, 1000)
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
        pass

    def set_label_samples(self, nTrainPerClass, nValidation, nTest): 
        """ 
        Sets number of samples per class
        """
        
        y = self.y
        #print("y: {}".format(y.size(0)))
        train_index, val_index, test_index = train_validation_test_split(y,
                    nTrainPerClass, nValidation, nTest)

        self.train_mask = sample_mask(train_index, num_nodes=y.size(0))
        self.val_mask = sample_mask(val_index, num_nodes=y.size(0))
        self.test_mask = sample_mask(test_index, num_nodes=y.size(0))

    def set_labels(self, labels):
        self.y = labels

    def shuffle_labels(self):
        """
        Randomly shuffles labels
        """
        y = self.y.numpy()
        np.random.shuffle(y)
        self.y = torch.Tensor(y).long()

    def set_features(self, features):
        """
        Set new features
        """
        self.x = features

    def read_data(self):
        """
        Reads data binaries (edges, labels, features)
            and stores as Tensors.
        """
        coo = readBinary(self.graph_bin_path)
        graph_tensor = torch.Tensor(coo).t()
        self.edge_index = graph_tensor.long() 
        #print("edge_index: {}".format(self.edge_index.size()))

        # create features tensor
        feats_data = np.array(readBinary(self.feats_bin_path))
        self.x = torch.Tensor(feats_data) 
        #print("feats_data: {}".format(self.x.size()))

        # process labels 
        labels_data = np.array(readBinary(self.labels_bin_path), dtype=np.uint8)
        self.y = torch.Tensor(labels_data).long()


    def update_data(self):
        """ 
        Updates data object using latest value assignments.
        Must be called after any setter, to finalize changes!
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


def train_validation_test_split(y, samplesPerClass, nValidation, nTest):
    """
    Return indices corresponding to stratified train, validation, and
    test splits.
    Note: no random seed is set, so splits may differ on different calls
        to this function for the same data.
    """
    
    y = y.numpy()
    label_indices = [[] for _ in range(len(y))]
    for i, label in enumerate(y):
        label_indices[label].append(i)
    
    ind_train = []
    ind_rest = []
    for label, indices in enumerate(label_indices):
        np.random.shuffle(indices)
        ind_train += indices[:samplesPerClass]
        ind_rest += indices[samplesPerClass:]

    np.random.shuffle(ind_rest)
    ind_valid = ind_rest[:nValidation]
    ind_test = ind_rest[nValidation:nValidation+nTest]
   
    print("({}, {}, {})".format(len(ind_train), len(ind_valid), len(ind_test)))
    duplicates = set(ind_train).intersection(set(ind_valid)).intersection(set(ind_test))
    assert not len(duplicates)

    return ind_train, ind_valid, ind_test


def sample_mask(index, num_nodes):
    """
    Converts list of indices into boolean mask (over num_nodes)
    """
    mask = torch.zeros((num_nodes, ), dtype=torch.uint8)
    mask[index] = 1
    return mask

name_shape_map = {
    "cycle": cycle,
    "house": house,
    "clique": clique
}

def prepare_data(folder, width_basis, nb_shapes, feature_type, basis_name, shape_name):
    """
    Generates graph, labels, and features vectors.
    Saves each to binary. 
    """

    # Generate structural graph
    basis_type = name_shape_map[basis_name]
    shape_type = name_shape_map[shape_name]
    list_shapes = [[shape_type]] * nb_shapes
    graph,_,_,role_id = build_structure(width_basis, basis_type, list_shapes, start=0,
                                rdm_basis_plugins=False, add_random_edges=0,
                                plot=False, savefig=False)
    print("Number of nodes in graph: {}".format(graph.number_of_nodes()))
    print("Number of edges in graph: {}".format(graph.number_of_edges()))
    
    # Read and relabel graph. Should be deterministic.
    graph, old_new_node_ids = relabelGraph(graph) 
    coo = graphToCOO(graph) 
    data_name = "{}_{}_{}_{}".format(basis_name, width_basis, shape_name, nb_shapes)
    saveBinary(coo, data_name, "edges", folder) 

    # Generate features. Depends only on graph
    if feature_type == "LDP":
       feats_data = ldp_features(graph)
    elif feature_type == "degree":
       feats_data = degreeOnlyFeatures(graph) 
    base_name = data_name + "-{}".format(feature_type)
    saveBinary(feats_data, base_name, "feats", folder)

    # Match labels to graph IDs. Depends only on graph
    labels_path = os.path.join(folder, "labels-{}.txt".format(data_name))
    node_labels = list(enumerate(role_id))
    node_labels = reorderLabels(old_new_node_ids, node_labels)
    node_labels = compressLabels(node_labels)
    #pprint(labelCounts(node_labels))
    saveBinary(node_labels, data_name, "labels", folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepares synthetic structure graph data')

    parser.add_argument('data_dir', help='Data directory')
    parser.add_argument('basis_width', type=int, help='Width of base structure (# nodes)')
    parser.add_argument('num_shapes', type=int, help='Number of shapes to attach to base')
    parser.add_argument('--feature_type', default="degree", help='Type of features to use')
    parser.add_argument('--basis_name', default='cycle', help='Name of base structure')
    parser.add_argument('--shape_name', default='house', help='Name of shapes to attach to base')
    args = parser.parse_args()

    prepare_data(args.data_dir, args.basis_width, args.num_shapes, args.feature_type, args.basis_name, args.shape_name)


