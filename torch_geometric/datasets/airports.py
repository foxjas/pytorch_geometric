import os.path
import torch
from torch_geometric.data import InMemoryDataset
from pretraining.gen_features import *
from utils import readBinary, saveBinary
from graph import readEdgelist, graphToCOO 

class Airport(InMemoryDataset):
    r"""The airport network datasets "europe", "brazil" and "usa" from the
    `"Revisiting Semi-Supervised Learning with Graph Embeddings"
    <https://arxiv.org/abs/1603.08861>`_ paper.
    Nodes represent documents and edges represent citation links.
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
    Processes airport data 
    - Node features: should be saved; could take while to process
    - Graph index: read raw graph and process
    - Train/val/test masks: generate on the fly?
        - Need to read in labels
    """
    graph_bin_path = os.path.join(folder, "{}_edges.dat".format(data_name))
    if not os.path.isfile(graph_bin_path):
        graph_path = os.path.join(folder, "{}.edgelist".format(data_name))
        coo = graphToCOO(readEdgelist(graph_path)) # TODO: replace with single path argument
        saveBinary(coo, data_name, "edges", folder)
        
    graph_data = torch.Tensor(readBinary(graph_bin_path)).t()
    print("graph_data: {}".format(graph_data.size()))

    # add check to see if these exist; if not, call utility to generate
    feats_path = os.path.join(folder, "{}_feats.dat".format(data_name))
    feats_data = torch.Tensor(readBinary(feats_path))
    print("feats_data: {}".format(feats_data.size()))
   
    
    """
    labels_path = os.path.join(folder, "{}_edges.dat".format(data_name))
    labels_data = torch.Tensor(readBinary(graph_path))
    """

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

def readLabels(fPath):
    """

    """
    pass


def sample_mask(index, num_nodes):
    mask = torch.zeros((num_nodes, ), dtype=torch.uint8)
    mask[index] = 1
    return mask
