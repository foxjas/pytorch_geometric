import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, ModuleDict
from torch_geometric.nn import GINConv, global_add_pool
from collections import OrderedDict


class MLP2(torch.nn.Module):
    """ Two layer MLP """
    def __init__(self, x, dim_in, dim_h, dim_out):
        super(MLP2, self).__init__()
        self.x = x
        self.linear1 = Linear(dim_in, dim_h)    
        self.linear_out = Linear(dim_h, dim_out)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self):
        x = self.dropout(F.relu(self.linear1(self.x)))
        x = self.linear_out(x)
        return F.log_softmax(x, dim=-1)


class MLP3(torch.nn.Module):
    """ Three layer MLP """
    def __init__(self, x, dim_in, dim_h, dim_out):
        super(MLP3, self).__init__()
        self.x = x
        self.linear1 = Linear(dim_in, dim_h)    
        self.linear2 = Linear(dim_h, dim_h)
        self.linear_out = Linear(dim_h, dim_out)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self):
        x = self.dropout(F.relu(self.linear1(self.x)))
        x = self.dropout(F.relu(self.linear2(x)))
        x = self.linear_out(x)
        return F.log_softmax(x, dim=-1)


class GIN(torch.nn.Module):
    """ 
    Graph Isomorphism Network, adapted from ../examples/mutag_gin.py 
    Supports variable number of convolutional layers
    """

    def __init__(self, x, edge_index, dim_in, dim_h, dim_out, layers_conv=3):
        super(GIN, self).__init__()
        self.x = x
        self.edge_index = edge_index
        self.layers_conv = layers_conv
        dim = dim_h 

        self.model = ModuleDict()
        for l in range(1, self.layers_conv+1):
            if l == 1:
                nn = Sequential(Linear(dim_in, dim), ReLU(), Linear(dim, dim))
            else:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            conv = GINConv(nn)
            self.model["conv{}".format(l)] = conv
            bn = torch.nn.BatchNorm1d(dim)
            self.model["bn{}".format(l)] = bn
        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, dim_out) 

    def forward(self):
        edge_index = self.edge_index
        x = self.x
        for l in range(1, self.layers_conv+1):
            x = F.relu(self.model["conv{}".format(l)](x, edge_index))
            x = self.model["bn{}".format(l)](x)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


def train_step(model, data, optimizer):
    model.train() # sets mode
    optimizer.zero_grad()
    # model() implicitly calls forward()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


def test_step(model, data):
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs
