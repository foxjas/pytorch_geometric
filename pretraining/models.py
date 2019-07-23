import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool

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
    Adapted from ../examples/mutag_gin.py 
    """

    def __init__(self, x, edge_index, dim_in, dim_h, dim_out):
        super(GIN, self).__init__()
        self.x = x
        self.edge_index = edge_index
        dim = dim_h 

        nn1 = Sequential(Linear(dim_in, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        """
        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)
        """
        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, dim_out) 

    def forward(self):
        edge_index = self.edge_index
        x = F.relu(self.conv1(self.x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        """
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = F.relu(self.fc1(x))
        """
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


def train(model, data, optimizer):
    model.train() # sets mode
    optimizer.zero_grad()
    # model() implicitly calls forward()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


def test(model, data):
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs
