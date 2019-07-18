import os.path
import sys
sys.path = [os.path.expanduser("~/pytorch_geometric")] + sys.path
import argparse
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.datasets import Airport 
from torch_geometric.nn import GINConv, global_add_pool
import numpy as np

dataset = None
data = None


class MLP2(torch.nn.Module):
    """ Two layer MLP """
    def __init__(self, hidden_dim):
        super(MLP2, self).__init__()
        dim_h = hidden_dim 
        dim_in = int(dataset.num_features)
        dim_out = int(dataset.num_classes)
        self.linear1 = Linear(dim_in, dim_h)    
        self.linear2 = Linear(dim_h, dim_out)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self):
        x = self.dropout(F.relu(self.linear1(data.x)))
        x = self.linear2(x)
        return F.log_softmax(x, dim=-1)


class MLP3(torch.nn.Module):
    """ Three layer MLP """
    def __init__(self, hidden_dim):
        super(MLP3, self).__init__()
        dim_h = hidden_dim 
        dim_in = int(dataset.num_features)
        dim_out = int(dataset.num_classes)
        self.linear1 = Linear(dim_in, dim_h)    
        self.linear2 = Linear(dim_h, dim_h)
        self.linear3 = Linear(dim_h, dim_out)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self):
        x = self.dropout(F.relu(self.linear1(data.x)))
        x = self.dropout(F.relu(self.linear2(x)))
        x = self.linear3(x)
        return F.log_softmax(x, dim=-1)


class GIN(torch.nn.Module):
    """ 
    Adapted from ../examples/mutag_gin.py 
    """

    def __init__(self, hidden_dim):
        super(GIN, self).__init__()

        num_features = dataset.num_features
        dim = hidden_dim

        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
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
        self.fc2 = Linear(dim, dataset.num_classes)

    def forward(self):
        edge_index = data.edge_index
        x = data.x
        x = F.relu(self.conv1(x, edge_index))
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


def train(model):
    model.train() # sets mode
    optimizer.zero_grad()
    # model() implicitly calls forward()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


def test(model):
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

TRIALS = 10
LEARNING_RATE = 0.01

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs baseline experiments')

    parser.add_argument('data_dir', help='Data directory')
    parser.add_argument('data_name', help='Dataset name')
    parser.add_argument('feature_type', default="LDP", help='Type of features to use')
    parser.add_argument('--model', default="mlp2", help='Model type')
    parser.add_argument('--hidden_dim', default=32, type=int, help='Dimension of hidden layer(s)')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs (full passes through dataset)')
    parser.add_argument('--verbose', default=False, action='store_true', help='Print additional training information')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model == "mlp2":
        model_type = MLP2
    elif args.model == "mlp3":
        model_type = MLP3
    elif args.model == "gin":
        model_type = GIN
    # Testing over multiple test sets 
    trial_test_acc = [] 

    for tr in range(TRIALS):
        dataset = Airport(args.data_dir, args.data_name, args.feature_type) 
        data = dataset[0]

        model, data = model_type(args.hidden_dim).to(device), data.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
        best_val_acc = test_acc = 0
        for epoch in range(args.epochs):
            train(model)
            train_acc, val_acc, tmp_test_acc = test(model)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
                if args.verbose:
                    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
                    print(log.format(epoch+1, train_acc, best_val_acc, test_acc))
        if args.verbose:
            print("-----------------------------------------------------")
        trial_test_acc.append(test_acc)

    # Report average test accuracy, standard deviation
    test_avg = np.mean(trial_test_acc)
    test_std = np.std(trial_test_acc)  
    log = 'Trials: {}, Test average: {:.4f}, Test std: {:.4f}'
    print(log.format(TRIALS, test_avg, test_std))
