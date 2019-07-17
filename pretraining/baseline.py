import os.path
import sys
sys.path = [os.path.expanduser("~/pytorch_geometric")] + sys.path
import argparse
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.datasets import Airport 
from torch_geometric.nn import GINConv, global_add_pool

dataset = None
data = None
model = None

class MLP(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(MLP, self).__init__()
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


def train():
    model.train() # sets mode
    optimizer.zero_grad()
    # model() implicitly calls forward()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs baseline experiments')

    parser.add_argument('data_dir', help='Data directory')
    parser.add_argument('data_name', help='Dataset name')
    parser.add_argument('feature_type', default="LDP", help='Type of features to use')
    parser.add_argument('--hidden_dim', default=32, type=int, help='Dimension of hidden layer')

    args = parser.parse_args()
    if not args.data_dir or not args.data_name:
        sys.exit()

    dataset = Airport(args.data_dir, args.data_name, args.feature_type) 
    data = dataset[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = MLP(args.hidden_dim).to(device), data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_val_acc = test_acc = 0
    for epoch in range(1, 201):
        train()
        train_acc, val_acc, tmp_test_acc = test()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, best_val_acc, test_acc))
