import os.path
import sys
sys.path = [os.path.expanduser("~/pytorch_geometric")] + sys.path
import argparse
import torch
from airports import Airport 
from models import train_step, test_step, MLP2, MLP3, GIN
import numpy as np
from pprint import pprint

LEARNING_RATE_PRE = 0.01
LEARNING_RATE = 0.01

def initParamsFromModel(model_source, model_target):
    """
    Initializes parameters of new model with weights of saved model,
        except linear layer corresponding to output
    """
    target_dict = model_target.state_dict()
    #pprint("before: {}".format(target_dict))
    source_dict = model_source.state_dict()
    source_dict = {k: v for k,v in source_dict.items() if '_out' not in k}  
    #source_dict = {k: v for k,v in source_dict.items()}  
    target_dict.update(source_dict)
    model_target.load_state_dict(target_dict)
    #pprint("after: {}".format(model_target.state_dict()))

def pretrain(model, data, epochs=200, verbose=False):
    """
    Stopping criteria?
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_PRE, weight_decay=5e-4)
    for epoch in range(epochs):
        train_step(model, data, optimizer)
        train_acc = train_accuracy(model, data)
        log = 'Epoch: {:03d}, Train: {:.4f}'
        if verbose:
            print(log.format(epoch+1, train_acc))
    if verbose:
        print("-----------------------------------------------------")

def train_accuracy(model, data):
    """
    Computes training accuracy
    """
    model.eval()
    logits, accs = model(), []
    mask = data.train_mask
    pred = logits[mask].max(1)[1]
    acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('data_dir', help='Data directory')
    parser.add_argument('data_name_pre', help='Pretraining dataset name')
    parser.add_argument('data_name', help='Dataset name')
    parser.add_argument('feature_type', default="LDP", help='Type of features to use')
    parser.add_argument('--model', default="mlp2", help='Model type')
    parser.add_argument('--hidden_dim', default=32, type=int, help='Dimension of hidden layer(s)')
    parser.add_argument('--train_ratio', default=0.6, type=float, help='Training data ratio')
    parser.add_argument('--epochs_pre', default=200, type=int, help='Number of epochs (full passes through dataset)')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs (full passes through dataset)')
    parser.add_argument('--device', default=0, type=int, help='GPU device ID')
    parser.add_argument('--trials', default=10, type=int, help='Number of trials to average over, for testing')
    parser.add_argument('--verbose', default=False, action='store_true', help='Print additional training information')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.device)
    if args.model == "mlp2":
        model_type = MLP2
    elif args.model == "mlp3":
        model_type = MLP3
    elif args.model == "gin":
        model_type = GIN

    dataset_pre = Airport(args.data_dir, args.data_name_pre, args.feature_type) 
    dataset_pre.set_label_split(1.0, 0, 0)
    dataset_pre.update_data()
    data_pre = dataset_pre[0].to(device)
    if model_type is GIN: 
        model_pre = model_type(data_pre.x, data_pre.edge_index, dataset_pre.num_features, args.hidden_dim, dataset_pre.num_classes)
    else:
        model_pre = model_type(data_pre.x, dataset_pre.num_features, args.hidden_dim, dataset_pre.num_classes)
    model_pre = model_pre.to(device)
    pretrain(model_pre, data_pre, args.epochs_pre, args.verbose)

    
    """ Fine-tuning """
    dataset = Airport(args.data_dir, args.data_name, args.feature_type) 
    trial_test_acc = [] 
    for tr in range(args.trials):
        dataset.set_label_split(args.train_ratio, 0.2, 0.2)
        dataset.update_data()
        data = dataset[0].to(device)

        if model_type is GIN: 
            model = model_type(data.x, data.edge_index, dataset.num_features, args.hidden_dim, dataset.num_classes)
        else:
            model = model_type(data.x, dataset.num_features, args.hidden_dim, dataset.num_classes)
        model = model.to(device)
        initParamsFromModel(model_pre, model)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
        best_val_acc = test_acc = 0
        for epoch in range(args.epochs):
            train_step(model, data, optimizer)
            train_acc, val_acc, tmp_test_acc = test_step(model, data)
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

    log = 'Pretrain Data: {}, Data: {}, Train ratio: {:4f}, Epochs: {}, Average: {:.4f}, Std: {:.4f}'
    print(log.format(args.data_name_pre, args.data_name, args.train_ratio, args.epochs, test_avg, test_std))
