import os.path
import sys
sys.path = [os.path.expanduser("~/pytorch_geometric")] + sys.path
import argparse
import torch
from torch_geometric.datasets import Airport 
from baseline import train, test, MLP2, MLP3, GIN
import numpy as np
from pprint import pprint

TRIALS = 1
LEARNING_RATE = 0.01

def initParamsFromModel(model_source, model_target):
    """
    Initializes parameters of new model with weights of saved model,
        except linear layer corresponding to output
    """
    target_dict = model_target.state_dict()
    #pprint("before: {}".format(target_dict))
    source_dict = model_source.state_dict()
    source_dict = {k: v for k,v in source_dict.items() if 'out' not in k}  
    #source_dict = {k: v for k,v in source_dict.items()}  
    target_dict.update(source_dict)
    model_target.load_state_dict(target_dict)
    #pprint("after: {}".format(model_target.state_dict()))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

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

    # TODO: data loading is hacky; should load all data once, except
    #   for the train/validation/test masks which change per trial
    for tr in range(TRIALS):
        dataset = Airport(args.data_dir, args.data_name, args.feature_type) 
        data = dataset[0].to(device)
        if model_type is GIN: 
            model = model_type(data.x, data.edge_index, dataset.num_features, args.hidden_dim, dataset.num_classes)
        else:
            model = model_type(data.x, dataset.num_features, args.hidden_dim, dataset.num_classes)
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
        best_val_acc = test_acc = 0
        for epoch in range(args.epochs):
            train(model, data, optimizer)
            train_acc, val_acc, tmp_test_acc = test(model, data)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
                if args.verbose:
                    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
                    print(log.format(epoch+1, train_acc, best_val_acc, test_acc))
        if args.verbose:
            print("-----------------------------------------------------")
        trial_test_acc.append(test_acc)
    """
    # Report average test accuracy, standard deviation
    test_avg = np.mean(trial_test_acc)
    test_std = np.std(trial_test_acc)  
    log = 'Trials: {}, Test average: {:.4f}, Test std: {:.4f}'
    print(log.format(TRIALS, test_avg, test_std))
    """
    
    for tr in range(TRIALS):
        dataset = Airport(args.data_dir, args.data_name, args.feature_type) 
        data = dataset[0].to(device)

        if model_type is GIN: 
            model_prt = model_type(data.x, data.edge_index, dataset.num_features, args.hidden_dim, dataset.num_classes)
        else:
            model_prt = model_type(data.x, dataset.num_features, args.hidden_dim, dataset.num_classes)
        model_prt = model_prt.to(device)
        initParamsFromModel(model, model_prt)

        optimizer = torch.optim.Adam(model_prt.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
        best_val_acc = test_acc = 0
        for epoch in range(args.epochs):
            train(model_prt, data, optimizer)
            train_acc, val_acc, tmp_test_acc = test(model_prt, data)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
                if args.verbose:
                    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
                    print(log.format(epoch+1, train_acc, best_val_acc, test_acc))
        if args.verbose:
            print("-----------------------------------------------------")
