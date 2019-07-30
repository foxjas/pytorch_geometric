import os.path
import sys
sys.path = [os.path.expanduser("~/pytorch_geometric")] + sys.path
import argparse
import torch
from pretraining.airports import Airport
from pretraining.models import MLP2,MLP3,GIN,train_step,test_step 
import numpy as np

def enumerate_masks_binary(args, dataset, feats_readonly, bm, pos=0):
    """
    Enumerates over all possible binary masks, recursively.
    """
    if pos >= len(bm):
        evaluate_accuracy(args, dataset, feats_readonly, mask)
    else:
        bm[pos] = 0
        enumerate_masks_binary(args, dataset, feats_readonly, bm, pos+1)
        bm[pos] = 1
        enumerate_masks_binary(args, dataset, feats_readonly, bm, pos+1)


def mask_features(bm, features):
    """
    Applies mask corresponding to columns to features.
    Returns copy of features. 
    - features: Torch tensor
    """
    features_masked = features.clone()
    for i, b in enumerate(bm):
        if not b:
            features_masked[:,i] = 0
    return features_masked


def evaluate_accuracy(args, dataset, features, mask):

    trial_test_acc = [] 
    if args.model == "mlp2":
        model_type = MLP2
    elif args.model == "mlp3":
        model_type = MLP3
    elif args.model == "gin":
        model_type = GIN

    for tr in range(args.trials):
        dataset.set_label_split(args.train_ratio, 0.2, 0.2)
        feats_masked = mask_features(mask, features)
        dataset.set_features(feats_masked)
        dataset.update_data()
        data = dataset[0].to(args.device)
        if model_type is GIN: 
            model = model_type(data.x, data.edge_index, dataset.num_features, args.hidden_dim, dataset.num_classes)
        else:
            model = model_type(data.x, dataset.num_features, args.hidden_dim, dataset.num_classes)
        model = model.to(args.device)

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
    log = 'Mask: {}, Test average: {:.4f}, Test std: {:.4f}'
    mask = map(int, mask)
    mask_str = [str(x) for x in mask]
    print(log.format(''.join(mask_str), test_avg, test_std))


LEARNING_RATE = 0.01


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs baseline experiments')
    parser.add_argument('data_dir', help='Data directory')
    parser.add_argument('data_name', help='Dataset name')
    parser.add_argument('feature_type', default="LDP", help='Type of features to use')
    parser.add_argument('--model', default="mlp2", help='Model type')
    parser.add_argument('--hidden_dim', default=32, type=int, help='Dimension of hidden layer(s)')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs (full passes through dataset)')
    parser.add_argument('--train_ratio', default=0.6, type=float, help='Training data ratio')
    parser.add_argument('--trials', default=10, type=int, help='Number of trials to average over, for testing')
    parser.add_argument('--verbose', default=False, action='store_true', help='Print additional training information')
    parser.add_argument('--device', default=0, type=int, help='GPU device ID')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.device)

    dataset = Airport(args.data_dir, args.data_name, args.feature_type, load_data=False)
    features_orig = dataset.x
    mask = np.zeros(dataset.x.shape[1])
    enumerate_masks_binary(args, dataset, features_orig, mask)


