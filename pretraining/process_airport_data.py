import os.path
import sys
sys.path = [os.path.expanduser("~/pytorch_geometric")] + sys.path
from gen_features import *
from graph import *
from utils import *

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs baseline experiments')

    parser.add_argument('data_dir', help='Data directory')
    parser.add_argument('data_name', help='Dataset name')
    parser.add_argument('feature_type', default="LDP", help='Type of features to use')
    args = parser.parse_args()

    prepare_airport_data(args.data_dir, args.data_name, args.feature_type)
