from torch_geometric.data import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset
import torch

import itertools


def generate_node_tuples(g, k: int):

    xs = itertools.combinations(g.x, k)

    xs = list(map(torch.stack, xs))

    X = torch.stack(xs)
    
    return X



if __name__ == "__main__":

    dataset = PygGraphPropPredDataset(name = "ogbg-molhiv", root = 'dataset/')
    
    for graph in dataset:
        X = generate_node_tuples(graph, 2)
        print(X.shape)
        exit()