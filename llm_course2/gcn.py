import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_dense_adj, to_networkx

# Import dataset from PyTorch Geometric
dataset = KarateClub()

# Print information
print(dataset)
print('------------')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

# Print first element
print(f'Graph: {dataset[0]}')

data = dataset[0]

# print(f'x = {data.x.shape}')
# print(data.x)

# print(f'edge_index = {data.edge_index.shape}')
# print(data.edge_index)

# A = to_dense_adj(data.edge_index)[0].numpy().astype(int)
# print(f'A = {A.shape}')
# print(A)

# print(f'y = {data.y.shape}')
# print(data.y)

# print(f'train_mask = {data.train_mask.shape}')
# print(data.train_mask)

# print(f'Edges are directed: {data.is_directed()}')
# print(f'Graph has isolated nodes: {data.has_isolated_nodes()}')
# print(f'Graph has loops: {data.has_self_loops()}')

G = to_networkx(data, to_undirected=True)
plt.figure(figsize=(12,12))
plt.axis('off')
nx.draw_networkx(G,
                pos=nx.spring_layout(G, seed=0),
                with_labels=True,
                node_size=800,
                node_color=data.y.numpy(), 
                cmap="hsv",
                vmin=-2,
                vmax=3,
                width=0.8,
                edge_color="grey",
                font_size=14
                )
plt.show()