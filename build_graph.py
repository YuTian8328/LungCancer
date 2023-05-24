import numpy as np
import torch
import networkx as nx
import pandas as pd
from utils import CHANNEL_MARKERS, Spot2Histology


def build_graph_from_raw(df, max_dist=70):
    """ Build a nx graph from raw csv file.
    Generate graph edges based on cells' euclidian distance. 
    A max distance is set as a threshold 
    to decide wether or not two nodes are connected."""
    G = nx.Graph()
    node_features = {}
    edge_features = {}
    for i in range(df.shape[0]):
        G.add_node(i)
    coordinates = df.loc[:, ['Location X', 'Location Y']].values
    markers = df[CHANNEL_MARKERS].values
    # sizes = df[['CELL AREA', 'CELL VOLUMETRY']].values

    for i in range(G.number_of_nodes()):
        node_features[i] = {'histology': [Spot2Histology[df.spot.values[0]]],
                            'coord': coordinates[i],
                            'marker': markers[i]}
    nx.set_node_attributes(G, node_features)

    for i in range(G.number_of_nodes()):  # iterate nodes
        neighbors = []  # list of neighbor index
        dists = []  # list of distance between cells
        for j in range(G.number_of_nodes()):
            # euclidian distance
            dist = np.linalg.norm(coordinates[i, :]-coordinates[j, :])
    #         print(dist)
            if dist < max_dist:  # critical distance
                dists.append(dist)
                neighbors.append(j)
            else:
                continue

        for k, n in enumerate(neighbors):
            if n > i:
                G.add_edge(i, n)
                edge_features[(i, n)] = {
                    "distance": dists[k],
                    "edge_type": 0 if dists[k] < 45 else 1
                }
    nx.set_edge_attributes(G, edge_features)
    return G


def get_edges_max_neighbors(coordinates, threshold=5):
    ''' generate graph edges based on cells' euclidian distance. 
    By default, 5 closest neighbors are considered as connected.
    Self loop is inserted'''
    num_nodes = coordinates.shape[0]
    edge_index = []  # list of edge indice

    for i in range(num_nodes):  # iterate nodes
        neighbors = []  # list of neighbor index
        dists = []  # list of distance between cells
        for j in range(num_nodes):
            # euclidian distance
            dist = np.linalg.norm(coordinates[i, :]-coordinates[j, :])
            # number of neighbors should be less than the threshold + 1(self loop)
            if len(dists) < threshold+1:
                dists.append(dist)
                neighbors.append(j)
            else:
                # only a certain number of closest cells are considered as connected
                if dist >= np.max(dists):
                    continue
                else:
                    idx = np.argmax(dists)
                    dists[idx] = dist
                    neighbors[idx] = j
        for n in neighbors:
            if [i, n] not in edge_index:
                edge_index.append([i, n])
                edge_index.append([n, i])

    edge_index = torch.tensor(
        edge_index, dtype=torch.long)  # resulted edge index
    # reshape the edge_index tensor to match GAE models
    edge_index = edge_index.t().contiguous()
    return edge_index


def get_edges_threshold_distance(coordinates, max_dist=100):
    ''' generate graph edges based on cells' euclidian distance, a critical distance is set as a threshold 
    to decide wether or not two nodes are connected. Self loop inserted'''

    num_nodes = coordinates.shape[0]
    edge_index = []  # list of edge indice

    for i in range(num_nodes):  # iterate nodes
        neighbors = []  # list of neighbor index
        dists = []  # list of distance between cells
        for j in range(num_nodes):
            # euclidian distance
            dist = np.linalg.norm(coordinates[i, :]-coordinates[j, :])
            if dist < max_dist:  # critical distance
                dists.append(dist)
                neighbors.append(j)
            else:
                continue

        for n in neighbors:
            if [i, n] not in edge_index:
                edge_index.append([i, n])
                edge_index.append([n, i])

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    # reshape the edge_index tensor to match GAE models
    edge_index = edge_index.t().contiguous()
    return edge_index
