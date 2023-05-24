"""
Adapted from https://gitlab.com/enable-medicine-public/space-gm/-/blob/main/data.py
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import multiprocessing
from scipy.stats import rankdata
import matplotlib.pyplot as plt

import torch_geometric as tg
import torch
from torch_geometric.data import Dataset
# from torch_geometric.utils import subgraph
from torch.utils.data import RandomSampler
from torch_geometric.loader import DataLoader
from utils import CHANNEL_MARKERS


def process_feature(G, key, node_ind=None, edge_ind=None, **kwargs):
    """ wrapper fn for generating node/edge features """
    if key in ["histology", "coord", "marker"]:
        print(key)
        v = list(G.nodes[node_ind][key])

        return v
    elif key in ["distance", "edge_type"]:
        v = G.edges[edge_ind][key]
        return [v]
    else:
        print(key)
        raise ValueError("Feature not recognized")


def get_feature_names(features, channel_markers=CHANNEL_MARKERS):
    """ helper fn for getting feature names """
    feat_names = []
    for feat in features:
        if feat in ["distance", "edge_type"]:
            feat_names.append(feat)
        elif feat == "coord":
            feat_names.extend(["coord-x", "coord-y"])
        elif feat == "histology":
            feat_names.extend(["histology"])
        elif feat == "marker":
            feat_names.extend(["marker-%s" %
                               exp for exp in channel_markers])
        else:
            raise ValueError("Feature not recognized")
    return feat_names


def nx_to_tg_graph(G,
                   node_features=["histology",
                                  "marker",
                                  "coord"],
                   edge_features=["edge_type",
                                  "distance"],
                   **kwargs):
    """ Build tensorgraphs from nx graph """

    data = {"x": [], "y": [], "edge_attr": [], "edge_index": []}

    for node_ind in G.nodes:
        feat_val = []
        for key in node_features:
            # print(key)
            feat_val.extend(process_feature(G, key, node_ind=node_ind))

        data["x"].append(feat_val)
        data["y"].append(feat_val[0])

    for edge_ind in G.edges:
        feat_val = []
        for key in edge_features:
            feat_val.extend(process_feature(
                G, key, edge_ind=edge_ind, **kwargs))
        data["edge_attr"].append(feat_val)
        data["edge_index"].append(edge_ind)
        data["edge_attr"].append(feat_val)
        data["edge_index"].append(tuple(reversed(edge_ind)))

    for key, value in data.items():
        data[key] = torch.tensor(value)
    data['edge_index'] = data['edge_index'].t().long()
    data = tg.data.Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()
    return data


def k_hop_subgraph(node_idx,
                   num_hops,
                   edge_index,
                   edge_type_mask=None,
                   relabel_nodes=False,
                   num_nodes=None,
                   flow='source_to_target'):
    """ A customized k-hop subgraph fn that could filter for edge_type """

    num_nodes = edge_index.max().item() + 1 if num_nodes is None else num_nodes

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)
    edge_type_mask = torch.ones_like(
        edge_mask) if edge_type_mask is None else edge_type_mask

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]
    next_root = node_idx

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[next_root] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])
        # use nodes connected with mask=True to span
        next_root = col[edge_mask * edge_type_mask]

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask


class GNNDataset(Dataset):
    """ Main dataset structure for model training / inference """

    def __init__(self,
                 root,
                 transform=[],
                 pre_transform=None,
                 graph_folder_name="graphs_from_raw",
                 processed_folder_name='tensor_graphs',
                 subsample_neighbor_size=0,
                 node_features=["histology", "marker", "coord"],
                 edge_features=["edge_type", "distance"],
                 channel_markers=CHANNEL_MARKERS,
                 subgraph_source=None,  # 'save', 'chunk_save', 'on-the-fly'
                 subgraph_allow_distant_edge=True,
                 subgraph_size_limit=0,
                 sampling_avoid_unassigned=True,
                 **kwargs):
        self.root = root
        self.graph_folder_name = graph_folder_name
        self.processed_folder_name = processed_folder_name
        # os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        # If to subsample local graphs, 0 = no subsampling
        self.subsample_neighbor_size = subsample_neighbor_size

        # Feature names
        self.node_features = node_features
        self.edge_features = edge_features
        self.channel_markers = channel_markers
        self.node_feature_names = get_feature_names(
            node_features, channel_markers=self.channel_markers)
        self.edge_feature_names = get_feature_names(
            edge_features, channel_markers=self.channel_markers)

        self.process_kwargs = kwargs
        # self.process_kwargs['cell_types'] = self.cell_types
        self.process_kwargs['channel_markers'] = self.channel_markers

        super(GNNDataset, self).__init__(root, None, pre_transform)
        # self.transform = transform
        self.subgraph_source = subgraph_source
        # self.subgraph_allow_distant_edge = subgraph_allow_distant_edge
        # self.subgraph_size_limit = subgraph_size_limit

        # self.N = len(self.processed_paths)
        # self.sampling_freq = {self.cell_types[ct]: 1./self.cell_type_freq[ct] for ct in self.cell_types}
        # self.sampling_freq = torch.from_numpy(np.array([self.sampling_freq[i] for i in range(len(self.sampling_freq))]))
        # # Avoid sampling unassigned cell
        # if sampling_avoid_unassigned:
        #     self.sampling_freq[self.cell_types['Unassigned']] = 0.
        self.cached_data = {}

    def set_indices(self, inds):
        """ Limit sampling to `inds` """
        self._indices = inds
        return

    def set_subgraph_source(self, subgraph_source):
        assert subgraph_source in ['save', 'chunk_save', 'on-the-fly']
        self.subgraph_source = subgraph_source

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.raw_folder_name)

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.processed_folder_name)

    @property
    def graph_dir(self) -> str:
        return os.path.join(self.root, self.graph_folder_name)

    @property
    def graph_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        downloading."""
        paths = []
        for file_name in os.listdir(self.graph_dir):
            paths.append(Path(self.root, self.graph_folder_name, file_name))
        return paths

    @property
    def raw_file_names(self):
        return sorted([f for f in os.listdir(self.raw_dir) if f.endswith('.gpkl')])

    @property
    def processed_file_names(self):
        return sorted([f for f in os.listdir(self.processed_dir)])

    def len(self):
        return len(self.processed_paths)

    def process(self):
        for graph_path in self.graph_paths:
            # aq_id = os.path.splitext(os.path.split(raw_path)[-1])[0]
            # if os.path.exists(os.path.join(self.processed_dir, '%s.0.gpt' % aq_id)):
            #     continue

            G = nx.read_gpickle(graph_path)
            data = nx_to_tg_graph(G,
                                  node_features=self.node_features,
                                  edge_features=self.edge_features,
                                  **self.process_kwargs)

            if not self.pre_transform is None:
                for transform_fn in self.pre_transform:
                    data = transform_fn(data)
            torch.save(data, os.path.join(self.processed_dir,
                                          '%s.pt' % os.path.split(graph_path)[1][:-5]))
        return

    def save_all_subgraphs_to_chunk(self):
        """ save individual n-hop subgraphs to file (one file per sample) """
        for idx, p in enumerate(self.processed_paths):
            data = self.get_full(idx)
            n_nodes = data.x.shape[0]
            neighbor_graph_path = p.replace(
                '.gpt', '.%d-hop.gpt' % self.subsample_neighbor_size)
            if os.path.exists(neighbor_graph_path):
                continue

            subgraphs = []
            for node_i in range(n_nodes):
                subgraphs.append(self.get_subgraph(idx, node_i))
            torch.save(subgraphs, neighbor_graph_path)
        return

    def save_all_subgraphs(self):
        """ (deprecated) save individual n-hop subgraph to file (one file per subgraph)  """
        for idx, p in enumerate(self.processed_paths):
            data = self.get_full(idx)
            n_nodes = data.x.shape[0]

            sub_graph_folder = os.path.join(os.path.split(
                p)[0], '%d-hop_neighborgraph' % self.subsample_neighbor_size)
            os.makedirs(sub_graph_folder, exist_ok=True)
            for node_i in range(n_nodes):
                neighbor_graph_path = os.path.join(
                    sub_graph_folder,
                    os.path.split(p)[1].replace('.gpt', '.%d-hop.%d.gpt' % (self.subsample_neighbor_size, node_i)))
                if not os.path.exists(neighbor_graph_path):
                    sub_g = self.get_subgraph(idx, node_i)
                    torch.save(sub_g, neighbor_graph_path)
        return

    def pick_center(self, data):
        """ Random sample center nodes, cell type balanced """
        cell_types = data["x"][:, 0].long()

        freq = self.sampling_freq.gather(0, cell_types)
        freq = freq / freq.sum()
        center_node_ind = np.random.choice(
            np.arange(len(freq)), p=freq.cpu().data.numpy())
        return center_node_ind

    def load_to_cache(self, idx, subgraphs=False):
        print(self.processed_paths)
        data = torch.load(self.processed_paths[idx])
        self.cached_data[idx] = data

        if subgraphs or self.subgraph_source == 'chunk_save':
            neighbor_graph_path = self.processed_paths[idx].replace(
                '.gpt', '.%d-hop.gpt' % self.subsample_neighbor_size)
            neighbor_graphs = torch.load(neighbor_graph_path)
            for j, ng in enumerate(neighbor_graphs):
                self.cached_data[(idx, j)] = ng

    def clear_cache(self):
        del self.cached_data
        self.cached_data = {}
        return

    def get_full(self, idx):
        """ Read entire sample """
        if idx in self.cached_data:
            return self.cached_data[idx]
        else:
            data = torch.load(self.processed_paths[idx])
            self.cached_data[idx] = data
            return data

    def get(self, idx):
        """ Read an n-hop subgraph from sample """
        data = self.get_full(idx)
        if self.subsample_neighbor_size == 0:
            return data
        else:
            center_ind = self.pick_center(data)
            if (idx, center_ind) in self.cached_data:
                return self.cached_data[(idx, center_ind)]

            if self.subgraph_source == 'on-the-fly':
                return self.get_subgraph(idx, center_ind)
            elif self.subgraph_source == 'save':
                return self.get_saved_subgraph(idx, center_ind)
            elif self.subgraph_source == 'chunk_save':
                return self.get_saved_subgraph_from_chunk(idx, center_ind)

    def get_saved_subgraph_from_chunk(self, idx, center_ind):
        """ Read subgraph from chunk file, use after calling `save_all_subgraphs_to_chunk` """
        full_graph_path = self.processed_paths[idx]
        neighbor_graph_path = full_graph_path.replace(
            '.gpt', '.%d-hop.gpt' % self.subsample_neighbor_size)
        if not os.path.exists(neighbor_graph_path):
            print("Subgraph save %s not found" % neighbor_graph_path)
            return self.get_subgraph(idx, center_ind)

        neighbor_graphs = torch.load(neighbor_graph_path)
        for j, ng in enumerate(neighbor_graphs):
            self.cached_data[(idx, j)] = ng
        return self.cached_data[(idx, center_ind)]

    def get_saved_subgraph(self, idx, center_ind):
        """ (deprecated) Read subgraph from individual file, use after calling `save_all_subgraphs` """
        full_graph_path = self.processed_paths[idx]
        neighbor_graph_path = os.path.join(
            os.path.split(full_graph_path)[0],
            '%d-hop_neighborgraph' % self.subsample_neighbor_size,
            os.path.split(full_graph_path)[1].replace('.gpt', '.%d-hop.%d.gpt' % (self.subsample_neighbor_size, center_ind)))
        if not os.path.exists(neighbor_graph_path):
            print("Subgraph save %s not found" % neighbor_graph_path)
            return self.get_subgraph(idx, center_ind)

        neighbor_graph = torch.load(neighbor_graph_path)
        self.cached_data[(idx, center_ind)] = neighbor_graph
        return neighbor_graph

    def get_subgraph(self, idx, center_ind):
        """ Generate subgraph on the fly """
        data = self.get_full(idx)
        if not self.subgraph_allow_distant_edge:
            edge_type_mask = (data.edge_attr[:, 0] == EDGE_TYPES["neighbor"])
        else:
            edge_type_mask = None
        sub_node_inds = k_hop_subgraph(int(center_ind),
                                       self.subsample_neighbor_size,
                                       data.edge_index,
                                       edge_type_mask=edge_type_mask,
                                       relabel_nodes=False,
                                       num_nodes=data.x.shape[0])[0]

        if self.subgraph_size_limit > 0:
            assert "center_coord" in self.node_features
            coord_feature_inds = [i for i, n in enumerate(
                self.node_feature_names) if n.startswith('center_coord')]
            assert len(coord_feature_inds) == 2
            center_cell_coord = data.x[[center_ind]][:, coord_feature_inds]
            neighbor_cells_coord = data.x[sub_node_inds][:, coord_feature_inds]
            dists = ((neighbor_cells_coord - center_cell_coord)**2).sum(1).sqrt()
            sub_node_inds = sub_node_inds[(dists < self.subgraph_size_limit)]

        sub_x = data.x[sub_node_inds]
        sub_edge_index, sub_edge_attr = subgraph(sub_node_inds,
                                                 data.edge_index,
                                                 edge_attr=data.edge_attr,
                                                 relabel_nodes=True)

        relabeled_node_ind = list(sub_node_inds.numpy()).index(center_ind)

        sub_data = {'center_node_index': relabeled_node_ind,
                    'original_center_node': center_ind,
                    'x': sub_x,
                    'edge_index': sub_edge_index,
                    'edge_attr': sub_edge_attr}
        for k in data:
            if not k[0] in sub_data:
                sub_data[k[0]] = k[1]

        sub_data = tg.data.Data.from_dict(sub_data)
        self.cached_data[(idx, center_ind)] = sub_data
        return sub_data

    def __getitem__(self, idx):
        data = self.get(self.indices()[idx])
        # for transform_fn in self.transform:
        #     data = transform_fn(data)
        return data

    def plot_subgraph(self, idx, center_ind, n=None):
        """ Plot neighborhood around node `center_ind` as voronoi """

        n = self.subsample_neighbor_size if n is None else n

        data = self.get_full(idx)
        nx_graph = nx.read_gpickle(self.raw_paths[idx])
        assert self.cell_types[nx_graph.nodes[center_ind]['cell_type']] == \
            data.x[center_ind, 0].item()

        # Same procedure as get_subgraph
        if not self.subgraph_allow_distant_edge:
            edge_type_mask = (data.edge_attr[:, 0] == EDGE_TYPES["neighbor"])
        else:
            edge_type_mask = None
        sub_node_inds = k_hop_subgraph(int(center_ind),
                                       n,
                                       data.edge_index,
                                       edge_type_mask=edge_type_mask,
                                       relabel_nodes=False,
                                       num_nodes=data.x.shape[0])[0]

        if self.subgraph_size_limit > 0:
            assert "center_coord" in self.node_features
            coord_feature_inds = [i for i, n in enumerate(
                self.node_feature_names) if n.startswith('center_coord')]
            assert len(coord_feature_inds) == 2
            center_cell_coord = data.x[[center_ind]][:, coord_feature_inds]
            neighbor_cells_coord = data.x[sub_node_inds][:, coord_feature_inds]
            dists = ((neighbor_cells_coord - center_cell_coord)**2).sum(1).sqrt()
            sub_node_inds = sub_node_inds[(dists < self.subgraph_size_limit)]

        sub_node_inds = sub_node_inds.data.numpy().astype(int)
        G = nx_graph.subgraph(sub_node_inds)
        x_c, y_c = G.nodes[center_ind]['center_coord']

        plot_codex_graph(G, cell_types=self.cell_types)
        xmin, xmax = plt.gca().xaxis.get_data_interval()
        ymin, ymax = plt.gca().yaxis.get_data_interval()
        scale = max(x_c - xmin, xmax - x_c, y_c - ymin, ymax - y_c) * 1.05
        plt.xlim(x_c - scale, x_c + scale)
        plt.ylim(y_c - scale, y_c + scale)
        plt.plot([x_c], [y_c], 'x', markersize=5, color='k')


class InfDataLoader(DataLoader):
    def __len__(self):
        return int(1e10)


class GNNgraphSampler(object):
    def __init__(self,
                 dataset,
                 selected_inds=None,
                 batch_size=64,
                 num_graphs_per_segment=32,
                 steps_per_segment=1000,
                 num_workers=None,
                 seed=None,
                 **kwargs):
        self.dataset = dataset
        self.selected_inds = list(
            dataset.indices()) if selected_inds is None else list(selected_inds)
        self.dataset.set_indices(self.selected_inds)

        self.batch_size = batch_size
        self.num_graphs_per_segment = num_graphs_per_segment
        self.steps_per_segment = steps_per_segment
        self.num_workers = multiprocessing.cpu_count(
        ) if num_workers is None else num_workers

        self.graph_inds_q = []
        self.fill_queue(seed=seed)

        self.step_counter = 0
        self.data_iter = None
        print("Initiate data loader, subgraph source: %s" %
              self.dataset.subgraph_source)
        self.get_new_segment()

    def fill_queue(self, seed=None):
        if not seed is None:
            np.random.seed(seed)
        fill_inds = sorted(set(self.selected_inds) - set(self.graph_inds_q))

        np.random.shuffle(fill_inds)
        self.graph_inds_q.extend(fill_inds)

    def get_new_segment(self):
        if self.num_graphs_per_segment <= 0:
            self.dataset.set_indices(self.selected_inds)
        else:
            graph_inds_in_segment = self.graph_inds_q[:self.num_graphs_per_segment]
            self.graph_inds_q = self.graph_inds_q[self.num_graphs_per_segment:]
            if len(self.graph_inds_q) < self.num_graphs_per_segment:
                self.fill_queue()

            self.dataset.clear_cache()
            print(1)
            self.dataset.set_indices(graph_inds_in_segment)
            for ind in graph_inds_in_segment:
                self.dataset.load_to_cache(ind, subgraphs=False)
        print(0)
        sampler = RandomSampler(
            self.dataset, replacement=True, num_samples=int(1e10))

        loader = InfDataLoader(self.dataset,
                               batch_size=self.batch_size,
                               sampler=sampler,
                               num_workers=self.num_workers)
        self.data_iter = iter(loader)
        self.step_counter = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.step_counter == self.steps_per_segment:
            self.get_new_segment()
        if not len(set(self.dataset.indices()) - set(self.selected_inds)) == 0:
            self.get_new_segment()
        batch = next(self.data_iter)
        self.step_counter += 1
        return batch
