# from data import GNNDataset,nx_to_tg_graph,process_feature,GNNgraphSampler
from data import nx_to_tg_graph, GNNDataset, GNNgraphSampler
import torch

tensor_dir = 'data/tensor_graphs'

# generate nx graphs from raw data

for file in os.listdir(graph_dir):
    if not file.endswith('.DS_Store'):
        graph_path = os.path.join(graph_dir, file)
        print(graph_path)
        G = nx.read_gpickle(graph_path)
        data = nx_to_tg_graph(G)
        torch.save(data, os.path.join(tensor_dir, '%s.gpt' % file[:-5]))
