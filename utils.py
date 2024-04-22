import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pickle
from sklearn.preprocessing import normalize
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils.convert import to_networkx
from karateclub.community_detection.overlapping import DANMF
from torch_geometric.utils import to_dense_adj, dense_to_sparse


def write_list(a_list, name=""):
    with open(name, 'wb') as fp:
        pickle.dump(a_list, fp)
        print('Done writing list into a binary file')

def read_list(name=""):
    with open(name, 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list

def danmf_clustering(dataset_name='Cora', graph='', clustering_overlap=False, membership_closeness=0.5):
    """
    Clustering the graph with DANMF. For details see:
    """
    num_labels = {'CiteSeer': 6, 'Cora': 7, 'PubMed': 3, 'WikiCS': 10}
    model = DANMF(layers=[32, 2 * num_labels[dataset_name]], pre_iterations=500, iterations=200)
    model.fit(graph)
    values = model.get_memberships().values()
    values_list = list(values)

    if clustering_overlap == False:
        near_clusters = values
    else:
        P = normalize(model._P, axis=1)
        near_clusters = []
        for i in range(P.shape[0]):
            row = P[i]
            max_in_row = np.max(row)
            npw = np.where(row >= (max_in_row * membership_closeness))
            tmp = npw[0].tolist()
            if max_in_row == 0:
                cluster_indices = [tmp[0]]
            else:
                cluster_indices = [x for x in tmp if x in values_list]
            near_clusters.append(cluster_indices)

    clusters = list(set(values_list))
    cluster_membership = {node: membership for node, membership in enumerate(near_clusters)}
    return cluster_membership

def plot_grah(graph):
    fig = plt.figure(figsize=(10, 10))
    nx.draw_networkx(graph,
                    pos=nx.spring_layout(graph, seed=0),
                    with_labels=False,
                    node_size=20,
                    #node_color=label,
                    cmap="hsv",
                    vmin=-10,
                    vmax=10,
                    width=0.1,
                    edge_color="grey",
                    font_size=1
                    )
    plt.show()
    return 0

def read_clusters(d_name='Cora'):
    dataset = Planetoid(root='data/Planetoid', name=d_name)
    data = dataset[0]
    graph = to_networkx(data, to_undirected=True)

    cluster_membership = danmf_clustering(dataset_name=d_name, graph=graph, clustering_overlap=True, membership_closeness=1)
    Cluster = np.array(list(cluster_membership.items()))    
    
    clu = []
    overlapping_nodes=[]
    for i in range(len(Cluster)):
        if (len(Cluster[i][1])>1):
            overlapping_nodes.append(Cluster[i].tolist())
            for j in range(len(Cluster[i][1])):
                clu.append([i, Cluster[i][1][j]])
        elif (len(Cluster[i][1]) == 1):
            clu.append([i, Cluster[i][1][0]])
    num_cluster=max(Cluster[:,1])[0]
    cluster_list=[]
    for i in range(num_cluster+1):
        G = to_networkx(data, to_undirected=True)
        x = (np.where(np.array(clu) == i)[0])
        cluster_c = np.array(clu)[x, 0]
        c = list(set(Cluster[:, 0]).difference(set(cluster_c)))
        for j in c:
            G.remove_node(j)
        cluster_list.append(G)
    return  cluster_list,overlapping_nodes,data  

