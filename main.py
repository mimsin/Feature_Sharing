import numpy as np
from model import GCN
from torch_geometric.utils.convert import from_networkx
from sklearn.preprocessing import normalize
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils.convert import to_networkx
#from karateclub.community_detection.overlapping import DANMF
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from utils import *
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import random



def combine_outputs(outputs,overlapping_nodes,cluster_graph_list):
    temp=[]
    for t in outputs:
        temp.append(torch.clone(t))

    with torch.no_grad():
        for ov in overlapping_nodes:
            l=[]
            sum=0
            for cl in ov[1]:
                t=np.array(cluster_graph_list[cl].nodes)
                loc=np.where(t==ov[0])[0][0]
                sum=sum+outputs[cl][loc]
                l.append(outputs[cl][loc])
            sum=sum/len(ov[1])
            random.shuffle(l)
            i=0
            for cl in ov[1]:
                t=np.array(cluster_graph_list[cl].nodes)
                loc=np.where(t==ov[0])[0][0]
                #temp[cl][loc]=l[i]
                temp[cl][loc]=sum
                i+=1
    return temp


def model_test(model, features, edge_list, target, test_mask):
    model.eval()
    temp = model(features, edge_list,0)
    out = model(temp, edge_list,1)
    # Use the class with highest probability.
    pred = out.argmax(dim=1)
    # Check against ground-truth labels.
    test_mask=torch.gt(test_mask, 0)
    test_correct = pred[test_mask] == target[test_mask]
    #print(test_mask.sum(),test_correct.sum(),test_mask.size())
    # Derive ratio of correct predictions.
    test_acc = int(test_correct.sum()) / int(test_mask.sum())
    return test_acc

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cluster_graph_list,overlapping_nodes,data=read_clusters(d_name='Cora',c_name='Cluster')
    #Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], 
    #train_mask=[2708], val_mask=[2708], test_mask=[2708])
    learning_rate = 0.01
    decay = 5e-4
    criterion = torch.nn.CrossEntropyLoss()
    num_features=data.num_features
    num_classes=7
    num_clusters=len(cluster_graph_list)
    epochs=1001
    model_list = []
    optimizers=[]
    for i in range(num_clusters):
        #torch.tensor([[j[0] for j in edges ],[j[1] for j in edges ]
        model=GCN(in_channels=num_features, out_channels=num_classes, hidden_channels=16)
        model = model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
        model_list.append(model)
        optimizers.append(opt)
    for epoch in range(epochs):
        out1=[]
        for i in range(num_clusters):
            features=data.x[list(cluster_graph_list[i].nodes)]
            pyg_graph = from_networkx(cluster_graph_list[i])
            model=model_list[i]
            model.train()
            out1.append(model(features, pyg_graph.edge_index,0))
        outcome=combine_outputs(out1,overlapping_nodes,cluster_graph_list)
        outs=[]
        for i in range(num_clusters):
            features=outcome[i]
            pyg_graph = from_networkx(cluster_graph_list[i])
            model=model_list[i]
            model.train()
            outs.append(model(features, pyg_graph.edge_index,1))
        outcome=combine_outputs(outs,overlapping_nodes,cluster_graph_list)
        for i in range(num_clusters):
            opt=optimizers[i]
            mask=data.train_mask[list(cluster_graph_list[i].nodes)]
            opt.zero_grad()
            mask=torch.gt(mask, 0)
            loss = criterion(outs[i][mask],  data.y[list(cluster_graph_list[i].nodes)][mask])
            loss.backward()
            opt.step()
            if epoch % 100 == 0:
                print(f'Cluster: {i} Epoch: {epoch:03d}, Loss: {loss:.4f}')
    count=0
    acc_count=0
    for i in range(num_clusters-1):
        features=data.x[list(cluster_graph_list[i].nodes)]
        pyg_graph = from_networkx(cluster_graph_list[i])

        test_acc = model_test(model=model_list[i], features=features, edge_list=pyg_graph.edge_index,
                              target=data.y[list(cluster_graph_list[i].nodes)], 
                              test_mask=data.test_mask[list(cluster_graph_list[i].nodes)])
        count+=data.test_mask[list(cluster_graph_list[i].nodes)].sum()
        acc_count+=test_acc*data.test_mask[list(cluster_graph_list[i].nodes)].sum()
        print(f'Test Accuracy: {test_acc:.4f}')
    print(f'Overall Accuracy: {acc_count/count:.4f}')



