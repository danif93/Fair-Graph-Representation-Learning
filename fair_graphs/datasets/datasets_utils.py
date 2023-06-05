"""Contains utilities methods and variables related to the project's datasets definitions."""

# ----- Standard Imports
import pickle
from copy import deepcopy
from itertools import product
import scipy.sparse as sp
from scipy.spatial import distance_matrix

# ----- Third Party Imports
import numpy as np
import torch as tr
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import pandas as pd

# ----- Library Imports
#from fair_graphs.datasets.graph_datasets import _GraphDataset # error: circular import


def edges_coo_from_adj_matrix(adj_mtx):
    return tr.tensor(np.array(adj_mtx.nonzero()), dtype=int)
    #return tr.stack(adj_mtx.nonzero(as_tuple=True))


# -------------------
# --- Splitting Utils
# -------------------

def train_test_split_graph_data(dataset,#: _GraphDataset,
                                *,
                                test_size = None,
                                train_size = None,
                                random_state = None,
                               ):
    # ----- Argument validation
    # TODO
    
    # ----- Instantiate train, test data and splitter utility
    tr_data, ts_data = deepcopy(dataset), deepcopy(dataset)
    splitter = ShuffleSplit(n_splits = 1,
                            test_size = test_size,
                            train_size = train_size,
                            random_state = random_state)
    
    # ----- Retrieve split indices and sample data
    tr_idxs, ts_idxs = next(splitter.split(tr_data.samples))
    tr_data.sample_data_from_indices(tr_idxs)
    ts_data.sample_data_from_indices(ts_idxs)
    
    return tr_data, ts_data


# ---------------------------------
# --- Data Cleaning Utility Methods
# ---------------------------------

def build_relationship(x, thresh = 0.5):
    """Given a data matrix, retrieve the set of edges connecting samples close with each other."""
    # ----- Arguments checking
    assert x.ndim == 2
    assert 0 <= thresh <= 1
    
    # ----- Build all the samples pairwise similarities
    euclid_dist = 1 / (1 + distance_matrix(x, x, p = 2))
    
    # ----- For each samples, retrieve the closest points
    edges = []
    for idx in range(euclid_dist.shape[0]):
        max_sim = np.partition(euclid_dist[idx], -2)[-2]
        neig_idxs = np.where(euclid_dist[idx] >= thresh*max_sim)[0]
        edges += [(idx, neig) for neig in neig_idxs if neig != idx]
    return np.array(edges, dtype=int)


def clean_store_graph_dataset(path_to_file,
                              sensitive_attr,
                              prediction_attr,
                              evaluated_group = None,
                              evaluated_class = None,
                              dropped_cols = None,
                              #one_hot_encode_categorical = False,
                              relationship_file = None,
                             ):
    """Clean, standardize and store project's graph data."""
    # ----- Arguments checking
    assert isinstance(sensitive_attr, str) and isinstance(prediction_attr, str)
    
    if isinstance(dropped_cols, str):
        dropped_cols = [dropped_cols]
    if dropped_cols is None:
        dropped_cols = []
    assert isinstance(dropped_cols, list)
    
    # ----- Load raw data
    raw_dataset = pd.read_csv(path_to_file)
    
    # ----- Select features columns
    to_drop = [prediction_attr] + dropped_cols
    cols = raw_dataset.columns.drop(to_drop)
    
    # ----- Sensitive attribute and labels encoding
    unique_sens, inverse_map = np.unique(raw_dataset[sensitive_attr], return_inverse=True)
    mapped_values = np.arange(len(unique_sens))
    # optionally, we can encode a single sensitive group to have sensitive value = 1
    # and the other groups to have value = 0 (OvA)
    if evaluated_group is not None:
        map_sens_val = np.where(unique_sens == evaluated_group)[0].item()
        mapped_values = [int(new_val == map_sens_val) for new_val in mapped_values]
        inverse_map = [int(new_val == map_sens_val) for new_val in inverse_map]
    sensitive_map = {new_val: orig_val for orig_val, new_val in zip(unique_sens, mapped_values)}
    raw_dataset[sensitive_attr] = inverse_map
    
    unique_lbls, inverse_map = np.unique(raw_dataset[prediction_attr], return_inverse=True)
    mapped_values = np.arange(len(unique_lbls))
    # as it has been done for the sensitive groups, perform OvA for the target labels
    if evaluated_class is not None: # OvA
        map_lbl_val = np.where(unique_lbls == evaluated_class)[0].item()
        mapped_values = [int(new_val == map_lbl_val) for new_val in mapped_values]
        inverse_map = [int(new_val == map_lbl_val) for new_val in inverse_map]
    labels_map = {new_val: orig_val for orig_val, new_val in zip(unique_lbls, mapped_values)}
    raw_dataset[prediction_attr] = inverse_map
    
    # ----- One-hot encode categorical variables
    # TBD, use pd.get_dummies(...) once the column type has been set as categorical (or at least something that is not numerical)
    #if one_hot_encode_categorical:
    
    # ----- Retrieve features, sensitive and target attributes
    features = raw_dataset[cols.drop(sensitive_attr)].values.astype(float)
    sensitive = raw_dataset[sensitive_attr].values.astype(int)
    labels = raw_dataset[prediction_attr].values.astype(int)
    
    # ----- Build connected graph or load relationship from txt file
    if relationship_file is None:
        edges = build_relationship(raw_dataset[cols], thresh=0.8)
    else:
        edges = np.loadtxt(relationship_file, dtype=int)
    
    adj_mtx = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(features.shape[0], features.shape[0]), dtype=float)
    adj_mtx += adj_mtx.T.multiply(adj_mtx.T > adj_mtx) - adj_mtx.multiply(adj_mtx.T > adj_mtx)
    adj_mtx += sp.eye(adj_mtx.shape[0])
    #adj_mtx = adj_mtx.toarray()
    
    # ----- Store data
    new_path_to_file = path_to_file.split('.')[0]
    pickle.dump({'features': features,
                 'sensitive': sensitive,
                 'sensitive_map': sensitive_map,
                 'labels': labels,
                 'labels_map': labels_map,
                 'adjacency_matrix': adj_mtx},
                open(f'{new_path_to_file}_sensitive_{sensitive_attr}_label_{prediction_attr}.pickle', 'wb'))


# -----------------
# --- Visualization
# -----------------

def plot_distributions_sunburst(graph_data, obj='samples'):
    assert obj in ['samples','edges']
    
    classes, smpls_classes = np.unique(graph_data.labels, return_counts=True)
    pos_msk, neg_msk = graph_data.labels==classes[1], graph_data.labels==classes[0]
    
    colors_dict = {0:{'main':'#31a354', 0:'#74c476', 1:'#006d2c'},
                   1:{'main':'#de2d26', 0:'#fc9272', 1:'#a50f15'}}
    
    if obj == 'samples':
        plt.title("Sensitive/class samples' distribution")
        
        groups, smpls_groups = np.unique(graph_data.sensitive, return_counts=True)
        msk_0, msk_1 = graph_data.sensitive==groups[0], graph_data.sensitive == groups[1]
        
        n_tot = len(graph_data)
        n_neg, n_pos = smpls_classes[0], smpls_classes[1]
        n_0, n_1 = smpls_groups[0], smpls_groups[1]
        n_neg_0, n_neg_1 = (msk_0 & neg_msk).sum(), (msk_1 & neg_msk).sum()
        n_pos_0, n_pos_1 = (msk_0 & pos_msk).sum(), (msk_1 & pos_msk).sum()
        
    else:
        plt.title("Homophily/class start-point edges' distribution")
        groups = ['hetero','homo']
        coo_edg = graph_data.adj_mtx.nonzero()
        neg_coo_edg = graph_data.adj_mtx[neg_msk].nonzero()
        pos_coo_edg = graph_data.adj_mtx[pos_msk].nonzero()
        smpls_classes = [len(neg_coo_edg[0]), len(pos_coo_edg[0])]
    
        n_tot = len(coo_edg[0])
        n_neg, n_pos = len(neg_coo_edg[0]), len(pos_coo_edg[0])
        n_0 = (graph_data.sensitive[coo_edg[0]] != graph_data.sensitive[coo_edg[1]]).sum()
        n_1 = (graph_data.sensitive[coo_edg[0]] == graph_data.sensitive[coo_edg[1]]).sum()
        n_neg_0 = (graph_data.sensitive[neg_coo_edg[0]] != graph_data.sensitive[neg_coo_edg[1]]).sum()
        n_neg_1 = (graph_data.sensitive[neg_coo_edg[0]] == graph_data.sensitive[neg_coo_edg[1]]).sum()
        n_pos_0 = (graph_data.sensitive[pos_coo_edg[0]] != graph_data.sensitive[pos_coo_edg[1]]).sum()
        n_pos_1 = (graph_data.sensitive[pos_coo_edg[0]] == graph_data.sensitive[pos_coo_edg[1]]).sum()

    size = 0.4
    plt.pie(smpls_classes, labels=classes, labeldistance=0.2,
            autopct='%1.2f%%', pctdistance=0.6, radius=1-size,
            wedgeprops=dict(width=size, edgecolor='w'),
            colors=[colors_dict[0]['main'], colors_dict[1]['main']])
    
    plt.pie([n_neg_0, n_neg_1, n_pos_1, n_pos_0], labels=[groups[0], groups[1], groups[1], groups[0]],
            autopct='%1.2f%%', pctdistance=0.8, radius=1,
            wedgeprops=dict(width=size, edgecolor='w'),
            colors=[colors_dict[i][j] for i,j in [(0,0),(0,1),(1,1),(1,0)]])
    
    tot_dict =  {f"c{classes[0]}": n_neg, f"c{classes[1]}": n_pos,
                 f"g{groups[0]}": n_0, f"g{groups[1]}": n_1,
                 f"{classes[0]}{groups[0]}": n_neg_0, f"{classes[0]}{groups[1]}": n_neg_1,
                 f"{classes[1]}{groups[0]}": n_pos_0, f"{classes[1]}{groups[1]}": n_pos_1}
    
    print(f"Total {obj}: {n_tot}")
    print("Class distribution:\n\t{}".format('\n\t'.join([f"class {c}: {tot_dict[f'c{c}']}" for c in classes])))
    print("Sensitive distribution:\n\t{}".format('\n\t'.join([f"group {g}: {tot_dict[f'g{g}']}" for g in groups])))
    print("Class-group distribution:\n\t{}".format('\n\t'.join([f"class {c}, group {g}: {tot_dict[f'{c}{g}']}"
                                                                for c,g in product(classes, groups)])))


def plot_cum_distributions_sunburst(graphs_data):
    colors_dict = {0:{'main':'#31a354', 0:'#74c476', 1:'#006d2c'},
                   1:{'main':'#de2d26', 0:'#fc9272', 1:'#a50f15'}}
    
    fig, axs = plt.subplots(2, len(graphs_data), figsize=(15,7), gridspec_kw = {'wspace':0, 'hspace':0})

    for d_idx, (name, graph_data) in enumerate(graphs_data.items()):
        classes, smpls_classes = np.unique(graph_data.labels, return_counts=True)
        pos_msk, neg_msk = graph_data.labels==classes[1], graph_data.labels==classes[0]
    
        for r_idx, obj in enumerate(['samples', 'edges']):
            if r_idx==0:
                axs[r_idx][d_idx].set_title(name, size=13)

            if obj == 'samples':
                if d_idx==0:
                    axs[r_idx][d_idx].set_ylabel("Sensitive/class distribution", labelpad=20, size=13)
                
                groups, smpls_groups = np.unique(graph_data.sensitive, return_counts=True)
                msk_0, msk_1 = graph_data.sensitive==groups[0], graph_data.sensitive == groups[1]
                
                n_tot = len(graph_data)
                n_neg, n_pos = smpls_classes[0], smpls_classes[1]
                n_0, n_1 = smpls_groups[0], smpls_groups[1]
                n_neg_0, n_neg_1 = (msk_0 & neg_msk).sum(), (msk_1 & neg_msk).sum()
                n_pos_0, n_pos_1 = (msk_0 & pos_msk).sum(), (msk_1 & pos_msk).sum()

                groups = [f's={g} 'for g in groups]
                
            else:
                if d_idx==0:
                    axs[r_idx][d_idx].set_ylabel("Homophily/class distribution", labelpad=20, size=13)
                groups = ['hetero','homo']
                coo_edg = graph_data.adj_mtx.nonzero()
                neg_coo_edg = graph_data.adj_mtx[neg_msk].nonzero(as_tuple=True)
                pos_coo_edg = graph_data.adj_mtx[pos_msk].nonzero(as_tuple=True)
                smpls_classes = [len(neg_coo_edg[0]), len(pos_coo_edg[0])]
            
                n_tot = len(coo_edg[0])
                n_neg, n_pos = len(neg_coo_edg[0]), len(pos_coo_edg[0])
                n_0 = (graph_data.sensitive[coo_edg[0]] != graph_data.sensitive[coo_edg[1]]).sum()
                n_1 = (graph_data.sensitive[coo_edg[0]] == graph_data.sensitive[coo_edg[1]]).sum()
                n_neg_0 = (graph_data.sensitive[neg_coo_edg[0]] != graph_data.sensitive[neg_coo_edg[1]]).sum()
                n_neg_1 = (graph_data.sensitive[neg_coo_edg[0]] == graph_data.sensitive[neg_coo_edg[1]]).sum()
                n_pos_0 = (graph_data.sensitive[pos_coo_edg[0]] != graph_data.sensitive[pos_coo_edg[1]]).sum()
                n_pos_1 = (graph_data.sensitive[pos_coo_edg[0]] == graph_data.sensitive[pos_coo_edg[1]]).sum()

            size = 0.5
            axs[r_idx][d_idx].pie(smpls_classes, labels=[f'y={-1 if c==0 else 1}' for c in classes], labeldistance=0.2,
                    #autopct='%1.1f%%',
                    pctdistance=0.6, radius=1-size,
                    wedgeprops=dict(width=size, edgecolor='w'),
                    textprops={'fontsize': 13},
                    colors=[colors_dict[0]['main'], colors_dict[1]['main']])
            
            axs[r_idx][d_idx].pie([n_neg_0, n_neg_1, n_pos_1, n_pos_0], labels=[groups[0], groups[1], groups[1], groups[0]],
                    #autopct='%1.1f%%',
                    pctdistance=0.8, radius=1,
                    wedgeprops=dict(width=size, edgecolor='w'),
                    textprops={'fontsize': 13},
                    colors=[colors_dict[i][j] for i,j in [(0,0),(0,1),(1,1),(1,0)]])
            
            tot_dict =  {f"c{classes[0]}": n_neg, f"c{classes[1]}": n_pos,
                        f"g{groups[0]}": n_0, f"g{groups[1]}": n_1,
                        f"{classes[0]}{groups[0]}": n_neg_0, f"{classes[0]}{groups[1]}": n_neg_1,
                        f"{classes[1]}{groups[0]}": n_pos_0, f"{classes[1]}{groups[1]}": n_pos_1}
    
            #print(f"Total {obj}: {n_tot}")
            #print("Class distribution:\n\t{}".format('\n\t'.join([f"class {c}: {tot_dict[f'c{c}']}" for c in classes])))
            #print("Sensitive distribution:\n\t{}".format('\n\t'.join([f"group {g}: {tot_dict[f'g{g}']}" for g in groups])))
            #print("Class-group distribution:\n\t{}".format('\n\t'.join([f"class {c}, group {g}: {tot_dict[f'{c}{g}']}"
            #                                                        for c,g in product(classes, groups)])))
    #fig.tight_layout()
    #fig.subplots_adjust(wspace=0, hspace=0)
    