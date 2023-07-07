"""Contains utilities methods and variables related to the project's traininf process."""

# ----- Standard Imports

# ----- Third Party Imports
import numpy as np
import torch as tr
from torch_geometric.utils import dropout_edge as tr_dropout_edge
from torchmetrics.functional import spearman_corrcoef, pearson_corrcoef

from icecream import ic
ic.configureOutput(includeContext=True)

# ----- Library Imports


# ----------------------
# --- Fairness Utilities
# ----------------------

def compute_sensitive_feature_correlations(data_matrix,
                                           sensitive_vect,
                                           method: str = 'spearman',
                                           absolute_values = False,
                                          ):
    assert method in ['pearson', 'spearman']
    
    if method == 'spearman':
        corr_function = spearman_corrcoef
    else: #method == 'pearson'
        corr_function = pearson_corrcoef
    
    feature_correlations = tr.empty(data_matrix.shape[1], dtype=float)
    for col_idx in range(data_matrix.shape[1]):
        feature_correlations[col_idx] = corr_function(data_matrix[:, col_idx], sensitive_vect)
        
    if absolute_values:
        feature_correlations = feature_correlations.abs()
    
    return feature_correlations


def dropout_edge(edges_indices,
                 sensitive_vect,
                 base_drop_rate,
                 highest_homo_perc = -1,
                ):    
    # ----- Argument validation
    # TODO
    #heterophily_mask = sensitive_vect[edges_indices[0]] != sensitive_vect[edges_indices[1]]
    #uni_hetero, count_hetero = tr.unique(heterophily_mask, return_counts=True)
    #print(f"Initial nodes heterophily: {[f'{val}: {cnt}' for val, cnt in zip(uni_hetero, count_hetero)]}")
    #print(f"[rateo: {round((count_hetero[1]/(count_hetero.sum())).item(),3)}]")
    
    # ----- Standard edges dropout
    # note: we assume that the graph is directed while dropping
    #kept_mask = tr.rand(edge_index.shape[1]) >= base_drop_rate
    #kept_edges_idxs = tr.masked_select(edges_indices, kept_mask).reshape(2, kept_mask.sum())
    
    kept_edges_idxs, kept_edges_mask = tr_dropout_edge(edges_indices, p=base_drop_rate, force_undirected=False)
    
    if highest_homo_perc != -1:
        # ----- Count edges with same/different sensitive attribute
        kept_hetero_msk = sensitive_vect[kept_edges_idxs[0]] != sensitive_vect[kept_edges_idxs[1]]
        # ----- If the edges are either all homo or hetero, fairdrop can't be performed
        if kept_hetero_msk.all() or (not kept_hetero_msk.any()):
            return kept_edges_idxs
        
        n_homo_edges, n_hetero_edges = tr.bincount(kept_hetero_msk.int())
        curr_homo_perc = (n_homo_edges / (n_homo_edges + n_hetero_edges)).item()
        
        #uni_hetero, count_hetero = tr.unique(kept_hetero_msk, return_counts=True)
        #print(f"random drop nodes heterophily: {[f'{val}: {cnt}' for val, cnt in zip(uni_hetero, count_hetero)]}")
        #print(f"[rateo: {round((count_hetero[1]/(count_hetero.sum())).item(),3)}]")

        # ---- If the percentage of homophily edges is greater than the desired one, then activate fairdrop
        if curr_homo_perc > highest_homo_perc:
            orig_hetero_mask = sensitive_vect[edges_indices[0]] != sensitive_vect[edges_indices[1]]
            # ----- retrieve the swap edges pool
            kept_homo = (kept_edges_mask & ~orig_hetero_mask).nonzero().squeeze(-1)
            n_kpt_homo = len(kept_homo)
            removed_hetero = (~kept_edges_mask & orig_hetero_mask).nonzero().squeeze(-1)
            n_rmv_hetero = len(removed_hetero)

            # ----- retrieve the number of swaps to obtain the desired homo edge percentage
            diff = int((curr_homo_perc-highest_homo_perc) * kept_edges_idxs.shape[1])
            # if there's not enough edges to cover for the total swaps
            if n_kpt_homo < diff or n_rmv_hetero < diff:
                #print(f"caution: wanted to perform {diff} swaps but:")
                #print(f"there are only {n_kpt_homo} kept homo and {n_rmv_hetero} removed hetero")
                diff = min(n_kpt_homo, n_rmv_hetero)

            #print(f"selected diff: {diff}")

            # ----- select and swap the homo-edges with hetero-ones
            rng = np.random.default_rng() # NOTE: random seed is not fixed
            to_rem_idxs = kept_homo[rng.choice(n_kpt_homo, size=diff, replace=False)]
            to_add_idxs = removed_hetero[rng.choice(n_rmv_hetero, size=diff, replace=False)]

            kept_edges_mask[to_rem_idxs] = False
            kept_edges_mask[to_add_idxs] = True
            kept_edges_idxs[0] = edges_indices[0, kept_edges_mask]
            kept_edges_idxs[1] = edges_indices[1, kept_edges_mask]
            
        #kept_hetero_msk = sensitive_vect[kept_edges_idxs[0]] != sensitive_vect[kept_edges_idxs[1]]
        #uni_hetero, count_hetero = tr.unique(kept_hetero_msk, return_counts=True)
        #print(f"final nodes heterophily: {[f'{val}: {cnt}' for val, cnt in zip(uni_hetero, count_hetero)]}")
        #print(f"[rateo: {round((count_hetero[1]/(count_hetero.sum())).item(),3)}]")
        
    return kept_edges_idxs


def drop_feature(data_matrix,
                 features_drop_rate,
                 sens_feat_idx = None,
                 correlated_attrs = None,
                 correlated_weights = None,
                 flip_sensitive = False,
                ):
    # ----- Argument validation
    assert data_matrix.ndim == 2
    assert 0 <= features_drop_rate < 1
    assert sens_feat_idx is None or isinstance(sens_feat_idx, int)
    assert not flip_sensitive or sens_feat_idx is not None
    assert not flip_sensitive or (tr.unique(data_matrix[:,sens_feat_idx]).cpu() == tr.tensor([0,1])).all()
    assert correlated_attrs is None or correlated_weights is not None
    
    # ----- Initialize random mask for dropping features
    drop_mask = tr.rand(data_matrix.shape[1]) < features_drop_rate
    
    # ----- Fair drop features
    if correlated_attrs is not None:
        augm_proba = correlated_weights + features_drop_rate
        drop_mask[correlated_attrs] = tr.rand(len(correlated_attrs)) < augm_proba
    
    # ----- Do not drop the sensitive feature
    if sens_feat_idx is not None:
        drop_mask[sens_feat_idx] = False 
    
    # it's not a drop, but a noise adding
    noisy_data = data_matrix.clone()
    # TOCHECK -  shouldn't it be different for every column?
    noisy_data[:, drop_mask] += tr.randn(1, device = noisy_data.device)

    if flip_sensitive:
        noisy_data[:, sens_feat_idx] = 1 - noisy_data[:, sens_feat_idx]

    return noisy_data


# -----------------
# --- Miscellaneous
# -----------------

def get_string_device_from_index(torch_device):
    assert isinstance(torch_device, int), "'torch_device' must be the torch device integer index (-1 for cpu)."
    if tr.cuda.is_available() and torch_device != -1:
        return f"cuda:{torch_device}"
    return "cpu"
