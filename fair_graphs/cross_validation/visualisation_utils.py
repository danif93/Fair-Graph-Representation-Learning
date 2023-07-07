# ----- Standard Imports
import pickle
import os
import matplotlib.pyplot as plt

# ----- Third-Party Imports
import pandas as pd
import numpy as np

# ----- Library Imports


# -------------
# --- Utilities
# -------------

def pareto_mask(pareto_indexes, n_start_point):
    is_efficient_mask = np.zeros(n_start_point, dtype=bool)
    is_efficient_mask[pareto_indexes] = True
    return is_efficient_mask
    
# Very slow for many datapoints. Fastest for many costs, most readable
def is_pareto_efficient_dumb(costs, return_mask=False):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    n_points = costs.shape[0]
    is_efficient = np.ones(n_points, dtype = bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(costs[:i]>c, axis=1)) and np.all(np.any(costs[i+1:]>c, axis=1))
    if return_mask:
        return pareto_mask(is_efficient, n_points)
    else:
        return is_efficient

# Fairly fast for many datapoints, less fast for many costs, somewhat readable
def is_pareto_efficient_simple(costs, return_mask=False):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    n_points = costs.shape[0]
    is_efficient = np.ones(n_points, dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    if return_mask:
        return pareto_mask(is_efficient, n_points)
    else:
        return is_efficient

# Faster than is_pareto_efficient_simple, but less readable.
def is_pareto_efficient(costs, return_mask=False):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs>costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        return pareto_mask(is_efficient, n_points)
    else:
        return is_efficient


# ----------
# --- Tables
# ----------

def print_selection_table(full_fn,
                          phase,
                          selection_strategies,
                          eval_metrics,
                          ):
    pd.options.display.max_colwidth = 120
    selection_results = pickle.load(open(full_fn, 'rb'))
    
    table_results = pd.DataFrame(columns=eval_metrics, index=selection_strategies)
    
    for sel_strat, eval_res in selection_results.items():
        if sel_strat not in selection_strategies: continue
        
        for eval_metr, score in eval_res.items():
            if eval_metr != 'params':
                metr = '_'.join(eval_metr.split('_')[2:])
                if f"mean_{phase}" not in eval_metr or metr not in eval_metrics:
                    continue
                avg_score = score
                std_score = eval_res[f"std_{phase}_{metr}"]
                score = f"{avg_score:.3f} ± {std_score:.3f}"
            else:
                metr = 'params'
                score = str(score)
            table_results.loc[sel_strat, metr] = score
        
    return table_results
    

# ---------
# --- Plots
# ---------

def plot_cloud_distributions_acc_roc(plot_dict,
                                     phase = 'test',
                                     file_path = '',
                                     pareto_front = False,
                                     ):
    aux_dict = {'DDP': {'fair': '#78c679', 'nofair': '#006837', 'lbl':'DDP'},
                'DEO+': {'fair': '#41b6c4', 'nofair': '#253494', 'lbl':'DEO$^+$'},
                'DEO-': {'fair': '#fecc5c', 'nofair': '#bd0026', 'lbl':'DEO$^-$'},
                'CF': {'fair': '#8c96c6', 'nofair': '#810f7c', 'lbl':'CF'}}

    fig, axs = plt.subplots(2,len(plot_dict), figsize=(13,6), sharey='row', sharex='col')
    axs[0,0].set_ylabel('ACC', size=22)
    axs[1,0].set_ylabel('AUROC', size=22)
    for idx, (metric, setting_dict) in enumerate(plot_dict.items()):
        nofair_res = pickle.load(open(os.path.join(file_path, setting_dict['nofair_fn']), 'rb'))
        fair_res = pickle.load(open(os.path.join(file_path, setting_dict['fair_fn']), 'rb'))
    
        axs[1,idx].set_xlabel(aux_dict[metric]['lbl'], size=22)
        #axs[0,idx].grid(True, alpha=.3)
        #axs[1,idx].grid(True, alpha=.3)
        
        x = fair_res[f"mean_{phase}_{setting_dict['fair_metr']}"].to_numpy()
        y_0 = fair_res[f"mean_{phase}_accuracy"].to_numpy()
        y_1 = fair_res[f"mean_{phase}_roc"].to_numpy()
        if pareto_front:
            mask = is_pareto_efficient(np.stack((x,y_0), axis=1), return_mask=True)
            axs[0,idx].scatter(abs(x[mask]),y_0[mask], c=aux_dict[metric]['fair'], label='Our', s=90)
            axs[0,idx].scatter(abs(x[~mask]),y_0[~mask], facecolor='none', edgecolors=aux_dict[metric]['fair'], alpha=.4, s=60)
            mask = is_pareto_efficient(np.stack((x,y_1), axis=1), return_mask=True)
            axs[1,idx].scatter(abs(x[mask]),y_1[mask], c=aux_dict[metric]['fair'], label='Our', s=90)
            axs[1,idx].scatter(abs(x[~mask]),y_1[~mask], facecolor='none', edgecolors=aux_dict[metric]['fair'], alpha=.4, s=60)
        else:
            axs[0,idx].scatter(abs(x),y_0, c=aux_dict[metric]['fair'], label='Our', s=90)
            axs[1,idx].scatter(abs(x),y_1, c=aux_dict[metric]['fair'], label='Our', s=90)

        x = nofair_res[f"mean_{phase}_{setting_dict['fair_metr']}"].to_numpy()
        y_0 = nofair_res[f"mean_{phase}_accuracy"].to_numpy()
        y_1 = nofair_res[f"mean_{phase}_roc"].to_numpy()
        if pareto_front:
            mask = is_pareto_efficient(np.stack((x,y_0), axis=1), return_mask=True)
            axs[0,idx].scatter(abs(x[mask]),y_0[mask], c=aux_dict[metric]['nofair'], label='Nifty', s=90)
            axs[0,idx].scatter(abs(x[~mask]),y_0[~mask], facecolors='none', edgecolors=aux_dict[metric]['nofair'], alpha=.4, s=60)
            mask = is_pareto_efficient(np.stack((x,y_1), axis=1), return_mask=True)
            axs[1,idx].scatter(abs(x[mask]),y_1[mask], c=aux_dict[metric]['nofair'], label='Nifty', s=90)
            axs[1,idx].scatter(abs(x[~mask]),y_1[~mask], facecolors='none', edgecolors=aux_dict[metric]['nofair'], alpha=.4, s=60)
        else:
            axs[0,idx].scatter(abs(x),y_0, c=aux_dict[metric]['nofair'], label='Nifty', s=90)
            axs[1,idx].scatter(abs(x),y_1, c=aux_dict[metric]['nofair'], label='Nifty', s=90)

        axs[0,idx].tick_params(labelsize=15)
        axs[0,idx].legend(fontsize=18, loc='upper center', bbox_to_anchor=(0, 1.3, 1, 0), ncols=2,
                          handletextpad=0, columnspacing=.7)
        axs[1,idx].tick_params(labelsize=15)
        #axs[1,idx].legend(fontsize=15)
        
    fig.tight_layout()


def plot_best_fair_wrt_util_perc_acc_roc(plot_dict,
                                         phase = 'test',
                                         file_path = '',
                                         ):
    aux_dict = {'DDP': {'fair': '#78c679', 'nofair': '#006837', 'lbl':'DDP'},
                'DEO+': {'fair': '#41b6c4', 'nofair': '#253494', 'lbl':'DEO$^+$'},
                'DEO-': {'fair': '#fecc5c', 'nofair': '#bd0026', 'lbl':'DEO$^-$'},
                'CF': {'fair': '#8c96c6', 'nofair': '#810f7c', 'lbl':'CF'},
                'demographicParity':'Diff. Demographic Parity',
                'equalOpportunityPos':'Diff. Equal Opportunity +',
                'equalOpportunityNeg':'Diff. Equal Opportunity -',
                'counterfactual':'Counterfactual Fairness'}
    percs = range(90,101,1)
    fig, axs = plt.subplots(len(plot_dict),2, figsize=(9,8), sharey='row', sharex='col')
    
    axs[-1,0].set_xlabel('% max ACC', size=20)
    axs[-1,1].set_xlabel('% max AUROC', size=20)

    for idx, (metric, setting_dict) in enumerate(plot_dict.items()):
        nofair_res = pickle.load(open(os.path.join(file_path, setting_dict['nofair_fn']), 'rb'))
        fair_res = pickle.load(open(os.path.join(file_path, setting_dict['fair_fn']), 'rb'))

        #axs[0,0].set_title(metric, size=25, pad=50)
        axs[idx,0].set_ylabel(aux_dict[metric]['lbl'], size=20)

        avg_y_0_f, avg_y_1_f, avg_y_0_nf, avg_y_1_nf = [], [], [], []
        std_y_0_f, std_y_1_f, std_y_0_nf, std_y_1_nf = [], [], [], []
        for p in percs:
            avg_y_0_f.append(fair_res[f"{p}_accuracy_min_{setting_dict['fair_metr']}"][f"mean_{phase}_{setting_dict['fair_metr']}"])
            std_y_0_f.append(fair_res[f"{p}_accuracy_min_{setting_dict['fair_metr']}"][f"std_{phase}_{setting_dict['fair_metr']}"])
            avg_y_1_f.append(fair_res[f"{p}_roc_min_{setting_dict['fair_metr']}"][f"mean_{phase}_{setting_dict['fair_metr']}"])
            std_y_1_f.append(fair_res[f"{p}_roc_min_{setting_dict['fair_metr']}"][f"std_{phase}_{setting_dict['fair_metr']}"])
            avg_y_0_nf.append(nofair_res[f"{p}_accuracy_min_{setting_dict['fair_metr']}"][f"mean_{phase}_{setting_dict['fair_metr']}"])
            std_y_0_nf.append(nofair_res[f"{p}_accuracy_min_{setting_dict['fair_metr']}"][f"std_{phase}_{setting_dict['fair_metr']}"])
            avg_y_1_nf.append(nofair_res[f"{p}_roc_min_{setting_dict['fair_metr']}"][f"mean_{phase}_{setting_dict['fair_metr']}"])
            std_y_1_nf.append(nofair_res[f"{p}_roc_min_{setting_dict['fair_metr']}"][f"std_{phase}_{setting_dict['fair_metr']}"])
        
        axs[idx,0].plot(percs, abs(np.array(avg_y_0_f)), '.-', c=aux_dict[metric]['fair'], label='Our')
        axs[idx,0].fill_between(percs,
                                abs(np.array(avg_y_0_f))-np.array(std_y_0_f),
                                abs(np.array(avg_y_0_f))+np.array(std_y_0_f),
                                color=aux_dict[metric]['fair'], alpha=.25, linewidth=2, linestyle='dotted')        
        axs[idx,1].plot(percs, abs(np.array(avg_y_1_f)), '.-', c=aux_dict[metric]['fair'], label='Our')
        axs[idx,1].fill_between(percs,
                                abs(np.array(avg_y_1_f))-np.array(std_y_1_f),
                                abs(np.array(avg_y_1_f))+np.array(std_y_1_f),
                                color=aux_dict[metric]['fair'], alpha=.25, linewidth=2, linestyle='dotted')
        axs[idx,0].plot(percs, abs(np.array(avg_y_0_nf)), '.-', c=aux_dict[metric]['nofair'], label='Nifty')
        axs[idx,0].fill_between(percs,
                                abs(np.array(avg_y_0_nf))-np.array(std_y_0_nf),
                                abs(np.array(avg_y_0_nf))+np.array(std_y_0_nf),
                                color=aux_dict[metric]['nofair'], alpha=.1, linewidth=2, linestyle='dotted')
        axs[idx,1].plot(percs, abs(np.array(avg_y_1_nf)), '.-', c=aux_dict[metric]['nofair'], label='Nifty')
        axs[idx,1].fill_between(percs,
                                abs(np.array(avg_y_1_nf))-np.array(std_y_1_nf),
                                abs(np.array(avg_y_1_nf))+np.array(std_y_1_nf),
                                color=aux_dict[metric]['nofair'], alpha=.1, linewidth=2, linestyle='dotted')
        
        axs[idx,1].legend(fontsize=16, loc='center right',
                          bbox_to_anchor=(1.5, .5, 0, 0), ncols=1, handletextpad=.5, columnspacing=.7,
                          )
        axs[idx,0].set_xticks(percs)
        axs[idx,1].set_xticks(percs)
        axs[idx,0].tick_params(labelsize=12)
        axs[idx,1].tick_params(labelsize=12)
        axs[idx,0].grid(alpha=.5)
        axs[idx,1].grid(alpha=.5)

    fig.tight_layout()
