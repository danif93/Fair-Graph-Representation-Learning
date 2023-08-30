"""Contains methods and variables related to the project's cross-validation pipeline."""

# ----- Standard Imports
import os
import pickle
from copy import deepcopy
from collections import defaultdict

# ----- Third Party Imports
import numpy as np
import torch as tr
import pandas as pd
from tqdm.auto import tqdm

# ----- Library Imports
from fair_graphs.models.graph_models import SSF, GCNEncoder, FairGCNAutoEncoder
from fair_graphs.datasets.graph_datasets import _GraphDataset
from fair_graphs.datasets.scalers import MinMaxScaler
from fair_graphs.cross_validation.cv_utils import product_dict, get_all_cv_arguments


# ---------------------------
# --- Cross-validation Method
# ---------------------------

def cross_validation(data: _GraphDataset,
                     cv_rounds: int,
                     evaluation_scorers: dict,
                     *,
                     scenario: str = 'inductive',
                     activate_fae: bool = False,
                     f_lmbd: str = '1'):
    # ----- argument checking
    assert isinstance(data, _GraphDataset)
    assert isinstance(cv_rounds, int) and cv_rounds > 0
    assert isinstance(evaluation_scorers, dict)# and all([])
    assert scenario in ['inductive','transductive','semiinductive']
    assert isinstance(activate_fae, bool)

    # ----- initialize model fixed, cv, and fit function arguments
    encoder_args, model_fixed_args, model_cv_args, fit_funct_args = get_all_cv_arguments(data)
    hyperparams_list = list(product_dict(**model_cv_args))
    
    # ----- data management
    num_samples = len(data)
    tr_end_idx = int(num_samples * .7)
    vl_end_idx = tr_end_idx + (num_samples - tr_end_idx)//2
    data_scaler = MinMaxScaler(feature_range=(0,1))
    tr_device = tr.device('cuda:0')
    
    # ----- results management
    complete_cv_results = []
    simple_cv_results = pd.DataFrame()

    # ----- cycle across hyperparameters
    for hyp_idx, hyperparams_sett in enumerate(hyperparams_list):
        print(f"running hyperpar {hyp_idx+1}/{len(hyperparams_list)}")

        if hyperparams_sett['highest_homo_perc'] == -1 and hyperparams_sett['drop_criteria'] is not None:
            continue

        complete_cv_results.append(defaultdict(list))

        # ----- cycle across cv rounds
        for round_idx in tqdm(range(cv_rounds)):
            # ----- retrieve data splits
            perm_idxs = np.random.default_rng(round_idx).permutation(num_samples)
            tr_idxs, vl_idxs, ts_idxs = \
                perm_idxs[:tr_end_idx], perm_idxs[tr_end_idx:vl_end_idx], perm_idxs[vl_end_idx:]
            
            # ----- prepare training data
            if scenario == 'transductive':
                tr_data = deepcopy(data)
                tr_data.set_labels_mask(tr.tensor(tr_idxs, dtype=tr.int64))
            else: # scenario in ['inductive','semiinductive']:
                tr_data = deepcopy(data).sample_data_from_indices(tr_idxs)
            tr_data.samples = data_scaler.fit_transform(tr_data.samples)
            tr_data = tr_data.to(tr_device)

            # ----- initialize model (and optionally load fair autoencoder)
            enc = GCNEncoder(**encoder_args)
            net = SSF(encoder=enc, **model_fixed_args, **hyperparams_sett)

            if activate_fae and hyperparams_sett['highest_homo_perc'] != -1:
                load_path = os.path.join("data", "preprocessed_features", f'{data}_feat')
                name_ext = 'dp' if hyperparams_sett['drop_criteria'] is None else f"eo_pos{hyperparams_sett['drop_criteria']}"
                name_ext = f"_lambda_{f_lmbd}_metric_{name_ext}"
                fair_enc = FairGCNAutoEncoder(**encoder_args)
                fair_enc.load_state_dict(save_path=load_path, name_extension=name_ext, device='cpu')
                net.encoder = deepcopy(fair_enc)
            
            # ----- fit model
            net = net.to(tr_device)
            net.fit(tr_data, **fit_funct_args)
            
            # ----- compute performances
            with tr.no_grad():
                if scenario == 'inductive':
                    tr_dict = {'outs': net.compute_augm_predictions(tr_data, 30, return_classes=False, counterfactual=False).cpu().numpy(),
                               'preds': net.compute_augm_predictions(tr_data, 30, return_classes=True, counterfactual=False).cpu().numpy(),
                               'count_outs': net.compute_augm_predictions(tr_data, 30, return_classes=False, counterfactual=True).cpu().numpy(),
                               'count_preds': net.compute_augm_predictions(tr_data, 30, return_classes=True, counterfactual=True).cpu().numpy(),
                               'labels': tr_data.labels.cpu().numpy(),
                               'sensitives': tr_data.sensitive.cpu().numpy()}
                    del tr_data
                    
                    vl_data = deepcopy(data).sample_data_from_indices(vl_idxs)
                    vl_data.samples = data_scaler.transform(vl_data.samples)
                    vl_data = vl_data.to(tr_device)
                    vl_dict = {'outs': net.compute_augm_predictions(vl_data, 30, return_classes=False, counterfactual=False).cpu().numpy(),
                               'preds': net.compute_augm_predictions(vl_data, 30, return_classes=True, counterfactual=False).cpu().numpy(),
                               'count_outs': net.compute_augm_predictions(vl_data, 30, return_classes=False, counterfactual=True).cpu().numpy(),
                               'count_preds': net.compute_augm_predictions(vl_data, 30, return_classes=True, counterfactual=True).cpu().numpy(),
                               'labels': vl_data.labels.cpu().numpy(),
                               'sensitives': vl_data.sensitive.cpu().numpy()}
                    del vl_data

                    ts_data = deepcopy(data).sample_data_from_indices(ts_idxs)
                    ts_data.samples = data_scaler.transform(ts_data.samples)
                    ts_data = ts_data.to(tr_device)
                    ts_dict = {'outs': net.compute_augm_predictions(ts_data, 30, return_classes=False, counterfactual=False).cpu().numpy(),
                               'preds': net.compute_augm_predictions(ts_data, 30, return_classes=True, counterfactual=False).cpu().numpy(),
                               'count_outs': net.compute_augm_predictions(ts_data, 30, return_classes=False, counterfactual=True).cpu().numpy(),
                               'count_preds': net.compute_augm_predictions(ts_data, 30, return_classes=True, counterfactual=True).cpu().numpy(),
                               'labels': ts_data.labels.cpu().numpy(),
                               'sensitives': ts_data.sensitive.cpu().numpy()}
                    del ts_data

                elif scenario == 'transductive':
                    tr_dict = {'outs': net.compute_augm_predictions(tr_data, 30, return_classes=False, counterfactual=False).cpu().numpy()[tr_idxs],
                               'preds': net.compute_augm_predictions(tr_data, 30, return_classes=True, counterfactual=False).cpu().numpy()[tr_idxs],
                               'count_outs': net.compute_augm_predictions(tr_data, 30, return_classes=False, counterfactual=True).cpu().numpy()[tr_idxs],
                               'count_preds': net.compute_augm_predictions(tr_data, 30, return_classes=True, counterfactual=True).cpu().numpy()[tr_idxs],
                               'labels': tr_data.labels.cpu().numpy()[tr_idxs],
                               'sensitives': tr_data.sensitive.cpu().numpy()[tr_idxs]}
                    vl_dict = {'outs': net.compute_augm_predictions(tr_data, 30, return_classes=False, counterfactual=False).cpu().numpy()[vl_idxs],
                               'preds': net.compute_augm_predictions(tr_data, 30, return_classes=True, counterfactual=False).cpu().numpy()[vl_idxs],
                               'count_outs': net.compute_augm_predictions(tr_data, 30, return_classes=False, counterfactual=True).cpu().numpy()[vl_idxs],
                               'count_preds': net.compute_augm_predictions(tr_data, 30, return_classes=True, counterfactual=True).cpu().numpy()[vl_idxs],
                               'labels': tr_data.labels.cpu().numpy()[vl_idxs],
                               'sensitives': tr_data.sensitive.cpu().numpy()[vl_idxs]}
                    ts_dict = {'outs': net.compute_augm_predictions(tr_data, 30, return_classes=False, counterfactual=False).cpu().numpy()[ts_idxs],
                               'preds': net.compute_augm_predictions(tr_data, 30, return_classes=True, counterfactual=False).cpu().numpy()[ts_idxs],
                               'count_outs': net.compute_augm_predictions(tr_data, 30, return_classes=False, counterfactual=True).cpu().numpy()[ts_idxs],
                               'count_preds': net.compute_augm_predictions(tr_data, 30, return_classes=True, counterfactual=True).cpu().numpy()[ts_idxs],
                               'labels': tr_data.labels.cpu().numpy()[ts_idxs],
                               'sensitives': tr_data.sensitive.cpu().numpy()[ts_idxs]}
                    
                else: # scenario == 'semiinductive'
                    tr_dict = {'outs': net.compute_augm_predictions(tr_data, 30, return_classes=False, counterfactual=False).cpu().numpy(),
                               'preds': net.compute_augm_predictions(tr_data, 30, return_classes=True, counterfactual=False).cpu().numpy(),
                               'count_outs': net.compute_augm_predictions(tr_data, 30, return_classes=False, counterfactual=True).cpu().numpy(),
                               'count_preds': net.compute_augm_predictions(tr_data, 30, return_classes=True, counterfactual=True).cpu().numpy(),
                               'labels': tr_data.labels.cpu().numpy(),
                               'sensitives': tr_data.sensitive.cpu().numpy()}
                    #del tr_data
                                        
                    vl_dict = {'outs': [], 'preds': [], 'count_outs': [], 'count_preds': [], 'labels': [], 'sensitives': []}
                    for vl_idx in vl_idxs:
                        indexes = np.append(tr_idxs, vl_idx)
                        test_data = deepcopy(data).sample_data_from_indices(indexes)
                        test_data.samples = data_scaler.transform(test_data.samples)
                        test_data = test_data.to(tr_device)
                        vl_dict['outs'].append(net.compute_augm_predictions(test_data, 30, return_classes=False, counterfactual=False)[-1].cpu().numpy())
                        vl_dict['preds'].append(net.compute_augm_predictions(test_data, 30, return_classes=True, counterfactual=False)[-1].cpu().numpy())
                        vl_dict['count_outs'].append(net.compute_augm_predictions(test_data, 30, return_classes=False, counterfactual=True)[-1].cpu().numpy())
                        vl_dict['count_preds'].append(net.compute_augm_predictions(test_data, 30, return_classes=True, counterfactual=True)[-1].cpu().numpy())
                        vl_dict['labels'].append(test_data.labels[-1].cpu().numpy())
                        vl_dict['sensitives'].append(test_data.sensitive[-1].cpu().numpy())
                    vl_dict = {k: np.array(v) for k,v in vl_dict.items()}
                    
                    ts_dict = {'outs': [], 'preds': [], 'count_outs': [], 'count_preds': [], 'labels': [], 'sensitives': []}
                    for ts_idx in ts_idxs:
                        indexes = np.append(tr_idxs, ts_idx)
                        test_data = deepcopy(data).sample_data_from_indices(indexes)
                        test_data.samples = data_scaler.transform(test_data.samples)
                        test_data = test_data.to(tr_device)
                        ts_dict['outs'].append(net.compute_augm_predictions(test_data, 30, return_classes=False, counterfactual=False)[-1].cpu().numpy())
                        ts_dict['preds'].append(net.compute_augm_predictions(test_data, 30, return_classes=True, counterfactual=False)[-1].cpu().numpy())
                        ts_dict['count_outs'].append(net.compute_augm_predictions(test_data, 30, return_classes=False, counterfactual=True)[-1].cpu().numpy())
                        ts_dict['count_preds'].append(net.compute_augm_predictions(test_data, 30, return_classes=True, counterfactual=True)[-1].cpu().numpy())
                        ts_dict['labels'].append(test_data.labels[-1].cpu().numpy())
                        ts_dict['sensitives'].append(test_data.sensitive[-1].cpu().numpy())
                    ts_dict = {k: np.array(v) for k,v in ts_dict.items()}

                    #del test_data
                    
            # ----- evaluate scorers
            for score_name, scorer in evaluation_scorers.items():
                tr_res = scorer(tr_dict)
                complete_cv_results[-1][f'complete_train_{score_name}'].append(tr_res)
                vl_res = scorer(vl_dict)
                complete_cv_results[-1][f'complete_validation_{score_name}'].append(vl_res)
                ts_res = scorer(ts_dict)
                complete_cv_results[-1][f'complete_test_{score_name}'].append(ts_res)
        
        # ----- average hyperparam results across cv rounds
        complete_cv_results[-1] = dict(complete_cv_results[-1])
        complete_cv_results[-1]['params'] = hyperparams_sett
        simple_cv_results.loc[hyp_idx, 'params'] = str(hyperparams_sett)
        for score_name in evaluation_scorers.keys():
            for split_name in ['train','validation','test']:
                complete_cv_results[-1][f'mean_{split_name}_{score_name}'] = \
                    np.nanmean(complete_cv_results[-1][f'complete_{split_name}_{score_name}'])
                complete_cv_results[-1][f'std_{split_name}_{score_name}'] = \
                    np.nanstd(complete_cv_results[-1][f'complete_{split_name}_{score_name}'])
                simple_cv_results.loc[hyp_idx, f'mean_{split_name}_{score_name}'] = \
                    complete_cv_results[-1][f'mean_{split_name}_{score_name}']
                simple_cv_results.loc[hyp_idx, f'std_{split_name}_{score_name}'] = \
                    complete_cv_results[-1][f'std_{split_name}_{score_name}']

    # ----- store results
    file_path = os.path.join('results', scenario, "fd_fae" if activate_fae else "fd")
    if not data.include_sensitive:
        file_path += '_sensitiveFalse'
    os.makedirs(file_path, exist_ok=True)
    pickle.dump(complete_cv_results, open(os.path.join(file_path, f'{data}_complete.pickle'), 'wb'))
    simple_cv_results.to_pickle(os.path.join(file_path, f'{data}_simple.pickle'))
