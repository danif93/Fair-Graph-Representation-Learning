# ----- Standard Imports
from collections import defaultdict
import itertools

# ----- Third-Party Imports
import numpy as np

# ----- Library Imports
from fair_graphs.datasets.graph_datasets import _GraphDataset
    

def get_all_arguments(data: _GraphDataset):
    encoder_args = {'in_channels': data.samples.shape[1],
                    'out_channels': data.samples.shape[1],
                    'base_model': 'gcn'}
    model_fixed_args = {'num_hidden': data.samples.shape[1],
                        'num_projection_hidden': data.samples.shape[1],
                        'num_class': 1,
                        'load_dict': {'save_path': 'data/model_init',
                                      'name_extension': f'init_{data.name}'}}
    model_cv_args = {'highest_homo_perc': [-1, .2, .5, .8,],
                     'drop_criteria': [None, 0, 1],
                     'edge_drop_rate': [.2, .5, .8],
                     'feat_drop_rate': [.2, .5, .8],
                     'sim_lambda': [.2, .5, .8]}
    fit_funct_args = {'num_epochs': 100,
                      'learning_rate': 1e-3,
                      'weight_decay': 1e-5,
                      'verbose': False}
    return encoder_args,model_fixed_args,model_cv_args,fit_funct_args


def average_train_test_splits(gsr_setting_results, exclude_nan=True):
    result_dict = defaultdict(list)
    for gsr_test_split in gsr_setting_results:
        for gsr_k, gsr_v in gsr_test_split.items():
            if gsr_k[:4] in ['mean','std_'] and (not np.isnan(gsr_v).any() or not exclude_nan):
                result_dict[gsr_k].append(gsr_v)

    for res_k, res_v in result_dict.items():
        # assert that all the not-nan values have the same sign: 
        # positive for greater-is-better metrics and negative for the lower-is-better ones
        assert all([v>=0 for split_v in res_v for v in split_v if not np.isnan(v)]) or \
            all([v<=0 for split_v in res_v for v in split_v if not np.isnan(v)])
        result_dict[res_k] = np.abs(np.nanmean(res_v, axis=0))
    
    # every split has been tested on the same hyperparameters configuration,
    # extract it from the last split
    result_dict['params'] = gsr_test_split['params']

    return result_dict


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))