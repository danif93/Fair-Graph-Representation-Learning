# ----- Standard Imports
from collections import defaultdict
import itertools

# ----- Third-Party Imports
import numpy as np

# ----- Library Imports
from fair_graphs.datasets.graph_datasets import _GraphDataset
    

def get_all_cv_arguments(data: _GraphDataset):
    encoder_args = {'in_channels': data.samples.shape[1],
                    'out_channels': data.samples.shape[1],
                    'base_model': 'gcn'}
    model_fixed_args = {'num_hidden': data.samples.shape[1],
                        'num_projection_hidden': data.samples.shape[1],
                        'num_class': 1,
                        'load_dict': {'save_path': 'data/model_init',
                                      'name_extension': f"init_{data.name}{'_sensitiveFalse' if not data.include_sensitive else ''}"}}
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


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))