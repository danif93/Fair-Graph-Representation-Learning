# ----- Standard Imports
import sys
import os
import argparse
from copy import deepcopy

if __name__ == '__main__':
    sys.path.append(os.path.join('..','..'))

# ----- Third Party Imports
from tqdm.auto import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import torch as tr

# ----- Library Imports
from fair_graphs.datasets.graph_datasets import (GermanData, BailData, CreditData,
                                                 PokecData, FacebookData, GooglePlusData)
from fair_graphs.datasets.scalers import MinMaxScaler
from fair_graphs.models.graph_models import FairAutoEncoder


def optim_fairAutoEncoder(num_splits, 
                          num_samples, 
                          num_feat,
                          data, 
                          db_name,
                          lambdas, 
                          learning_rate = 1e-3, 
                          num_epochs = 500, 
                          verbose = False, 
                          metric = 'dp',
                          pos = 0,
                          train_percentage = .7,
                        ):
    scaler = MinMaxScaler(feature_range = (0, 1))
    mses = np.zeros((len(lambdas)))
    fair_losses = np.zeros((len(lambdas)))

    num_samples = len(data)
    indices = np.arange(num_samples)
    y = tr.stack((data.labels, data.sensitive)).T.cpu() #if include_sensitive else dataset.labels.cpu()

    trn_device = tr.device("cuda:1")

    for split_idx in tqdm(range(num_splits)):
        trn_idxs, _ = train_test_split(indices, train_size=train_percentage, stratify=y, random_state=split_idx)
        trn_data = deepcopy(data)
        trn_data.sample_data_from_indices(trn_idxs)
        trn_data.samples = scaler.fit_transform(trn_data.samples)
        trn_data = trn_data.to(trn_device)

        for idxl, lam in enumerate(lambdas):
            fae = FairAutoEncoder(num_feat, num_feat)
            mse, fair_loss = fae.fit(trn_data,
                                    learning_rate = learning_rate,
                                    num_epochs = num_epochs,
                                    verbose = verbose,
                                    lambda_ = lam,
                                    metric = metric,
                                    pos = pos,
                                    db = db_name,
                                    split_idx = split_idx)
            mses[idxl] += mse
            fair_losses[idxl] += fair_loss

    mses /= num_splits
    fair_losses /= num_splits

    return mses, fair_losses


def main():
    os.chdir(os.path.join("..", ".."))

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_test_splits',
                        type = int,
                        default = 50,
                        help = '')
    args = parser.parse_args()

    datasets = {
        #'pokec_z': PokecData(sensitive_attribute='region', target_attribute='marital_status_indicator',
        #                     include_sensitive=True, num_samples=1000, pre_scale_features=False, region_suffix='z'),
        #'facebook': FacebookData(sensitive_attribute='gender', target_attribute='egocircle',
        #                         include_sensitive=True, num_samples=1000, pre_scale_features=False),
        'gplus': GooglePlusData(sensitive_attribute='gender', target_attribute='egocircle',
                                include_sensitive=True, num_samples=1000, pre_scale_features=False),
        #'german': GermanData(sensitive_attribute='Gender', target_attribute='GoodCustomer',
        #                     include_sensitive=True, num_samples=1000, pre_scale_features=False),
        #'credit': CreditData(sensitive_attribute='Age', target_attribute='NoDefaultNextMonth',
        #                     include_sensitive=True, num_samples=1000, pre_scale_features=False),
        #'bail': BailData(sensitive_attribute='WHITE', target_attribute='RECID',
        #                 include_sensitive=True, num_samples=1000, pre_scale_features=False),
    }

    lambdas = [1, 1e-1, 1e-2, 1e-3]
    
    for name, data in datasets.items():
        print(f"\nDataset {data}")
        num_samples, num_feat = data.samples.shape

        optim_fairAutoEncoder(num_splits = args.num_test_splits,
                              num_samples = num_samples,
                              num_feat = num_feat,
                              data = data,
                              db_name = name,
                              lambdas = lambdas,
                              metric = 'dp')
        optim_fairAutoEncoder(num_splits = args.num_test_splits,
                              num_samples = num_samples,
                              num_feat = num_feat,
                              data = data,
                              db_name = name,
                              lambdas = lambdas,
                              metric = 'eo',
                              pos = 0)
        optim_fairAutoEncoder(num_splits = args.num_test_splits,
                              num_samples = num_samples,
                              num_feat = num_feat,
                              data = data,
                              db_name = name,
                              lambdas = lambdas,
                              metric = 'eo',
                              pos = 1)
        

if __name__ == '__main__':
    main()
    