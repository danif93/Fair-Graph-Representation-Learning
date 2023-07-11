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
from fair_graphs.models.graph_models import FairGCNAutoEncoder


def optim_fairAutoEncoder(num_splits, 
                          data, 
                          lambdas, 
                          learning_rate = 1e-3, 
                          num_epochs = 100, 
                          metric = 'dp',
                          pos = 0,
                          train_percentage = .7,
                          verbose = False,
                        ):
    scaler = MinMaxScaler(feature_range = (0, 1))
    mses = np.zeros((len(lambdas)))
    fair_losses = np.zeros((len(lambdas)))

    num_samples, num_feat = data.samples.shape
    indices = np.arange(num_samples)
    y = tr.stack((data.labels, data.sensitive)).T.cpu() #if include_sensitive else dataset.labels.cpu()

    trn_device = tr.device("cuda:0")

    for split_idx in tqdm(range(num_splits)):
        trn_idxs, _ = train_test_split(indices, train_size=train_percentage, stratify=y, random_state=split_idx)
        trn_data = deepcopy(data)
        trn_data.sample_data_from_indices(trn_idxs)
        trn_data.samples = scaler.fit_transform(trn_data.samples)
        trn_data = trn_data.to(trn_device)

        for idxl, lam in enumerate(lambdas):
            fae = FairGCNAutoEncoder(in_channels=num_feat, out_channels=num_feat)
            fae = fae.to(trn_device)
            mse, fair_loss = fae.fit(trn_data,
                                     num_epochs = num_epochs,
                                     learning_rate = learning_rate,
                                     fair_lambda = lam,
                                     metric = metric,
                                     pos = pos,
                                     verbose = verbose)
            mses[idxl] += mse
            fair_losses[idxl] += fair_loss

            str_ext = f'_split_idx_{split_idx}_lambda_{lam}_metric_{metric}'
            if metric != 'dp':
                str_ext += f'_pos{pos}'
            folder_path = os.path.join('data', 'preprocessed_features', f'{data}_feat')
            os.makedirs(folder_path, exist_ok=True)
            fae.save_state_dict(save_path=folder_path, name_extension=str_ext, device='cpu')

    mses /= num_splits
    fair_losses /= num_splits

    return mses, fair_losses


def main():
    os.chdir(os.path.join("..", ".."))

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_test_splits',
                        type = int,
                        default = 30,
                        help = '')
    args = parser.parse_args()

    datasets = [
        PokecData(sensitive_attribute='region', target_attribute='marital_status_indicator',
                  include_sensitive=True, num_samples=10000, pre_scale_features=False),
        FacebookData(sensitive_attribute='gender', target_attribute='egocircle',
                     include_sensitive=True, num_samples=0, pre_scale_features=False),
        GooglePlusData(sensitive_attribute='gender', target_attribute='egocircle',
                       include_sensitive=True, num_samples=0, pre_scale_features=False),
        GermanData(sensitive_attribute='Gender', target_attribute='GoodCustomer',
                   include_sensitive=True, num_samples=0, pre_scale_features=False),
        CreditData(sensitive_attribute='Age', target_attribute='NoDefaultNextMonth',
                   include_sensitive=True, num_samples=0, pre_scale_features=False),
        BailData(sensitive_attribute='WHITE', target_attribute='RECID',
                 include_sensitive=True, num_samples=0, pre_scale_features=False),
    ]

    lambdas = [1, 1e-1, 1e-2, 1e-3]
    
    for data in datasets:
        print(f"\nDataset {data}")

        optim_fairAutoEncoder(num_splits = args.num_test_splits,
                              data = data,
                              lambdas = lambdas,
                              metric = 'dp')
        optim_fairAutoEncoder(num_splits = args.num_test_splits,
                              data = data,
                              lambdas = lambdas,
                              metric = 'eo',
                              pos = 0)
        optim_fairAutoEncoder(num_splits = args.num_test_splits,
                              data = data,
                              lambdas = lambdas,
                              metric = 'eo',
                              pos = 1)
        

if __name__ == '__main__':
    main()
    