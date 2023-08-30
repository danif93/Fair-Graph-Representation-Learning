"""Defines project's pytorch graph datasets."""

# ----- Standard Imports
import os
import pickle

# ----- Third Party Imports
import numpy as np
import torch as tr
from torch.utils import data

# ----- Library Imports
from fair_graphs.datasets.scalers import MinMaxScaler


# -------------------------------
# --- Pytorch Dataset Base Module
# -------------------------------

class _GraphDataset(data.Dataset):
    def __init__(self,
                 full_file_path: str,
                 name: str,
                 *,
                 include_sensitive: bool = True,
                 num_samples: int = 0,
                 pre_scale_features: bool = False,
                ):
        # ----- Read and initialize variables from file and tranform to tensors
        data_dict = pickle.load(open(full_file_path, 'rb'))
        self.name = name
        
        self.samples = tr.tensor(data_dict['features']).float()
        
        self.include_sensitive = include_sensitive
        self.sensitive = tr.tensor(data_dict['sensitive'])
        
        self.labels = tr.tensor(data_dict['labels'])
        self.labels_msk = tr.arange(len(self.labels))
        
        self.adj_mtx = data_dict['adjacency_matrix']
        
        # ----- Optionally include sensitive attribute within features
        if include_sensitive:
            self.samples = tr.column_stack((self.samples, self.sensitive))
            
        # ----- Optionally reduce the number of samples
        if 0 < num_samples <= len(self.samples):
            rng = np.random.default_rng(42)
            rand_idxs = rng.choice(len(self.samples), size=num_samples, replace=False)
            self.sample_data_from_indices(rand_idxs)
        
        # ----- Optionaly (0-1)-scale features
        if pre_scale_features:
            scaler = MinMaxScaler(feature_range = (0, 1))
            self.samples = scaler.fit_transform(self.samples)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        return (self.samples[index], self.sensitive[index],
                self.labels[index], self.adj_mtx[index])
    
    def __str__(self):
        return self.name
    
    def sample_data_from_indices(self, indices):
        self.samples = self.samples[indices]
        self.sensitive = self.sensitive[indices]
        self.labels = self.labels[indices]
        self.labels_msk = tr.arange(len(self.labels))
        self.adj_mtx = self.adj_mtx[indices][:,indices]
        return self
    
    def set_labels_mask(self, indices):
        if indices.dtype == tr.bool:
            self.labels_msk = tr.where(indices)[0]
        else:
            assert indices.dtype == tr.int64
            self.labels_msk = indices

    def to(self, destination):
        self.samples = self.samples.to(destination)
        self.sensitive = self.sensitive.to(destination)
        self.labels = self.labels.to(destination)
        return self
    
    
# ----------------------------
# --- Project Datasets Classes
# ----------------------------        
        
class BailData(_GraphDataset):
    def __init__(self,
                 sensitive_attribute,
                 target_attribute,
                 *,
                 include_sensitive = True,
                 num_samples = 0,
                 pre_scale_features = False,
                ):
        file_name = f'bail_sensitive_{sensitive_attribute}_label_{target_attribute}.pickle'
        super().__init__(os.path.join('data', 'bail', file_name),
                         name = 'bail',
                         include_sensitive = include_sensitive,
                         num_samples = num_samples,
                         pre_scale_features = pre_scale_features)      


class CreditData(_GraphDataset):
    def __init__(self,
                 sensitive_attribute,
                 target_attribute,
                 *,
                 include_sensitive = True,
                 num_samples = 0,
                 pre_scale_features = False,
                ):
        file_name = f'credit_sensitive_{sensitive_attribute}_label_{target_attribute}.pickle'
        super().__init__(os.path.join('data', 'credit', file_name),
                         name = 'credit',
                         include_sensitive = include_sensitive,
                         num_samples = num_samples,
                         pre_scale_features = pre_scale_features)
        

class GermanData(_GraphDataset):
    def __init__(self,
                 sensitive_attribute,
                 target_attribute,
                 *,
                 include_sensitive = True,
                 num_samples = 0,
                 pre_scale_features = False,
                ):
        file_name = f'german_sensitive_{sensitive_attribute}_label_{target_attribute}.pickle'
        super().__init__(os.path.join('data', 'german', file_name),
                         name = 'german',
                         include_sensitive = include_sensitive,
                         num_samples = num_samples,
                         pre_scale_features = pre_scale_features)
        
        
class PokecData(_GraphDataset):
    def __init__(self,
                 sensitive_attribute,
                 target_attribute,
                 *,
                 include_sensitive = True,
                 num_samples = 0,
                 pre_scale_features = False,
                ):
        file_name = f'pokec_sensitive_{sensitive_attribute}_label_{target_attribute}.pickle'
        super().__init__(os.path.join('data', 'pokec', file_name),
                         name = 'pokec',
                         include_sensitive = include_sensitive,
                         num_samples = num_samples,
                         pre_scale_features = pre_scale_features)


class FacebookData(_GraphDataset):
    def __init__(self,
                 sensitive_attribute,
                 target_attribute,
                 *,
                 include_sensitive = True,
                 num_samples = 0,
                 pre_scale_features = False,
                ):
        file_name = f'facebook_sensitive_{sensitive_attribute}_label_{target_attribute}.pickle'
        super().__init__(os.path.join('data', 'facebook', file_name),
                         name = 'facebook',
                         include_sensitive = include_sensitive,
                         num_samples = num_samples,
                         pre_scale_features = pre_scale_features)


class GooglePlusData(_GraphDataset):
    def __init__(self,
                 sensitive_attribute,
                 target_attribute,
                 *,
                 include_sensitive = True,
                 num_samples = 0,
                 pre_scale_features = False,
                ):
        file_name = f'gplus_sensitive_{sensitive_attribute}_label_{target_attribute}.pickle'
        super().__init__(os.path.join('data', 'gplus', file_name),
                         name = 'gplus',
                         include_sensitive = include_sensitive,
                         num_samples = num_samples,
                         pre_scale_features = pre_scale_features)
