"""Defines project's pytorch graph neural networks."""

# ----- Standard Imports
import os
from copy import deepcopy as dc

# ----- Third Party Imports
import torch as tr
from torch import optim
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits
from torch_geometric.nn import GCNConv
from tqdm.auto import tqdm

# ----- Library Imports
from fair_graphs.training.training_utils import (get_string_device_from_index,
                                                 dropout_edge, drop_feature)
from fair_graphs.training.distr_distances import FirstOrderMatching
from fair_graphs.metrics.metrics import cosine_distance
from fair_graphs.datasets.graph_datasets import _GraphDataset
from fair_graphs.datasets.datasets_utils import edges_coo_from_adj_matrix


# ------------------------------
# --- Preprocessing Auto-Encoder
# ------------------------------

class FairAutoEncoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 base_model: str = 'gcn',
                ):
        super().__init__()
        
        self.base_model = base_model        
        self.encoder = GCNConv(in_channels, out_channels)
        self.decoder = nn.Linear(out_channels, in_channels) 
        self.relu = nn.ReLU()
        
        self.fair_loss = FirstOrderMatching('mean')     
        
        self.name = '_fairAutoEncoder'
    
    def encode(self, x, edge_index):
        encoder = self.relu(self.encoder(x, edge_index))
        return encoder
    
    def decode(self, x):
        decoder = self.decoder(x)
        return decoder
    
    def forward(self, x, edge_index):
        x = self.encode(x, edge_index)
        x = self.decode(x)
        return x
    
    def save_state_dict(self, save_path='', name_extension='', device=None):
        path = os.path.join( os.getcwd(), save_path)
        if save_path != '':
            os.makedirs(path, exist_ok=True)
        
        fullname = os.path.join(path, f"{self.name}")
        if name_extension != '':
            fullname += f"_{name_extension}"
        fullname += '.pth'
        
        current_device = next(self.parameters()).device
        if device is not None and device != current_device:
            if isinstance(device, int):
                device = get_string_device_from_index(device)
            self.to(device)
            tr.save(self.state_dict(), fullname)
            self.to(current_device)
        else:
            tr.save(self.state_dict(), fullname)
    
    def load_encoder_state_dict(self, fullname=''):
        old_encoder = tr.load(fullname)
        old_encoder = old_encoder.get('encoder.lin.weight')
        self.encoder.lin.weight = dc(nn.Parameter(old_encoder))
        return self
    
    def fit(self,
            trainset: _GraphDataset,
            *,
            num_epochs: int = 100,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-5,
            lambda_: float = 1e-3,
            verbose: bool = False,
            metric: str = 'dp', 
            pos: int = 0,
            db: str = 'German',
            store_str_ext: str = '',
            split_idx: int = 0
           ):
        
        trn_device = next(self.parameters()).device
        trn_edges = trainset.adj_mtx
        trn_edges = edges_coo_from_adj_matrix(trn_edges).to(trn_device)

        trn_features = trainset.samples.float().to(trn_device)
        trn_labels = trainset.labels.float().to(trn_device)
        trn_sensitive = trainset.sensitive.float().to(trn_device)
        
        criterion = nn.MSELoss()
        optim = tr.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.train()
        ep_range = range(1, num_epochs+1)
                
        if metric == 'dp':
            pos_mask_attr0 = (trn_sensitive == 0)
            pos_mask_attr1 = (trn_sensitive == 1)
        else:
            pos_mask_attr0 = (trn_labels == pos) & (trn_sensitive == 0)
            pos_mask_attr1 = (trn_labels == pos) & (trn_sensitive == 1)
        
        for ep in (tqdm(ep_range) if verbose else ep_range):
            optim.zero_grad(set_to_none=True)
            
            emb = self.encode(trn_features, trn_edges)
            pred_features = self.decode(emb)
            
            emb_sensitive_attr0 = emb[pos_mask_attr0]
            emb_sensitive_attr1 = emb[pos_mask_attr1]
            
            mse = criterion(pred_features, trn_features)
            fair_loss = self.fair_loss.forward(emb_sensitive_attr0, emb_sensitive_attr1)
            
            loss = mse + lambda_ * fair_loss
            loss.backward()
            
            optim.step()
            
            if ep % 10 == 0 and verbose:
                print(f'Epoch {ep} -> Loss {loss.item()}')
        
        if metric == 'dp':
            str_ext = f'{store_str_ext}_split_idx_{split_idx}_lambda_{lambda_}_metric_{metric}'
        else:
            str_ext = f'{store_str_ext}_split_idx_{split_idx}_lambda_{lambda_}_metric_{metric}_pos{pos}'
            
        self.save_state_dict(save_path = db+'_feat', name_extension = str_ext)

        return mse, fair_loss


# ----------------
# --- GNN Base Modules
# ----------------

class Encoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 base_model: str = 'gcn',
                ):
        super().__init__()
        
        self.base_model = base_model
        if self.base_model == 'gcn':
            self.conv = GCNConv(in_channels, out_channels)
        else:
            raise ValueError(f"'base_model'={base_model} is not available.")

    def forward(self, x: tr.Tensor, edge_index: tr.Tensor):
        x = self.conv(x, edge_index)
        return x

    
class _GraphNeuralNetwork(nn.Module):
    def __init__(self,
                 encoder: Encoder,
                 classifier: nn.Module,
                 model_name: str,
                ):
        super().__init__()
        
        self.encoder = encoder
        self.classifier = classifier
        self.name = f"{model_name}_{self.encoder.base_model}"
    
    def encode(self, x, edge_index):
        emb = self.encoder(x, edge_index)
        return emb
    
    def classify(self, x):
        cl = self.classifier(x)
        return cl
    
    def forward(self, x, edge_index):
        x = self.encode(x, edge_index)
        x = self.classify(x)
        return x
    
    def save_state_dict(self, save_path='', name_extension='', device=None):
        if save_path != '':
            os.makedirs(save_path, exist_ok=True)
        
        fullname = os.path.join(save_path, f"{self.name}")
        if name_extension != '':
            fullname += f"_{name_extension}"
        fullname += '.pth'
        
        current_device = next(self.parameters()).device
        if device is not None and device != current_device:
            if isinstance(device, int):
                device = get_string_device_from_index(device)
            self.to(device)
            tr.save(self.state_dict(), fullname)
            self.to(current_device)
        else:
            tr.save(self.state_dict(), fullname)

    def load_state_dict(self, name_extension='', save_path='', device=None):
        fullname = os.path.join(save_path, f"{self.name}")
        if name_extension != '':
            fullname += f"_{name_extension}"
        fullname += '.pth'
        if isinstance(device, int):
            device = f"cuda:{device}" if device != -1 else "cpu"
        super().load_state_dict(tr.load(fullname, map_location=device))
        return self

    
# ---------------------------------
# --- GNN Instantiated Full Modules
# ---------------------------------

class SSF(_GraphNeuralNetwork):
    def __init__(self,
                 encoder: Encoder,
                 num_hidden: int,
                 num_projection_hidden: int,
                 num_class: int,
                 *,
                 drop_criteria: int = None, # None: drop from all or one of the classes
                 edge_drop_rate: float = .3,
                 feat_drop_rate: float = .3,
                 sim_lambda: float = 0.,
                 highest_homo_perc: float = -1,
                 load_dict: dict = None,
                ):
        super().__init__(encoder = encoder,
                         classifier = nn.Linear(num_hidden, num_class),
                         model_name = 'ssf')
    
        # Projection
        self.projector = nn.Sequential(
            nn.Linear(num_hidden, num_projection_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(num_projection_hidden, num_hidden),
        )

        # Prediction
        self.predictor = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
        )
        
        self.dr_crit = drop_criteria
        self.edge_drop_rate = edge_drop_rate
        self.feat_drop_rate = feat_drop_rate
        self.sim_lambda = sim_lambda
        self.highest_homo_perc = highest_homo_perc
        
        if load_dict is not None:
            self.load_state_dict(**load_dict)
        else:
            self.apply(SSF._weights_init)
    
    @staticmethod
    def _weights_init(mod):
        if isinstance(mod, nn.Linear):
            nn.init.xavier_uniform_(mod.weight)
            if mod.bias is not None:
                nn.init.constant_(mod.bias, val = 0.)
                    
    def get_store_extension(self):
        ext = f'edr_{self.edge_drop_rate:.2f}'
        ext += f'_fdr_{self.feat_drop_rate:.2f}'
        ext += f'_sl_{self.sim_lambda:.2f}'
        if self.highest_homo_perc != -1:
            ext += f'_hhm_{self.highest_homo_perc:.2f}'
        return ext
    
    def project(self, x):
        proj = self.projector(x)
        return proj
    
    def predict(self, x):
        pred = self.predictor(x)
        return pred
    
    def fit(self,
            trainset: _GraphDataset,
            *,
            num_epochs: int = 100,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-5,
            verbose: bool = False,
           ):
        # ----- Argument validation
        # TODO
        assert 0 < self.edge_drop_rate < 1
        assert 0 < self.feat_drop_rate < 1
        assert 0 <= self.sim_lambda <= 1
        assert self.highest_homo_perc == -1 or (0 <= self.highest_homo_perc <= 1)
                
        # ----- Retrieve the edges to be dropped
        if self.dr_crit is not None:
            dr_msk = trainset.labels == self.dr_crit
            if not dr_msk.any():
                raise RuntimeError('There are no edges with the selected class that can be dropped')
        else:
            dr_msk = tr.ones_like(trainset.labels)
        
        # drop edges wich start OR ends with nodes from a selected class
        dr_idxs = dr_msk.nonzero().squeeze(-1)
        str_node_drop, end_node_drop = [], []
        str_node_keep, end_node_keep = [], []

        for str_node, end_node in zip(*trainset.adj_mtx.nonzero()):
            if str_node in dr_idxs or end_node in dr_idxs:
                str_node_drop.append(str_node)
                end_node_drop.append(end_node)
            else:
                str_node_keep.append(str_node)
                end_node_keep.append(end_node)
        to_drop = tr.tensor((str_node_drop, end_node_drop), dtype=int, device=trainset.sensitive.device)
        to_keep = tr.tensor((str_node_keep, end_node_keep), dtype=int, device=trainset.sensitive.device)
        
        # ----- Prepare data for training
        trn_labels = trainset.labels.float().unsqueeze(1)

        # ----- Set-up optimizer
        full_optim = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        self.train()
        ep_range = range(1, num_epochs+1)
        for ep in (tqdm(ep_range) if verbose else ep_range):
            full_optim.zero_grad(set_to_none=True)
                        
            # ----- Perform [fair] edges drop
            drop_edges_1 = dropout_edge(to_drop,
                                        sensitive_vect = trainset.sensitive,
                                        base_drop_rate = self.edge_drop_rate,
                                        highest_homo_perc = self.highest_homo_perc)
            drop_edges_1 = tr.cat((to_keep, drop_edges_1), axis=1)
            
            drop_edges_2 = dropout_edge(to_drop,
                                        sensitive_vect = trainset.sensitive,
                                        base_drop_rate = self.edge_drop_rate,
                                        highest_homo_perc = self.highest_homo_perc)
            drop_edges_2 = tr.cat((to_keep, drop_edges_2), axis=1)
            
            # ----- Perform features drop eventually including counterfactuals (on sensitive)
            drop_feat_stand = drop_feature(data_matrix = trainset.samples,
                                           features_drop_rate = self.feat_drop_rate,
                                           sens_feat_idx = -1 if trainset.include_sensitive else None,
                                           correlated_attrs = None,
                                           correlated_weights = None,
                                           flip_sensitive = False)
            drop_feat_count = drop_feature(data_matrix = trainset.samples,
                                           features_drop_rate = self.feat_drop_rate,
                                           sens_feat_idx = -1 if trainset.include_sensitive else None,
                                           correlated_attrs = None,
                                           correlated_weights = None,
                                           flip_sensitive = True)
            
            # ----- Counterfactual similarity loss
            embedd_stand = self.encode(drop_feat_stand, drop_edges_1)
            proj_stand = self.project(embedd_stand)
            pred_stand = self.predict(proj_stand)

            embedd_count = self.encode(drop_feat_count, drop_edges_2)
            proj_count = self.project(embedd_count)
            pred_count = self.predict(proj_count)

            # STOPGRAD on the second argument following nifty implementation
            cosine_stand = cosine_distance(pred_stand, proj_count.detach()) 
            cosine_count = cosine_distance(pred_count, proj_stand.detach())
            cosine_loss = (cosine_stand + cosine_count)/2
            
            # ----- Data target loss
            class_stand = self.classify(embedd_stand)
            class_count = self.classify(embedd_count)
            
            bce_stand = binary_cross_entropy_with_logits(class_stand[trainset.labels_msk], trn_labels[trainset.labels_msk])
            bce_count = binary_cross_entropy_with_logits(class_count[trainset.labels_msk], trn_labels[trainset.labels_msk])
            
            class_loss = (bce_stand + bce_count)/2
                        
            # ----- Combine losses and optimization step
            loss = (1 - self.sim_lambda)*class_loss + self.sim_lambda*cosine_loss
            loss.backward()
            full_optim.step()

    def compute_augm_predictions(self,
                                 dataset: _GraphDataset,
                                 n_augm: int,
                                 *,
                                 return_classes: bool = False,
                                 counterfactual: bool = False):
        assert isinstance(n_augm, int) and n_augm >= 1
        augm_preds = tr.empty((n_augm, len(dataset.labels)), device=dataset.samples.device)

        if counterfactual:
            assert dataset.include_sensitive
            samples = dataset.samples
            samples[:, -1] = 1 - samples[:, -1]
            sensitives = 1 - dataset.sensitive
        else:
            samples = dataset.samples
            sensitives = dataset.sensitive

        for augm_idx in range(n_augm):
            # ----- Drop edges
            edges = edges_coo_from_adj_matrix(dataset.adj_mtx).to(samples.device)
            augm_edges = dropout_edge(edges,
                                      sensitive_vect = sensitives,
                                      base_drop_rate = self.edge_drop_rate,
                                      highest_homo_perc = self.highest_homo_perc)
            # ----- Retrieve predictions
            y_pred = self(samples, augm_edges).squeeze(-1)
            if return_classes:
                y_pred = tr.where(tr.sigmoid(y_pred) < .5, 0,1)
            augm_preds[augm_idx] = y_pred
            
            # ----- Aggregate results
            if return_classes:
                y_pred = augm_preds.mode(dim=0).values
            else:
                y_pred = augm_preds.float().mean(dim=0)
                
        return y_pred


# -----------------
# --- Miscellaneous
# -----------------

MODELS_STRING_MAP = {
    'ssf': SSF,
}