"""Defines project's pytorch graph neural networks."""

# ----- Standard Imports
import os

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


# --------------------
# --- Encoders Modules
# --------------------

class _BaseEncoder(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 name: str,
                 ):
        super().__init__()
        self.encoder = encoder
        self.name = name
    
    def encode(self,
               x: tr.Tensor,
               edge_index: tr.Tensor,
               ):
        x = self.encoder(x, edge_index)
        return x

    def forward(self,
                x: tr.Tensor,
                edge_index: tr.Tensor,
                ):
        x = self.encode(x, edge_index)
        return x
    
    def save_state_dict(self, save_path='', name_extension='', device=None):
        if save_path != '':
            os.makedirs(save_path, exist_ok=True)
        
        fullname = os.path.join(save_path, self.name)
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

    def load_state_dict(self, save_path='', name_extension='', device=None):
        fullname = os.path.join(save_path, self.name)
        if name_extension != '':
            fullname += f"_{name_extension}"
        fullname += '.pth'
        if isinstance(device, int):
            device = get_string_device_from_index(device)
        super().load_state_dict(tr.load(fullname, map_location=device))
        return self


class GCNEncoder(_BaseEncoder):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 ):
        super().__init__(encoder = GCNConv(in_channels, out_channels),
                         name = 'gcnEncoder')
    

class FairGCNAutoEncoder(_BaseEncoder):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 ):
        encoder = nn.Sequential(
            GCNConv(in_channels, out_channels),
            nn.ReLU()
            )
        super().__init__(encoder = encoder,
                         name = 'fairGCNAutoEncoder')
        self.decoder = nn.Linear(out_channels, in_channels)

    def encode(self,
               x: tr.Tensor,
               edge_index: tr.Tensor,
               ):
        x = self.encoder[0](x, edge_index)
        x = self.encoder[1](x)
        return x
    
    def decode(self,
               x: tr.Tensor,
               ):
        x = self.decoder(x)
        return x
    
    def fit(self,
            trainset: _GraphDataset,
            *,
            num_epochs: int = 100,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-5,
            fair_lambda: float = 1e-3,
            metric: str = 'dp', 
            pos: int = 0,
            verbose: bool = False,
           ):
        
        trn_device = next(self.parameters()).device
        trn_edges = edges_coo_from_adj_matrix(trainset.adj_mtx).to(trn_device)
        
        util_loss = nn.MSELoss()
        fair_loss = FirstOrderMatching('mean')     

        optim = tr.optim.Adam(self.parameters(), lr=learning_rate,
                              weight_decay=weight_decay)
        self.train()
                
        if metric == 'dp':
            pop0_mask = trainset.sensitive == 0
            pop1_mask = ~pop0_mask
        else:
            pop0_mask = (trainset.labels == pos) & (trainset.sensitive == 0)
            pop1_mask = (trainset.labels == pos) & (trainset.sensitive == 1)
        
        for ep in (tqdm(range(num_epochs)) if verbose else range(num_epochs)):
            optim.zero_grad(set_to_none=True)
            
            embedd = self.encode(trainset.samples, trn_edges)
            pred_features = self.decode(embedd)
            
            embedd_pop0 = embedd[pop0_mask]
            embedd_pop1 = embedd[pop1_mask]
            
            u_l = util_loss(pred_features, trainset.samples)
            f_l = fair_loss(embedd_pop0, embedd_pop1)
            
            loss = u_l + fair_lambda*f_l
            loss.backward()
            optim.step()
            
            if ep % 10 == 0 and verbose:
                print(f'Epoch {ep} -> Loss {loss.item()}')
                print(f"\t util:{u_l}; fair: {f_l}")

        return u_l, f_l


# ---------------
# --- GNN Modules
# ---------------
    
class _GraphNeuralNetwork(nn.Module):
    def __init__(self,
                 encoder: _BaseEncoder,
                 classifier: nn.Module,
                 model_name: str,
                 ):
        super().__init__()
        
        self.encoder = encoder
        self.classifier = classifier
        self.name = f"{model_name}_{self.encoder.name}"
    
    def encode(self, x, edge_index):
        x = self.encoder(x, edge_index)
        return x
    
    def classify(self, x):
        x = self.classifier(x)
        return x
    
    def forward(self, x, edge_index):
        x = self.encode(x, edge_index)
        x = self.classify(x)
        return x
    
    def save_state_dict(self, save_path='', name_extension='', device=None):
        if save_path != '':
            os.makedirs(save_path, exist_ok=True)
        
        fullname = os.path.join(save_path, self.name)
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
        fullname = os.path.join(save_path, self.name)
        if name_extension != '':
            fullname += f"_{name_extension}"
        fullname += '.pth'
        if isinstance(device, int):
            device = get_string_device_from_index(device)
        super().load_state_dict(tr.load(fullname, map_location=device))
        return self


class SSF(_GraphNeuralNetwork):
    def __init__(self,
                 encoder: _BaseEncoder,
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
        self.predictor = nn.Linear(num_hidden, num_hidden)
        
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
                nn.init.constant_(mod.bias, val=0.)
    
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
        assert isinstance(trainset, _GraphDataset)
        assert isinstance(num_epochs, int) and num_epochs > 0
        assert learning_rate > 0
        assert weight_decay >= 0
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
        ep_range = range(num_epochs)
        for _ in (tqdm(ep_range) if verbose else ep_range):
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
    
    def get_store_extension(self):
        ext = f'edr_{self.edge_drop_rate:.2f}'
        ext += f'_fdr_{self.feat_drop_rate:.2f}'
        ext += f'_sl_{self.sim_lambda:.2f}'
        if self.highest_homo_perc != -1:
            ext += f'_hhm_{self.highest_homo_perc:.2f}'
        return ext


# -----------------
# --- Miscellaneous
# -----------------

MODELS_STRING_MAP = {
    'ssf': SSF,
}