# ----- Standard Imports

# ----- Third-Party Imports
import numpy as np
import torch as tr

# ----- Library Imports
from fair_graphs.datasets.graph_datasets import _GraphDataset
#from fair_graphs.models.graph_models import _GraphNeuralNetwork # circular import
from fair_graphs.training.training_utils import dropout_edge
from fair_graphs.datasets.datasets_utils import edges_coo_from_adj_matrix


# --------------
# --- Base Class
# --------------

class GraphScorer:
    def __init__(self,
                 need_class_predictions: bool = False,
                 num_augmented: int = 30,
                ):
        self.need_classes = need_class_predictions
        self.n_augm = num_augmented
        
    def __call__(self,
                 estimator,#: _GraphNeuralNetwork,
                 datasets: _GraphDataset,
                ):
        pass
    
    @staticmethod
    def retrieve_data_mask(dataset: _GraphDataset,
                           eval_class = None,
                           sens_group = None,
                          ):
        test_mask = tr.full((dataset.samples.shape[0],), True, device=dataset.samples.device)
        if eval_class is not None:
            test_mask &= dataset.labels == eval_class
        if sens_group is not None:
            test_mask &= dataset.sensitive == sens_group
        return test_mask
    
    def compute_augmented_preds(self,
                                graph_model,
                                samples,
                                sensitive,
                                edges,
                                num_labels,
                               ):
        assert isinstance(self.n_augm, int) and self.n_augm >= 1
        
        with tr.no_grad():
            # ----- Predictions collector
            augm_preds = tr.empty((self.n_augm, num_labels), device=samples.device)
            
            # ----- Cycle across augmented rounds
            for augm_idx in range(self.n_augm):
                # ----- Drop edges
                augm_edges = dropout_edge(edges,
                                          sensitive_vect = sensitive,
                                          base_drop_rate = graph_model.edge_drop_rate,
                                          highest_homo_perc = graph_model.highest_homo_perc)
                # ----- Retrieve predictions
                y_pred = graph_model(samples, augm_edges).squeeze(-1)
                if self.need_classes:
                    y_pred = tr.where(tr.sigmoid(y_pred) < .5, 0,1)
                augm_preds[augm_idx] = y_pred
            
            # ----- Aggregate results
            if self.need_classes:
                y_pred = augm_preds.mode(dim=0).values
            else:
                y_pred = augm_preds.float().mean(dim=0)
                
        return y_pred


# ------------------------    
# --- Instantiated Classes
# ------------------------

class SubgroupsMetricScorer(GraphScorer):
    def __init__(self,
                 evaluation_metric,
                 evaluation_class = None,
                 sensitive_group = None,
                 need_class_predictions: bool = False,
                 inverse_metric: bool = False,
                 num_augmented: int = 30,
                ):
        super().__init__(need_class_predictions, num_augmented)
        self.eval_metric = evaluation_metric
        self.eval_class = evaluation_class
        self.sens_group = sensitive_group
        self.inverse_metric = inverse_metric
        
    def __str__(self):
        self_str = f"SubgroupsMetricScorer_{self.eval_metric}"
        if self.need_classes:
            self_str += "_onPreds"
        if self.sens_group is not None:
            self_str += f"_sensGroup:{self.sens_group}"
        if self.sens_group is not None:
            self_str += f"_evalClass:{self.eval_class}"
        return self_str
    
    def __call__(self, estimator, dataset):        
        # ----- Retrieve the test samples
        test_idx = GraphScorer.retrieve_data_mask(dataset,
                                                  eval_class=self.eval_class,
                                                  sens_group=self.sens_group)
        test_idx = test_idx.nonzero().squeeze(-1).cpu()
        
        if len(test_idx) == 0:
            # print(f"class: {self.eval_class}, samples: {(y == self.eval_class).sum()}")
            # print(f"group: {self.sens_group}, samples: {(s == self.sens_group).sum()}")
            # raise RuntimeError(f"No samples with sensitive {self.sens_group} and class {self.eval_class} have been found in the batch")
            return np.nan    
                
        test_samples = dataset.samples[test_idx]
        test_labels = dataset.labels[test_idx]
        test_sensitive = dataset.sensitive[test_idx]
        test_edges = dataset.adj_mtx[test_idx][:, test_idx]
        test_edges = edges_coo_from_adj_matrix(test_edges).to(test_samples.device)
        
        # ----- Compute predictions' average or mode over the augmented graphs
        y_pred = self.compute_augmented_preds(graph_model = estimator,
                                              samples = test_samples,
                                              sensitive = test_sensitive,
                                              edges = test_edges,
                                              num_labels = test_labels.shape[0])
                                              
        score = self.eval_metric(y_pred, test_labels).item()

        if self.inverse_metric:
            score = -score
        
        return round(score, 3)
    
    
class DDPMetricScorer(GraphScorer):
    def __init__(self,
                 evaluation_metric = None,
                 sensitive_group = None,
                 evaluation_class = None,
                 need_class_predictions: bool = False,
                 num_augmented: int = 30,
                ):
        super().__init__(need_class_predictions, num_augmented)
        self.eval_metric = evaluation_metric
        self.sens_group = sensitive_group
        self.eval_class = evaluation_class
    
    def __str__(self):
        self_str = f"DDPMetricScorer"
        if self.eval_metric is not None:
            self_str += f"_evalMetric:{self.eval_metric}"
        if self.need_classes:
            self_str += "_onPreds"
        if self.sens_group is not None:
            self_str += f"_sensGroup:{self.sens_group}"
        if self.sens_group is not None:
            self_str += f"_evalClass:{self.eval_class}"
        return self_str
    
    def __call__(self, estimator, dataset):
        # ----- Retrieve the test samples
        test_idx = GraphScorer.retrieve_data_mask(dataset, eval_class=self.eval_class)     
        test_idx = test_idx.nonzero().squeeze(-1).cpu()
        
        test_samples = dataset.samples[test_idx]
        test_sensitive = dataset.sensitive[test_idx]
        test_labels = dataset.labels[test_idx]
        test_edges = dataset.adj_mtx[test_idx][:,test_idx]
        test_edges = edges_coo_from_adj_matrix(test_edges).to(test_samples.device)
        
        # ----- Prepare sensitive group masks
        if self.sens_group is None:
            unique_groups = tr.unique(test_sensitive)
            assert len(unique_groups) <= 2, \
                "Since a sensitive population is not specified, the sensitive vector must be binary."
            chosen_sens = unique_groups[0]
        else:
            chosen_sens = self.sens_group
        
        evl_group_mask = test_sensitive == chosen_sens
        oth_group_mask = ~evl_group_mask
        
        if (not evl_group_mask.any()) or evl_group_mask.all():
            # raise RuntimeError("Either all or none of the samples belong to the evaluated group")
            return np.nan
        
        # ----- Compute predictions' average or mode over the augmented graphs
        y_pred = self.compute_augmented_preds(graph_model = estimator,
                                              samples = test_samples,
                                              sensitive = test_sensitive,
                                              edges = test_edges,
                                              num_labels = test_labels.shape[0])
                
        if self.eval_metric is not None:
            score_evl_group = self.eval_metric(y_pred[evl_group_mask], test_labels[evl_group_mask])
            score_oth_group = self.eval_metric(y_pred[oth_group_mask], test_labels[oth_group_mask])
        else:
            score_evl_group = y_pred[evl_group_mask].mean()
            score_oth_group = y_pred[oth_group_mask].mean()

        score = tr.abs(score_evl_group - score_oth_group).item()
        score = -score
        
        return round(score, 3)


class CounterfactualScorer(GraphScorer):
    def __init__(self,
                sensitive_group = None,
                evaluation_class = None,
                need_class_predictions: bool = False,
                num_augmented: int = 30,
            ):
        super().__init__(need_class_predictions, num_augmented)
        self.sens_group = sensitive_group
        self.eval_class = evaluation_class

    def __str__(self):
        self_str = f"CounterfactualScorer"
        if self.need_classes:
            self_str += "_onPreds"
        if self.sens_group is not None:
            self_str += f"_sensGroup:{self.sens_group}"
        if self.sens_group is not None:
            self_str += f"_evalClass:{self.eval_class}"
        return self_str

    def __call__(self, estimator, dataset):
        assert all([sv in [0,1] for sv in dataset.sensitive]) #TOFIX: assumes sensitive in [0,1]
        assert all([sv in [0,1] for sv in dataset.samples[:,-1]]) #TOFIX: assumes sensitive in [0,1]

        # ----- Retrieve the test samples
        test_idx = GraphScorer.retrieve_data_mask(dataset,
                                                  eval_class=self.eval_class,
                                                  sens_group=self.sens_group)
        test_idx = test_idx.nonzero().squeeze(-1).cpu()
        
        if len(test_idx) == 0:
            # print(f"class: {self.eval_class}, samples: {(y == self.eval_class).sum()}")
            # print(f"group: {self.sens_group}, samples: {(s == self.sens_group).sum()}")
            # raise RuntimeError(f"No samples with sensitive {self.sens_group} and class {self.eval_class} have been found in the batch")
            return np.nan
                
        test_samples = dataset.samples[test_idx]
        test_labels = dataset.labels[test_idx]
        test_sensitive = dataset.sensitive[test_idx]
        test_edges = dataset.adj_mtx[test_idx][:, test_idx]
        test_edges = edges_coo_from_adj_matrix(test_edges).to(test_samples.device)

        # ----- Compute predictions' average or mode over the augmented graphs
        y_pred_real = self.compute_augmented_preds(graph_model = estimator,
                                                   samples = test_samples,
                                                   sensitive = test_sensitive,
                                                   edges = test_edges,
                                                   num_labels = test_labels.shape[0])

        count_samples = tr.clone(test_samples)
        count_samples[:,-1] = 1 - count_samples[:,-1] #TOFIX: assumes sensitive in [0,1]
        y_pred_count = self.compute_augmented_preds(graph_model = estimator,
                                                    samples = count_samples,
                                                    sensitive = 1 - test_sensitive, #TOFIX: assumes sensitive in [0,1]
                                                    edges = test_edges,
                                                    num_labels = test_labels.shape[0])
                
        score = tr.abs(y_pred_real - y_pred_count).mean().item()
        score = -score
        
        return round(score, 3)