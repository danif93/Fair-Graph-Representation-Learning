"""Contains methods and variables related to the project's cross-validated training process."""

# ----- Standard Imports
import os

# ----- Third Party Imports
import numpy as np
import torch as tr

# ----- Library Imports
from fair_graphs.datasets.graph_datasets import _GraphDataset
from fair_graphs.datasets.datasets_utils import train_test_split_graph_data
from fair_graphs.datasets.scalers import MinMaxScaler
from fair_graphs.models.graph_models import _GraphNeuralNetwork


def optimize_gnn(graph_dataset: _GraphDataset,
                 graph_net: _GraphNeuralNetwork,
                 evaluation_metrics: dict,
                 *,
                 test_size = .25,
                 rnd_seed = 42,
                 training_epochs = 10,
                 device_idx = -1,
                 learning_rate = 1e-3,
                 weight_decay = 1e-5,
                 store_model = False,
                ):
    # ----- Arguments validation
    # TOCHECK
    assert isinstance(graph_dataset, _GraphDataset)
    assert isinstance(graph_net, _GraphNeuralNetwork)
    assert isinstance(evaluation_metrics, dict)
    assert 0 < test_size < 1
    assert isinstance(rnd_seed, int)
    assert isinstance(training_epochs, int) and training_epochs >= 1
    assert isinstance(device_idx, int)
    assert isinstance(store_model, bool)
    
    # ----- Fixing random seeds
    #np.random.seed(rnd_seed)
    #tr.manual_seed(rnd_seed)
    #if device_idx > -1:
    #    tr.cuda.manual_seed(rnd_seed)
        
    # ----- Train/test split
    trn_dataset, tst_dataset = train_test_split_graph_data(graph_dataset,
                                                           test_size = test_size,
                                                           random_state = rnd_seed)
    scaler = MinMaxScaler(feature_range = (0, 1))
    trn_dataset.samples = scaler.fit_transform(trn_dataset.samples)
    tst_dataset.samples = scaler.transform(tst_dataset.samples)
    
    # ----- Move to selected device
    trn_device = tr.device('cpu' if not tr.cuda.is_available() or device_idx == -1 else f'cuda:{device_idx}')
    graph_net = graph_net.to(trn_device)
    trn_dataset = trn_dataset.to(trn_device)
    
    # ----- Train model
    graph_net.fit(trainset = trn_dataset,
                  num_epochs = training_epochs,
                  learning_rate = learning_rate,
                  weight_decay = weight_decay,
                  verbose = True)
    
    # ----- Model evaluation
    tst_dataset = tst_dataset.to(trn_device)
    
    res_dict = {}
    for metric_name, scorer in evaluation_metrics.items():
        res_dict[f"train_{metric_name}"] = scorer(graph_net, trn_dataset)
        res_dict[f"test_{metric_name}"] = scorer(graph_net, tst_dataset)
    
    # ---- Model storing
    if store_model:
        ext = f"{graph_net.get_store_extension()}_ep_{training_epochs}_data_{graph_dataset}"
        graph_net.save_state_dict(save_path = os.path.join('trained_models', 'simple_train'),
                                  name_extension = ext)
    
    return res_dict
