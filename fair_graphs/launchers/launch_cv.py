"""Entry point for the cross-validation pipeline"""

# ----- Standard Imports
import sys
import os
import warnings
import argparse

if __name__ == '__main__':
    sys.path.append(os.path.join('..','..'))

# ----- Third Party Imports
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

# ----- Library Imports
from fair_graphs.datasets.graph_datasets import (GermanData, BailData, CreditData,
                                                 PokecData, FacebookData, GooglePlusData)
from fair_graphs.metrics.scorers import SubgroupsMetricScorer, DDPMetricScorer, CounterfactualScorer
from fair_graphs.cross_validation.method import cross_validation
from fair_graphs.cross_validation.method_inverted import cross_validation as inv_cross_validation


def main():
    os.chdir(os.path.join("..", ".."))

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_test_splits',
                        type = int,
                        default = 30,
                        help = '')
    parser.add_argument('--fair_autoencoder',
                        action = 'store_true',
                        help = '')
    args = parser.parse_args()

    eval_scorers = {
        # utility scorers
        'roc': SubgroupsMetricScorer(roc_auc_score),
        "rocSens0": SubgroupsMetricScorer(roc_auc_score, sensitive_group=0),
        "rocSens1": SubgroupsMetricScorer(roc_auc_score, sensitive_group=1),
        "accuracy": SubgroupsMetricScorer(accuracy_score, need_class_predictions=True),
        "accuracySens0": SubgroupsMetricScorer(accuracy_score, sensitive_group=0, need_class_predictions=True),
        "accuracySens1": SubgroupsMetricScorer(accuracy_score, sensitive_group=1, need_class_predictions=True),
        "accuracyPos": SubgroupsMetricScorer(accuracy_score, evaluation_class=1, need_class_predictions=True),
        "accuracyNeg": SubgroupsMetricScorer(accuracy_score, evaluation_class=0, need_class_predictions=True),

        # fairness scorers
        'demographicParity': DDPMetricScorer(),
        'demographicParityPreds': DDPMetricScorer(need_class_predictions=True),
        'equalOpportunityPos': DDPMetricScorer(evaluation_class=1),
        'equalOpportunityPosPreds': DDPMetricScorer(evaluation_class=1, need_class_predictions=True),
        'equalOpportunityNeg': DDPMetricScorer(evaluation_class=0),
        'equalOpportunityNegPreds': DDPMetricScorer(evaluation_class=0, need_class_predictions=True),
        "demographicParityRoc": DDPMetricScorer(roc_auc_score),
        "demographicParityAccuracy": DDPMetricScorer(accuracy_score, need_class_predictions=True),
        "equalOpportunityPosAccuracy": DDPMetricScorer(accuracy_score, evaluation_class=1, need_class_predictions=True),
        "equalOpportunityNegAccuracy": DDPMetricScorer(accuracy_score, evaluation_class=0, need_class_predictions=True),
        "counterfactual": CounterfactualScorer(),
        "counterfactualPreds": CounterfactualScorer(need_class_predictions=True),
        "counterfactualPos": CounterfactualScorer(evaluation_class=1),
        "counterfactualPosPreds": CounterfactualScorer(evaluation_class=1, need_class_predictions=True),
        "counterfactualNeg": CounterfactualScorer(evaluation_class=0),
        "counterfactualNegPreds": CounterfactualScorer(evaluation_class=0, need_class_predictions=True),
    }

    datasets = [
        PokecData(sensitive_attribute='region', target_attribute='marital_status_indicator',
                  include_sensitive=True, num_samples=1000, pre_scale_features=False),
        FacebookData(sensitive_attribute='gender', target_attribute='egocircle',
                     include_sensitive=True, num_samples=1000, pre_scale_features=False),
        GooglePlusData(sensitive_attribute='gender', target_attribute='egocircle',
                       include_sensitive=True, num_samples=1000, pre_scale_features=False),
        GermanData(sensitive_attribute='Gender', target_attribute='GoodCustomer',
                   include_sensitive=True, num_samples=1000, pre_scale_features=False),
        CreditData(sensitive_attribute='Age', target_attribute='NoDefaultNextMonth',
                   include_sensitive=True, num_samples=1000, pre_scale_features=False),
        BailData(sensitive_attribute='WHITE', target_attribute='RECID',
                 include_sensitive=True, num_samples=1000, pre_scale_features=False),
        ]
    
    for data in datasets:
        for sett in ['inductive', 'transductive', 'semiinductive']:
            print(f"\nDataset {data} on scenario {sett} {'with' if args.fair_autoencoder else 'without'} fairautoencoder")
            if sett == 'semiinductive':
                inv_cross_validation(data,
                                     args.num_test_splits,
                                     eval_scorers,
                                     scenario = sett,
                                     activate_fae = args.fair_autoencoder,
                                     f_lmbd = 1)
            else:
                cross_validation(data,
                                 args.num_test_splits,
                                 eval_scorers,
                                 scenario = sett,
                                 activate_fae = args.fair_autoencoder,
                                 f_lmbd = 1)
            

if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    main()
    