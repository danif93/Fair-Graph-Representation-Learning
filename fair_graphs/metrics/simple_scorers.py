# ----- Standard Imports

# ----- Third-Party Imports
import numpy as np

# ----- Library Imports


# --------------
# --- Base Class
# --------------

class GraphScorer:
    def __init__(self,
                 need_class_predictions: bool = False,
                ):
        self.need_classes = need_class_predictions
        
    def __call__(self, output_dictionary):
        pass

    @staticmethod
    def retrieve_data_mask(labels,
                           sensitives,
                           eval_class = None,
                           sens_group = None,
                          ):
        test_mask = np.full(len(labels), True)
        if eval_class is not None:
            test_mask &= labels == eval_class
        if sens_group is not None:
            test_mask &= sensitives == sens_group
        return test_mask


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
                ):
        super().__init__(need_class_predictions)
        self.eval_metric = evaluation_metric
        self.eval_class = evaluation_class
        self.sens_group = sensitive_group
        self.inverse_metric = inverse_metric
    
    def __call__(self, output_dictionary):
        msk = GraphScorer.retrieve_data_mask(labels = output_dictionary['labels'],
                                             sensitives = output_dictionary['sensitives'],
                                             eval_class = self.eval_class,
                                             sens_group = self.sens_group)
        labels = output_dictionary['labels'][msk]
        outputs = output_dictionary['outs'][msk]
        predictions = output_dictionary['preds'][msk]

        try:
            if self.need_classes:
                score = self.eval_metric(labels, predictions)
            else:
                score = self.eval_metric(labels, outputs)
        except ValueError: # roc-auc may rise ValueError if both classes are not present (some dataset are highly biased)
            score = np.nan

        if self.inverse_metric:
            score = -score
            
        return round(score, 3)
    
    
class DDPMetricScorer(GraphScorer):
    def __init__(self,
                 evaluation_metric = None,
                 sensitive_group = None,
                 evaluation_class = None,
                 need_class_predictions: bool = False,
                ):
        super().__init__(need_class_predictions)
        self.eval_metric = evaluation_metric
        self.sens_group = sensitive_group
        self.eval_class = evaluation_class
    
    def __call__(self, output_dictionary):
        msk = GraphScorer.retrieve_data_mask(labels = output_dictionary['labels'],
                                             sensitives = output_dictionary['sensitives'],
                                             eval_class = self.eval_class)
        labels = output_dictionary['labels'][msk]
        sensitives = output_dictionary['sensitives'][msk]
        outputs = output_dictionary['outs'][msk]
        predictions = output_dictionary['preds'][msk]
        
        # ----- Prepare sensitive group masks
        if self.sens_group is None:
            unique_groups = np.unique(sensitives)
            assert len(unique_groups) <= 2, \
                "Since a sensitive population is not specified, the sensitive vector must be binary."
            chosen_sens = unique_groups[0]
        else:
            chosen_sens = self.sens_group
        
        evl_group_mask = sensitives == chosen_sens
        oth_group_mask = ~evl_group_mask
        
        if evl_group_mask.all() or oth_group_mask.all():
            # raise RuntimeError("Either all or none of the samples belong to the evaluated group")
            return np.nan
        
        try:
            if self.need_classes:
                if self.eval_metric is not None:
                    score_evl_group = self.eval_metric(labels[evl_group_mask], predictions[evl_group_mask])
                    score_oth_group = self.eval_metric(labels[oth_group_mask], predictions[oth_group_mask])
                else:
                    score_evl_group = predictions[evl_group_mask].mean()
                    score_oth_group = predictions[oth_group_mask].mean()
            else:
                if self.eval_metric is not None:
                    score_evl_group = self.eval_metric(labels[evl_group_mask], outputs[evl_group_mask])
                    score_oth_group = self.eval_metric(labels[oth_group_mask], outputs[oth_group_mask])
                else:
                    score_evl_group = outputs[evl_group_mask].mean()
                    score_oth_group = outputs[oth_group_mask].mean()
            score = -np.abs(score_evl_group - score_oth_group)
        except ValueError: # roc-auc may rise ValueError if both classes are not present (some dataset are highly biased)
            score = np.nan

        return round(score, 3)


class CounterfactualScorer(GraphScorer):
    def __init__(self,
                sensitive_group = None,
                evaluation_class = None,
                need_class_predictions: bool = False,
            ):
        super().__init__(need_class_predictions)
        self.sens_group = sensitive_group
        self.eval_class = evaluation_class

    def __call__(self, output_dictionary):
        msk = GraphScorer.retrieve_data_mask(labels = output_dictionary['labels'],
                                             sensitives = output_dictionary['sensitives'],
                                             eval_class = self.eval_class,
                                             sens_group = self.sens_group)
        outputs = output_dictionary['outs'][msk]
        predictions = output_dictionary['preds'][msk]
        count_outputs = output_dictionary['count_outs'][msk]
        count_predictions = output_dictionary['count_preds'][msk]
        
        if self.need_classes:
            score = -np.abs(predictions - count_predictions).mean()
        else:
            score = -np.abs(outputs - count_outputs).mean()        
        return round(score, 3)