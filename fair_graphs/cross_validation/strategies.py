# ----- Standard Imports

# ----- Third-Party Imports
import pandas as pd

# ----- Library Imports


# --------------
# --- BASE CLASS
# --------------

class SelectionStrategy:
    def __call__(self, cv_results):
        pass


# ------------------------
# --- INSTANTIATED CLASSES
# ------------------------

class SimpleBestMetricStrategy(SelectionStrategy):
    def __init__(self, evaluation_metric, greater_is_better=True, phase='test', verbose=False):
        assert phase in ['train', 'validation', 'test']
        
        self.eval_m = evaluation_metric
        self.g_i_b = greater_is_better
        self.phase = phase
        self.verb = verbose
    
    def __call__(self, cv_results):
        # important: dataframe indexes will be equal to the positional index of the result matrix
        pd_cv_res = pd.Series(cv_results[f"mean_{self.phase}_{self.eval_m}"])
        
        if self.verb:
            print(f"Whole {self.eval_m} column:")
            print(pd_cv_res)
        
        if self.g_i_b:
            idx_best = pd_cv_res.idxmax()
        else:
            idx_best = pd_cv_res.idxmin()
            
        if self.verb:
            print(f"best {self.eval_m} index: {idx_best}")
            
        return idx_best
        
        
class BestDDPOnUtilityPercentile(SelectionStrategy):
    def __init__(self,
                 ddp_metric,
                 utility_metric,
                 max_accuracy_percentile,
                 is_ddp_negated = True,
                 phase = 'test',
                 verbose = False,
                ):
        if max_accuracy_percentile > 1:
            max_accuracy_percentile /= 100.
        assert 0 < max_accuracy_percentile <= 1
        assert phase in ['train', 'validation', 'test']
        
        self.ddp_m = ddp_metric
        self.util_m = utility_metric
        self.m_a_p = round(max_accuracy_percentile, 2)
        self.neg_ddp = is_ddp_negated
        self.phase = phase
        self.verb = verbose
    
    def __call__(self, cv_results):
        util_res = pd.Series(cv_results[f"mean_{self.phase}_{self.util_m}"])
        fair_res = pd.Series(cv_results[f"mean_{self.phase}_{self.ddp_m}"])
        
        if self.verb:
            print("\tUtility")
            print(util_res)
            print("\tFairness")
            print(fair_res)
        
        # select the entries with accuracy >= the given percentile
        max_util = util_res.max()
        util_tresh = max_util * self.m_a_p
        geq_mask = util_res >= util_tresh
        
        if self.verb:
            print(f"Params within {self.m_a_p}% of best utility [{util_tresh} out of {max_util}]")
            print(util_res[geq_mask])
        
        # among the surviving accuracies, select the entry with the best ddp metric
        # the ddp obtained through the sklearn scorer is negated, so the greater (close to 0) the better
        if self.neg_ddp:
            idx_best_ddp = fair_res[geq_mask].idxmax()
        else:
            idx_best_ddp = fair_res[geq_mask].idxmin()
            
        if self.verb:
            print(f"Best {self.ddp_m} index: {idx_best_ddp}")
            
        return idx_best_ddp
    