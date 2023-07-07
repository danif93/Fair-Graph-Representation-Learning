# ----- Standard Imports
import pickle
import os
from collections import defaultdict

# ----- Third-Party Imports
import pandas as pd

# ----- Library Imports
from fair_graphs.cross_validation.strategies import BestDDPOnUtilityPercentile, SimpleBestMetricStrategy


def build_multiindex(data, path):
    import warnings
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

    df = pd.read_pickle(os.path.join(path,f'{data}_simple.pickle'))

    new_df = pd.DataFrame()
    metr_cols = df.columns[1:]
    for idx, row in df.iterrows():
        hyps = row['params'][1:-1].split(',')
        hyp_dict = {}
        for h in hyps:
            h = h.split(':')
            hyp_dict[h[0].split("'")[1]] = float(h[1][1:]) if h[1][1:]!='None' else ''
        new_df.loc[idx, pd.MultiIndex.from_product((['params'], hyp_dict.keys()))] = hyp_dict.values()
        for sc, vl in row[metr_cols].items():
            if 'valid' in sc:
                spl = sc.split('_')
                sc = f"{spl[0]}_validation_{spl[2]}"
            new_df.loc[idx, sc] = vl

    new_df.to_pickle(os.path.join(path,f'{data}_simple_multi.pickle'))


def split_nifty_from_ours(data, path="results/"):
    df = pd.read_pickle(os.path.join(path, f'{data}_simple_multi.pickle'))

    nifty = (df['params']['highest_homo_perc'] == -1) & (df['params']['drop_criteria'] == '')
    df.loc[nifty].to_pickle(os.path.join(path, f'{data}_simple_multi_nifty.pickle'))
    ours = (df['params']['highest_homo_perc'] != -1) & (df['params']['highest_homo_perc'] != 1)
    df.loc[ours].to_pickle(os.path.join(path, f'{data}_simple_multi_ours.pickle'))

    ours_dp = ours & (df['params']['drop_criteria'] == '')
    df.loc[ours_dp].to_pickle(os.path.join(path, f'{data}_simple_multi_ours_dp.pickle'))
    ours_eop = ours & (df['params']['drop_criteria'] == 1)
    df.loc[ours_eop].to_pickle(os.path.join(path, f'{data}_simple_multi_ours_eop.pickle'))
    ours_eon = ours & (df['params']['drop_criteria'] == 0)
    df.loc[ours_eon].to_pickle(os.path.join(path, f'{data}_simple_multi_ours_eon.pickle'))


def retrieve_best_res_from_hyperparams_df(
        selection_strats,
        evaluation_scorers,
        file_name: str,
        selection_phase: str = 'validation',
        file_path: str = "results/",
        verbose_selection: bool = False,
        ):
    df_res = pd.read_pickle(os.path.join(file_path, file_name))

    best_results = defaultdict(dict)
    # ----- Cycle across the hyp-params selection strategies
    for sel_strat in selection_strats:
        # ----- Initialize strategy class
        if '_min_' in sel_strat:
            # e.g.: '95_accuracy_min_demographicParity'
            spl = sel_strat.split('_')
            percent = float(spl[0])
            utility_fun = spl[1]
            fairness_fun = spl[3]
            strategy = BestDDPOnUtilityPercentile(ddp_metric = fairness_fun,
                                                  utility_metric = utility_fun,
                                                  max_accuracy_percentile = percent,
                                                  is_ddp_negated = True,
                                                  phase = selection_phase,
                                                  verbose = verbose_selection)
        else:
            strategy = SimpleBestMetricStrategy(evaluation_metric = sel_strat,
                                                greater_is_better = True,
                                                phase = selection_phase,
                                                verbose = verbose_selection)
            
        # ----- Retrieve best cv results according to strategy
        best_idx = strategy(df_res)
        # ----- Retrieve scorers results
        best_results[sel_strat]['params'] = df_res.loc[best_idx, "params"].to_dict()
        for scorer_name in evaluation_scorers:
            best_results[sel_strat][f"mean_train_{scorer_name}"] = \
                df_res[f"mean_train_{scorer_name}"][best_idx]
            best_results[sel_strat][f"std_train_{scorer_name}"] = \
                df_res[f"std_train_{scorer_name}"][best_idx]

            best_results[sel_strat][f"mean_validation_{scorer_name}"] = \
                df_res[f"mean_validation_{scorer_name}"][best_idx]
            best_results[sel_strat][f"std_validation_{scorer_name}"] = \
                df_res[f"std_validation_{scorer_name}"][best_idx]

            best_results[sel_strat][f"mean_test_{scorer_name}"] = \
                df_res[f"mean_test_{scorer_name}"][best_idx]
            best_results[sel_strat][f"std_test_{scorer_name}"] = \
                df_res[f"std_test_{scorer_name}"][best_idx]
            
    file_path = os.path.join(file_path, 'selection_strategy_results')
    os.makedirs(file_path, exist_ok = True)
    pickle.dump(dict(best_results), open(os.path.join(file_path, file_name), 'wb'))
