#!/usr/bin/python
# -*- coding: utf-8 -*-
"""utility functions for this numerai tournament example

utils.py includes functions related to follows.
* Numerai API
* loading yacs config
* evaluation metrics from official examples or Forum

"""

import os
import shutil
from numerapi import NumerAPI
import numpy as np
import pandas as pd
import scipy
import scipy.stats
from argparse import Namespace
from yacs.config import CfgNode

TARGET_COL = "target"
ERA_COL = "era"


def create_api(public_id: str=None, secret_key: str=None) -> NumerAPI:
    """create a Numerai API instance
    if you want to submit your prediction,
    you need to create api key via "setting" of numerai webpage 

    Args:
        public_id (str, optional): user id. Defaults to None.
        secret_key (str, optional): user api secret key. Defaults to None.

    Returns:
        NumerAPI: numerai api instance
    """
    if public_id == secret_key == None:
        return NumerAPI(verbosity="info")
    else:
        return NumerAPI(public_id, secret_key)
    

def load_config(args: Namespace) -> CfgNode:
    """load yaml config file overwriting default_params.py

    Args:
        args (Namespace): args namespace from argparse

    Returns:
        CfgNode: cfg overwritten by the yaml config file
    """
    from default_param import _C as cfg
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cfg_ = cfg.clone()
    if os.path.isfile(args.config_file):
        conf = args.config_file
        print(f"Configuration file loaded from {conf}.")
        cfg_.merge_from_file(conf)
        cfg_.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, os.path.splitext(os.path.basename(conf))[0])
    else:
        print("Use default configuration.")
        cfg_.OUTPUT_DIR = os.path.join(cfg_.OUTPUT_DIR, "default")
    
    if cfg_.LOAD_TUNED and args.mode != "tune":
        cfg_ = load_tuned(cfg_)
    cfg_.freeze()
    
    print(f"output dirname: {cfg_.OUTPUT_DIR}")
    if not args.debug:
        os.makedirs(cfg_.OUTPUT_DIR, exist_ok=True)
        if os.path.isfile(args.config_file):
            shutil.copy2(args.config_file, os.path.join(cfg_.OUTPUT_DIR, 'config.yaml'))

    return cfg_

def load_tuned(cfg: CfgNode) -> CfgNode:
    """load tuned parameters from optuna database

    Args:
        cfg (CfgNode): cfg

    Returns:
        CfgNode: cfg overwritten by the optuna database
    """
    print("load params from optuna database")
    import optuna
    study = optuna.load_study(storage=os.path.join("sqlite:///", cfg.OUTPUT_DIR, "optuna.db"), study_name="my_opt")

    trial_dict = study.best_trial.params
    
    for key in list(trial_dict.keys()):
        if type(trial_dict[key]) == str:
            exec(f"cfg.{key} = '{trial_dict[key]}'")
        else:
            exec(f"cfg.{key} = {trial_dict[key]}")
    
    return cfg


"""
the functions below are from the official numerai tournament example.
"""

def get_biggest_change_features(corrs, n):
    all_eras = corrs.index.sort_values()
    h1_eras = all_eras[:len(all_eras) // 2]
    h2_eras = all_eras[len(all_eras) // 2:]

    h1_corr_means = corrs.loc[h1_eras, :].mean()
    h2_corr_means = corrs.loc[h2_eras, :].mean()

    corr_diffs = h2_corr_means - h1_corr_means
    worst_n = corr_diffs.abs().sort_values(ascending=False).head(n).index.tolist()
    return worst_n


def neutralize(df,
               columns,
               neutralizers=None,
               proportion=1.0,
               normalize=True,
               era_col="era"):
    if neutralizers is None:
        neutralizers = []
    unique_eras = df[era_col].unique()
    computed = []
    for u in unique_eras:
        df_era = df[df[era_col] == u]
        scores = df_era[columns].values
        if normalize:
            scores2 = []
            for x in scores.T:
                x = (scipy.stats.rankdata(x, method='ordinal') - .5) / len(x)
                x = scipy.stats.norm.ppf(x)
                scores2.append(x)
            scores = np.array(scores2).T
        exposures = df_era[neutralizers].values

        scores -= proportion * exposures.dot(
            np.linalg.pinv(exposures.astype(np.float32)).dot(scores.astype(np.float32)))
        #scores -= proportion * exposures.dot(
        #    np.linalg.lstsq(exposures.astype(np.float32), scores.astype(np.float32)))

        scores /= scores.std(ddof=0)

        computed.append(scores)

    return pd.DataFrame(np.concatenate(computed),
                        columns=columns,
                        index=df.index)


def neutralize_series(series, by, proportion=1.0):
    scores = series.values.reshape(-1, 1)
    exposures = by.values.reshape(-1, 1)

    # this line makes series neutral to a constant column so that it's centered and for sure gets corr 0 with exposures
    exposures = np.hstack(
        (exposures,
         np.array([np.mean(series)] * len(exposures)).reshape(-1, 1)))

    correction = proportion * (exposures.dot(
        np.linalg.lstsq(exposures, scores, rcond=None)[0]))
    corrected_scores = scores - correction
    neutralized = pd.Series(corrected_scores.ravel(), index=series.index)
    return neutralized


def unif(df):
    x = (df.rank(method="first") - 0.5) / len(df)
    return pd.Series(x, index=df.index)


def get_feature_neutral_mean(df, prediction_col):
    feature_cols = [c for c in df.columns if c.startswith("feature")]
    df.loc[:, "neutral_sub"] = neutralize(df, [prediction_col],
                                          feature_cols)[prediction_col]
    scores = df.groupby(ERA_COL).apply(
        lambda x: (unif(x["neutral_sub"]).corr(x[TARGET_COL]))).mean()
    return np.mean(scores)


def fast_score_by_date(df, columns, target, tb=None, era_col="era"):
    unique_eras = df[era_col].unique()
    computed = []
    for u in unique_eras:
        df_era = df[df[era_col] == u]
        era_pred = np.float64(df_era[columns].values.T)
        era_target = np.float64(df_era[target].values.T)

        if tb is None:
            ccs = np.corrcoef(era_target, era_pred)[0, 1:]
        else:
            tbidx = np.argsort(era_pred, axis=1)
            tbidx = np.concatenate([tbidx[:, :tb], tbidx[:, -tb:]], axis=1)
            ccs = [np.corrcoef(era_target[tmpidx], tmppred[tmpidx])[0, 1] for tmpidx, tmppred in zip(tbidx, era_pred)]
            ccs = np.array(ccs)

        computed.append(ccs)

    return pd.DataFrame(np.array(computed), columns=columns, index=df[era_col].unique())


def validation_metrics(validation_data, pred_cols, example_col, fast_mode=False):
    validation_stats = pd.DataFrame()
    feature_cols = [c for c in validation_data if c.startswith("feature_")]
    for pred_col in pred_cols:
        # Check the per-era correlations on the validation set (out of sample)
        validation_correlations = validation_data.groupby(ERA_COL).apply(
            lambda d: unif(d[pred_col]).corr(d[TARGET_COL]))

        mean = validation_correlations.mean()
        std = validation_correlations.std(ddof=0)
        sharpe = mean / std

        validation_stats.loc["mean", pred_col] = mean
        validation_stats.loc["std", pred_col] = std
        validation_stats.loc["sharpe", pred_col] = sharpe

        rolling_max = (validation_correlations + 1).cumprod().rolling(window=9000,  # arbitrarily large
                                                                      min_periods=1).max()
        daily_value = (validation_correlations + 1).cumprod()
        max_drawdown = -((rolling_max - daily_value) / rolling_max).max()
        validation_stats.loc["max_drawdown", pred_col] = max_drawdown

        payout_scores = validation_correlations.clip(-0.25, 0.25)
        payout_daily_value = (payout_scores + 1).cumprod()

        apy = (
            (
                (payout_daily_value.dropna().iloc[-1])
                ** (1 / len(payout_scores))
            )
            ** 49  # 52 weeks of compounding minus 3 for stake compounding lag
            - 1
        ) * 100

        validation_stats.loc["apy", pred_col] = apy

        if not fast_mode:
            # Check the feature exposure of your validation predictions
            max_per_era = validation_data.groupby(ERA_COL).apply(
                lambda d: d[feature_cols].corrwith(d[pred_col]).abs().max())
            max_feature_exposure = max_per_era.mean()
            validation_stats.loc["max_feature_exposure", pred_col] = max_feature_exposure

            # Check feature neutral mean
            feature_neutral_mean = get_feature_neutral_mean(validation_data, pred_col)
            validation_stats.loc["feature_neutral_mean", pred_col] = feature_neutral_mean

            # Check top and bottom 200 metrics (TB200)
            tb200_validation_correlations = fast_score_by_date(
                validation_data,
                [pred_col],
                TARGET_COL,
                tb=200,
                era_col=ERA_COL
            )

            tb200_mean = tb200_validation_correlations.mean()[pred_col]
            tb200_std = tb200_validation_correlations.std(ddof=0)[pred_col]
            tb200_sharpe = mean / std

            validation_stats.loc["tb200_mean", pred_col] = tb200_mean
            validation_stats.loc["tb200_std", pred_col] = tb200_std
            validation_stats.loc["tb200_sharpe", pred_col] = tb200_sharpe

        # MMC over validation
        mmc_scores = []
        corr_scores = []
        for _, x in validation_data.groupby(ERA_COL):
            series = neutralize_series(unif(x[pred_col]), (x[example_col]))
            mmc_scores.append(np.cov(series, x[TARGET_COL])[0, 1] / (0.29 ** 2))
            corr_scores.append(unif(x[pred_col]).corr(x[TARGET_COL]))

        val_mmc_mean = np.mean(mmc_scores)
        val_mmc_std = np.std(mmc_scores)
        corr_plus_mmcs = [c + m for c, m in zip(corr_scores, mmc_scores)]
        corr_plus_mmc_sharpe = np.mean(corr_plus_mmcs) / np.std(corr_plus_mmcs)

        validation_stats.loc["mmc_mean", pred_col] = val_mmc_mean
        validation_stats.loc["corr_plus_mmc_sharpe", pred_col] = corr_plus_mmc_sharpe

        # Check correlation with example predictions
        per_era_corrs = validation_data.groupby(ERA_COL).apply(lambda d: unif(d[pred_col]).corr(unif(d[example_col])))
        corr_with_example_preds = per_era_corrs.mean()
        validation_stats.loc["corr_with_example_preds", pred_col] = corr_with_example_preds

    # .transpose so that stats are columns and the model_name is the row
    return validation_stats.transpose()

def correlation(predictions, targets):
    ranked_preds = predictions.rank(pct=True, method="first")
    return np.corrcoef(ranked_preds, targets)[0, 1]

def score(df):
    return correlation(df["prediction"], df["target"])

if __name__ == "__main__":
    napi = create_api()
    
    