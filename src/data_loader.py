#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
function related to numerai tournament data

current features
* data fetching, loading in parquet format
* pytorch dataloader using era-batch
* load data subset using eras
* load riskiest features, multiple targets
* feature selection
"""

import os
import pickle
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pandas as pd
from numerapi import NumerAPI

from typing import List, Tuple
from yacs.config import CfgNode
from argparse import Namespace

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utils import get_biggest_change_features


def numerai_loader(cfg: CfgNode, split: str = "train", rand: bool = True, era: List = None) -> DataLoader:
    """load pytorch dataloader

    Args:
        cfg (CfgNode): cfg for parameters
        split (str, optional): dataset split ["train", "val", "test"]. Defaults to "train".
        rand (bool, optional): flag for randomize the loading order. Defaults to True.
        era (List, optional): era list for eraboost training. Defaults to None. Defaults to None.

    Returns:
        DataLoader: [description]
    """
    dataset = numerai_dataset(cfg, split=split, era=era)

    loader = DataLoader(
        dataset, 
        shuffle=rand,
        batch_size=cfg.SOLVER.ERA_BATCH_SIZE,
        num_workers=cfg.SOLVER.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True)
    return loader


def numerai_dataset(cfg: CfgNode, split: str = "train", era: List = None) -> Dataset:
    """load pytorch dataset

    Args:
        cfg (CfgNode): cfg for parameters
        split (str, optional): dataset split ["train", "val", "test"]. Defaults to "train".
        era (List, optional): era list for eraboost training. Defaults to None.

    Returns:
        Dataset: [description]
    """
    target_names = parse_targets(cfg)
    if split == "train":
        train_data, feature_names = load_data(cfg, era=era)
        dataset = SimpleDataset(df=train_data,
                                features=feature_names,
                                targets=target_names)
    if split == "val":
        validation_data, feature_names = load_data(
            cfg, split="validation")
        dataset = SimpleDataset(df=validation_data,
                                features=feature_names,
                                targets=target_names)
    elif split == "test":
        tournament_data, feature_names = load_data(
            cfg, split="tournament")
        dataset = SimpleDataset(df=tournament_data,
                                features=feature_names,
                                targets=target_names)
    return dataset


class SimpleDataset(Dataset):
    def __init__(self, df, features, targets):

        self.df = df
        self.features = features
        self.targets = targets
        self.eras = df.era.unique()
        
        self.batches_x = {era : self.df.loc[self.df.era.isin([era]), self.features].values - 0.5 for era in self.eras}
        self.batches_y = {era : self.df.loc[self.df.era.isin([era]), self.targets].values for era in self.eras}

    def __len__(self):
        return len(self.eras)

    def __getitem__(self, idx):
        era = self.eras[idx]

        #x = self.df.loc[self.df.era.isin(era), self.features].values - 0.5
        #y = self.df.loc[self.df.era.isin(era), self.targets].values
        
        x = self.batches_x[era]
        y = self.batches_y[era]

        return [x, y]
    

def collate_fn(data):
    x_list, y_list = zip(*data)
    
    x = np.concatenate(x_list, axis=0, dtype=np.float32)[None, :, :]
    y = np.concatenate(y_list, axis=0, dtype=np.float32)[None, :, :]
    
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    
    return {"input": x, "gt": y}


def parse_targets(cfg: CfgNode) -> List[str]:
    """parse target names by output dimentions

    Args:
        cfg (CfgNode): cfg for parameters

    Returns:
        List[str]: selected target names
    """
    if cfg.MODEL.DIM_OUT == 1:
        target_names = ["target"]
    elif cfg.MODEL.DIM_OUT == 3:
        target_names = ["target", "target_nomi_v4_60", "target_jerome_v4_20"]
    elif cfg.MODEL.DIM_OUT == 20:
        target_names = ["target_nomi_v4_20", "target_nomi_v4_60",
                        "target_jerome_v4_20", "target_jerome_v4_60",
                        "target_janet_v4_20", "target_janet_v4_60",
                        "target_ben_v4_20", "target_ben_v4_60",
                        "target_alan_v4_20", "target_alan_v4_60",
                        "target_paul_v4_20", "target_paul_v4_60",
                        "target_george_v4_20", "target_george_v4_60",
                        "target_william_v4_20", "target_william_v4_60",
                        "target_arthur_v4_20", "target_arthur_v4_60",
                        "target_thomas_v4_20", "target_thomas_v4_60"]
    assert len(target_names) == cfg.MODEL.DIM_OUT
    return target_names


def load_example_live_predictions(napi: NumerAPI, cfg):
    """load example live predictions for submission

    Args:
        napi (NumerAPI): Numerai API instance
        cfg (CfgNode): cfg for parameters

    Returns:
        pd.DataFrame: live prediction
    """
    n_round = napi.get_current_round()
    dir_path = Path(cfg.DATASET_DIR) / f"numerai_live_{n_round}"
    live_pred = pd.read_parquet(
        dir_path / "live_example_preds.parquet")
    return live_pred


def load_example_validation_predictions(napi: NumerAPI, cfg: CfgNode) -> pd.DataFrame:
    """load example validation predictions

    Args:
        napi (NumerAPI): Numerai API instance
        cfg (CfgNode): cfg for parameters

    Returns:
        pd.DataFrame: validation prediction
    """
    dataset_dir = Path(cfg.DATASET_DIR)
    live_pred = pd.read_parquet(
        dataset_dir / "validation_example_preds.parquet")
    return live_pred


def load_riskiest_features(napi: NumerAPI, args: Namespace, cfg: CfgNode) -> List[str]:
    """load riskiest features

    Args:
        napi (NumerAPI): Numerai API instance
        args (Namespace): namespace from argparse
        cfg (CfgNode): cfg for parameter

    Returns:
        List[str]: list of riskiest feature names
    """
    pkl_path = os.path.join(cfg.DATASET_DIR, "riskiest_features.pkl")
    if not os.path.exists(pkl_path):
        train_data, feature_names = load_data(
            napi, cfg, split="train")
        all_feature_corrs = train_data.groupby("era").apply(
            lambda d: d[feature_names].corrwith(d["target"]))
        riskiest_features = get_biggest_change_features(all_feature_corrs, 50)

        with open(pkl_path, "wb") as f:
            pickle.dump(riskiest_features, f)

    with open(pkl_path, "rb") as f:
        riskiest_features = pickle.load(f)

    return riskiest_features


def load_data(cfg: CfgNode, split: str = "train", era: List = None, inference: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    """load pandas df
    load feature subset if feature selection applied

    Args:
        cfg (CfgNode): cfg for parameter
        split (str, optional): dataset split ["train", "validation", "tournament"]. Defaults "train".
        era (List, optional): era list for eraboost training. Defaults to None.
        inference (bool, optional): inference flag. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, List[str]]: df and list of feature names
    """
    if split in ["train", "validation"]:
        if cfg.USE_VAL_SET_FOR_TRAIN:
            parquet_path = [check_train_val_path(cfg, split="train"),
                            check_train_val_path(cfg, split='validation')]
        else:
            parquet_path = [check_train_val_path(cfg, split=split)]
    elif split == "tournament":
        parquet_path = [check_live_path(cfg)]
        
    if cfg.FS.APPLY and not inference:
        feature_names = feature_selection(cfg)
        columns = ["era", "data_type"] + feature_names + parse_targets(cfg)
    else:
        columns = None
    
    data = pd.concat([pd.read_parquet(p, columns=columns) for p in parquet_path])

    feature_names = [
        f for f in data.columns if f.startswith("feature")
    ]

    if era is not None:
        data = data[data.era.isin(era)]

    return data, feature_names


def check_train_val_path(cfg, split="train"):
    parquet_path = os.path.join(
        cfg.DATASET_DIR, f"{split}_fillna.parquet")
    if not os.path.isfile(parquet_path):
        original_parquet_path = os.path.join(
            cfg.DATASET_DIR, f"{split}.parquet")
        if not os.path.isfile(original_parquet_path):
            dir_path = Path(cfg.DATASET_DIR)
            fetch_current_dataset(napi=napi, dest_path=dir_path)
        data = pd.read_parquet(original_parquet_path)
        data.fillna(0.5, inplace=True)
        data.to_parquet(parquet_path)
    
    return parquet_path


def check_live_path(cfg):
    napi = NumerAPI()
    n_round = napi.get_current_round()
    dir_path = Path(cfg.DATASET_DIR) / f"numerai_live_{n_round}"
    if not dir_path.exists():
        print("Fetching live data...")
        fetch_current_dataset(napi=napi, dest_path=dir_path)
    parquet_path = dir_path / "live.parquet"
    return parquet_path


def fetch_dataset(napi: NumerAPI, dest_path: str) -> None:
    """fetch train and validation datasets of numerai tournament

    Args:
        napi (NumerAPI): Numerai API instance
        dest_path (str): path to save datasets
    """
    os.makedirs(dest_path, exist_ok=True)
    filenames = ['train.parquet',
                 'validation.parquet',
                 'validation_example_preds.parquet',
                 'features.json']
    for filename in filenames:
        napi.download_dataset(
            filename=filename, dest_path=os.path.join(dest_path, filename))


def fetch_current_dataset(napi: NumerAPI, dest_path: str) -> None:
    """fetch tournament dataset of numerai tournament

    Args:
        napi (NumerAPI): Numerai API instance
        dest_path (str): [description]
    """
    os.makedirs(dest_path, exist_ok=True)
    filenames = ['live.parquet', "live_example_preds.parquet"]
    for filename in filenames:
        napi.download_dataset(
            filename=os.path.join("v4", filename), dest_path=str(dest_path / filename))


def feature_selection(cfg: CfgNode) -> List[str]:
    """feature selection

    Args:
        cfg (CfgNode): CfgNode for parameters

    Returns:
        List: list of selected feature names
    """
    import pickle
    
    pkl_path = os.path.join(cfg.DATASET_DIR, f"numerai_selected_features_{cfg.FS.ALGORITHM}_{cfg.FS.TEST_SET}.pkl")
    if not os.path.isfile(pkl_path):
        print(f"Start feature selection...")

        _cfg = cfg.clone()
        _cfg.defrost()
        _cfg.FS.APPLY = False

        if _cfg.FS.TEST_SET == "val":
            validation_data, feature_names = load_data(_cfg, split="validation")
            test_set = validation_data
        elif _cfg.FS.TEST_SET == "train":
            train_data, feature_names = load_data(_cfg, split="train")
            test_set = train_data

        if cfg.FS.ALGORITHM == "mda":
            model = pickle.load(open(_cfg.FS.MODEL, 'rb'))
            diff = np.array(MDA(model, feature_names, test_set))
            arg = np.argsort(diff[:, 1])
            feature_names_select = diff[arg][:, 0].tolist()[:_cfg.FS.FEATURE_NUM]

        print("-----------selected features------------")
        print(feature_names_select)
        
        with open(pkl_path, "wb") as f:
            pickle.dump(feature_names_select, f)

    with open(pkl_path, "rb") as f:
        feature_names_select = pickle.load(f)
    
    return feature_names_select


def MDA(model, features, testSet) -> List:
    """Feature Selection by Marcos Lopez de Prado
    from https://forum.numer.ai/t/feature-selection-by-marcos-lopez-de-prado/3170/24
    Args:
        model ([type]): GBDT(XGBoost or LightGBM) or MLP model
        features ([type]): numerai feature names
        testSet ([type]): test set df

    Returns:
        List: feature importance on each features
    """
    # predict with a pre-fitted model on an OOS validation set
    testSet['pred'] = model.predict(testSet[features])
    corr, std = numerai_score(testSet)  # save base scores
    diff = []
    np.random.seed(42)
    for col in tqdm(features):   # iterate through each features

        X = testSet.copy()
        # shuffle the a selected feature column, while maintaining the distribution of the feature
        np.random.shuffle(X[col].values)
        # run prediction with the same pre-fitted model, with one shuffled feature
        testSet['pred'] = model.predict(X[features])
        corrX, stdX = numerai_score(testSet)  # compare scores...
        diff.append((col, corrX-corr))

    return diff


def correlation(predictions, targets):
    ranked_preds = predictions.rank(pct=True, method="first")
    return np.corrcoef(ranked_preds, targets)[0, 1]


def score(df):
    return correlation(df['pred'], df['target'])


def numerai_score(df):
    scores = df.groupby('era').apply(score)
    return scores.mean(), scores.std(ddof=0)


if __name__ == "__main__":
    import time
    from utils import create_api
    from default_param import _C as cfg
    napi = create_api()

    start = time.time()

    numerai_loader(napi, cfg, split='train')
    numerai_loader(napi, cfg, split='val')

    end = time.time()

    print(end - start)
