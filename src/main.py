import lightgbm
from torch.utils.data.dataloader import DataLoader
from utils import *
from model import ModelWithLoss
from data_loader import numerai_loader, load_example_validation_predictions, load_data, load_riskiest_features
from torch.optim import SGD
from torch_optimizer import RAdam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
import torch
import os
import argparse
import random
from tqdm import tqdm
from typing import List, Union
from yacs.config import CfgNode
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter('ignore')


EXAMPLE_PREDS_COL = "example_preds"
TARGET_COL = "target"
ERA_COL = "era"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="pytorch training code")
    parser.add_argument("--config_file", type=str, default='',
                        metavar="FILE", help='path to config file')
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument(
        "--mode", type=str, choices=["train", "test", "submit", "tune"], default="train")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    return parser.parse_args()


def era_boost_train(args: argparse.Namespace, cfg: CfgNode) -> None:
    """training models by era-boosted train
    
    Args:
        args (argparse.Namespace): argparse namespace
        cfg (CfgNode): cfg for parameters
    """
    proportion = 0.5

    models = []
    worst_eras = None

    pred_cols = set()
    ensemble_col = "ensemble"
    df_train = load_data(cfg, split="training", inference=True)[0]
    for i in range(cfg.MODEL.ERABOOST_ITER):
        print(f"ERABOOST epoch {i+1}/{cfg.MODEL.ERABOOST_ITER}")
        data_loader = {"train": numerai_loader(cfg, "train", era=worst_eras)}
        models.append(train(args=args, cfg=cfg,
                      data_loader=data_loader, model=Build_Model(args, cfg)))
        pred = inference_on_data(cfg=cfg, model=models[-1], data_loader=numerai_loader(cfg, split="train", rand=False))
        cur_col = f"pred{i}"
        pred_cols.add(cur_col)
        df_train[cur_col] = pred
        df_train[cur_col] = df_train.groupby(ERA_COL).apply(
            lambda d: d[[cur_col]].rank(pct=True))
        df_train[ensemble_col] = sum([df_train[col]
                                     for col in pred_cols]).rank(pct=True)
        era_scores = pd.Series(index=df_train[ERA_COL].unique())
        print("getting per era scores")
        for era in df_train[ERA_COL].unique():
            era_df = df_train[df_train[ERA_COL] == era]
            era_scores[era] = correlation(
                era_df[ensemble_col], era_df[TARGET_COL])
        era_scores.sort_values(inplace=True)
        worst_eras = era_scores[era_scores <=
                                era_scores.quantile(proportion)].index
        print(list(worst_eras))

    for i, model in enumerate(models):
        save_one_model(cfg, model, os.path.join(cfg.OUTPUT_DIR, f"model{i}"))


def time_series_CV(args: argparse.Namespace, cfg: CfgNode, save_model: bool = True) -> float:
    """training models by time-series cross-validation
    and calculate cv score

    Args:
        args (argparse.Namespace): argparse namespace
        cfg (CfgNode): cfg for parameters
        save_model (bool, optional): flag for whether saving models. Defaults to True.

    Returns:
        float: cross-validation score based on Sharpe
    """
    num_folds = cfg.MODEL.CV_ITER

    models = []

    cv_scores = []
    df_train = load_data(cfg, split="training")[0]
    eras = df_train.era.unique()
    del df_train
    num_era = int(len(eras) / (num_folds + 1)) + 1
    for i in range(num_folds):
        print(f"Time series CV epoch {i+1}/{num_folds}")
        train_eras = eras[:num_era * (i+1)]
        valid_eras = eras[num_era * (i+1): num_era * (i+2)]
        data_loader = {"train": numerai_loader(cfg, "train", era=train_eras),
                       "val": numerai_loader(cfg, "train", era=valid_eras, rand=False)}
        models.append(train(args=args, cfg=cfg,
                      data_loader=data_loader, model=Build_Model(args, cfg)))
        pred = inference_on_data(cfg=cfg, model=models[-1], data_loader=data_loader["val"])
        df_train = load_data(cfg, split="training", era=valid_eras, inference=True)[0]
        cur_col = f"prediction"
        df_train[cur_col] = pred
        df_train[cur_col] = df_train.groupby(ERA_COL).apply(
            lambda d: d[[cur_col]].rank(pct=True))
        validation_correlations = df_train.groupby(ERA_COL).apply(score)
        mean = validation_correlations.mean()
        std = validation_correlations.std(ddof=0)
        sharpe = mean / std
        cv_scores.append(sharpe)

        del data_loader
        del df_train

    if save_model:
        for i, model in enumerate(models):
            save_one_model(cfg, model, os.path.join(cfg.OUTPUT_DIR, f"model{i}"))
    return np.mean(cv_scores)


def train_one_model(args: argparse.Namespace, cfg: CfgNode) -> None:
    """training one model

    Args:
        args (argparse.Namespace): argparse namespace
        cfg (CfgNode): cfg for parameters
    """
    data_loader = {"train": numerai_loader(cfg, "train")}
    model = train(args=args, cfg=cfg, data_loader=data_loader,
                  model=Build_Model(args, cfg))
    save_one_model(cfg, model, os.path.join(cfg.OUTPUT_DIR, f"model"))


def save_one_model(cfg: CfgNode, model: Union[LGBMRegressor, ModelWithLoss], path: str):
    """save one model

    Args:
        cfg (CfgNode): cfg for parameters
        model ([type]): GBDT model or Pytorch NN model
        path (str): path to saving destination

    Raises:
        ValueError: unknown model type
    """
    if cfg.MODEL.TYPE == "gbdt":
        import pickle
        postfix = ".pkl"
        pickle.dump(model, open(path + postfix, "wb"))
    elif cfg.MODEL.TYPE == "mlp":
        postfix = ".pth"
        torch.save(model.state_dict(), path + postfix)
    else:
        raise ValueError(f"unknown model type {cfg.MODEL.TYPE}")

def Build_Model(args: argparse.Namespace, cfg: CfgNode) -> Union[LGBMRegressor, ModelWithLoss]:
    """build GBDT or Pytorch NN model

    Args:
        args (argparse.Namespace): argparse namespace
        cfg (CfgNode): cfg for parameters

    Raises:
        ValueError: unknown model type

    Returns:
        Union[LGBMRegressor, ModelWithLoss]: GBDT or Pytorch NN model
    """
    if cfg.MODEL.TYPE == "gbdt":
        params = {"max_depth": cfg.MODEL.GBDT.MAX_DEPTH,
                  "learning_rate": cfg.MODEL.GBDT.LR,
                  "n_estimators": cfg.MODEL.GBDT.N_ESTIMATORS,
                  "num_leaves": cfg.MODEL.GBDT.NUM_LEAVES,
                  "n_jobs": 16,
                  "colsample_bytree": 0.1}
        model = MultiOutputRegressor(LGBMRegressor(**params), n_jobs=1)
    elif cfg.MODEL.TYPE == "mlp":
        model = ModelWithLoss(cfg).cuda()
    else:
        raise ValueError(f"unknown model type {cfg.MODEL.TYPE}")
    return model


def train(**kwargs):
    """training models based on model type

    Returns:
        [type]: trained GBDT or Pytorch NN model
    """
    if kwargs['cfg'].MODEL.TYPE == "gbdt":
        return train_GBDT(**kwargs)
    elif kwargs['cfg'].MODEL.TYPE == "mlp":
        return train_NN(**kwargs)


def train_GBDT(args: argparse.Namespace, cfg: CfgNode, data_loader: DataLoader, model: LGBMRegressor) -> LGBMRegressor:
    """train gbdt model

    Args:
        args (argparse.Namespace): argparse namespace
        cfg (CfgNode): cfg for parameters
        data_loader (DataLoader): pytorch dataloader
        model (LGBMRegressor): gbdt model to train

    Returns:
        LGBMRegressor: trained gbdt model
    """
    dataset = data_loader["train"].dataset
    df_train = dataset.df
    features = dataset.features
    targets = dataset.targets
    model.fit(df_train[features], df_train[targets])
    return model


def train_NN(args: argparse.Namespace, cfg: CfgNode, data_loader: DataLoader, model: ModelWithLoss) -> ModelWithLoss:
    """train pytorch NN model

    Args:
        args (argparse.Namespace): argparse namespace
        cfg (CfgNode): cfg for parameters
        data_loader (DataLoader): pytorch dataloader
        model (ModelWithLoss): pytorch NN model to train

    Returns:
        ModelWithLoss: trained NN model
    """

    if cfg.SOLVER.OPTIMIZER == "radam":
        optimizer = RAdam(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=cfg.SOLVER.LR,
                          weight_decay=cfg.MODEL.REG_WEIGHT)
    elif cfg.SOLVER.OPTIMIZER == "sgd":
        optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=cfg.SOLVER.LR,
                        momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer,
                                  factor=cfg.SOLVER.DECAY_FACTOR,
                                  patience=10)

    #print('Training Starts!!!')
    with tqdm(range(cfg.SOLVER.ITER), ncols=100) as pbar:
        for i in pbar:
            model.train()
            logging_loss = 0
            for data_dict in data_loader["train"]:
                optimizer.zero_grad()
                _, loss = model(data_dict)
                loss = loss.mean()

                loss.backward()

                optimizer.step()
                logging_loss += loss.item()
            scheduler.step(logging_loss)
            logging_loss /= len(data_loader["train"])

            np.set_printoptions(
                precision=4, floatmode='maxprec', suppress=True)

            pbar.set_postfix(
                dict(
                    loss=np.asarray(
                        [logging_loss]),
                    lr=[optimizer.param_groups[0]['lr']]
                ))

            if optimizer.param_groups[0]['lr'] < cfg.SOLVER.LR * 1e-3:
                print("early stopping")
                break

    return model


def val(args: argparse.Namespace, cfg: CfgNode, model: Union[LGBMRegressor, ModelWithLoss]) -> float:
    """performs validation on numerai validation set

    Args:
        args (argparse.Namespace): argparse namespace
        cfg (CfgNode): cfg for parameters
        model ([type]): GBDT model or Pytorch NN model

    Returns:
        float: Sharpe value
    """
    val_loader = numerai_loader(cfg, split="val", rand=False)
    test_pred = inference_on_data(cfg, model, data_loader=val_loader)

    validation_data, feature_names = load_data(cfg, split="validation")
    validation_data["prediction"] = test_pred
    validation_correlations = validation_data.groupby(ERA_COL).apply(score)

    mean = validation_correlations.mean()
    std = validation_correlations.std(ddof=0)
    sharpe = mean / std
    return sharpe


def ensemble_on_era(df: pd.DataFrame, preds: List, riskiest_features: List[str]) -> pd.DataFrame:
    """ensemble predictions based on each era

    Args:
        df (pd.DataFrame): dataframe
        preds (List): list of predictions by models
        riskiest_features (List[str]): riskiest features to neutralize

    Returns:
        pd.DataFrame: dataframe containing ensembled prediction
    """
    pred_cols = set()
    for i, pred in enumerate(preds):
        cur_col = f"pred{i}"
        pred_cols.add(cur_col)
        df[cur_col] = np.sum(pred, axis=1) if len(pred.shape) == 2 else pred
        df[cur_col] = neutralize(df=df,
                                 columns=[cur_col],
                                 neutralizers=riskiest_features,
                                 proportion=1.0,
                                 normalize=True,
                                 era_col=ERA_COL)[[cur_col]]
    df[list(pred_cols)] = df.groupby(ERA_COL).apply(
        lambda d: d[list(pred_cols)].rank(pct=True))
    df["prediction"] = sum([df[col] for col in pred_cols]).rank(pct=True)
    return df


def inference_on_data(cfg: CfgNode, model: Union[LGBMRegressor, ModelWithLoss], data_loader: DataLoader) -> np.ndarray:
    """inference on data

    Args:
        cfg (CfgNode): cfg for parameters
        model (Union[LGBMRegressor, ModelWithLoss]): GBDT model or Pytorch NN model
        data_loader (DataLoader): pytorch dataloader

    Raises:
        ValueError: unknown model types

    Returns:
        np.ndarray: prediction
    """
    if cfg.MODEL.TYPE == "gbdt":
        return inference_GBDT(cfg, model, data_loader)
    elif cfg.MODEL.TYPE == "mlp":
        return inference_NN(cfg, model, data_loader)
    else:
        raise ValueError(f"unknown model type {cfg.MODEL.TYPE}")

def inference_GBDT(cfg: CfgNode, model: Union[LGBMRegressor, ModelWithLoss], data_loader: DataLoader) -> np.ndarray:
    """inference gbdt model on data

    Args:
        cfg (CfgNode): cfg for parameters
        model (Union[LGBMRegressor, ModelWithLoss]): GBDT model or Pytorch NN model
        data_loader (DataLoader): pytorch dataloader

    Returns:
        np.ndarray: prediction
    """
    dataset = data_loader.dataset
    pred = model.predict(dataset.df[dataset.features])
    pred = np.mean(pred, axis=-1)
    return pred.squeeze()


def inference_NN(cfg: CfgNode, model: Union[LGBMRegressor, ModelWithLoss], data_loader: DataLoader) -> np.ndarray:
    """inference pytorch NN model on data

    Args:
        cfg (CfgNode): cfg for parameters
        model (Union[LGBMRegressor, ModelWithLoss]): GBDT model or Pytorch NN model
        data_loader (DataLoader): pytorch dataloader

    Returns:
        np.ndarray: prediction
    """
    pred = []
    for data_dict in data_loader:
        with torch.no_grad():
            pred_cur = model.predict(data_dict)
            pred.append(pred_cur.detach().cpu().numpy())
    pred = np.concatenate(pred, axis=1)
    pred = np.mean(pred, axis=2)  # mean for multi-targets
    return pred.squeeze()


def load_models(args: argparse.Namespace, cfg: CfgNode) -> List[Union[LGBMRegressor, ModelWithLoss]]:
    """load saved models

    Args:
        args (argparse.Namespace): argparse namespace
        cfg (CfgNode): cfg for parameters

    Returns:
        List[Union[LGBMRegressor, ModelWithLoss]]: list of loaded models
    """
    models = []

    if cfg.MODEL.ERABOOST:
        models = [load_one_model(args, cfg, f"model{i}")
                  for i in range(cfg.MODEL.ERABOOST_ITER)]
    elif cfg.MODEL.TIME_SERIES_CV:
        models = [load_one_model(args, cfg, f"model{i}")
                  for i in range(cfg.MODEL.CV_ITER)]
    else:
        models = [load_one_model(args, cfg, "model")]

    return models


def load_one_model(args: argparse.Namespace, cfg: CfgNode, path: str) -> Union[LGBMRegressor, ModelWithLoss]:
    """load one model

    Args:
        args (argparse.Namespace): argparse namespace
        cfg (CfgNode): cfg for parameters
        path (str): path to the saved model

    Raises:
        ValueError: unknown model type

    Returns:
        Union[LGBMRegressor, ModelWithLoss]: loaded model
    """
    if cfg.MODEL.TYPE == "gbdt":
        import pickle
        postfix = ".pkl"
        model = pickle.load(
            open(os.path.join(cfg.OUTPUT_DIR, path + postfix), 'rb'))
    elif cfg.MODEL.TYPE == "mlp":
        postfix = ".pth"
        model = Build_Model(args, cfg)
        model.load_state_dict(torch.load(
            os.path.join(cfg.OUTPUT_DIR, path + postfix)))
    else:
        raise ValueError(f"unknown model type {cfg.MODEL.TYPE}")

    return model


def test(args: argparse.Namespace, cfg: CfgNode) -> None:
    """performs validation on numerai validation set
    and save prediction for diagnostic tools
    
    Args:
        args (argparse.Namespace): argparse namespace
        cfg (CfgNode): cfg for parameters
    """
    models = load_models(args, cfg)

    napi = create_api()

    validation_data, feature_names = load_data(
        cfg, split="validation", inference=True)
    riskiest_features = load_riskiest_features(napi, args, cfg)

    print("Generating predictions...")
    val_loader = numerai_loader(cfg, split="val", rand=False)
    val_preds = [inference_on_data(cfg, model, data_loader=val_loader)
                 for model in tqdm(models)]

    validation_data = ensemble_on_era(
        validation_data, val_preds, riskiest_features)
    example_validation_preds = load_example_validation_predictions(napi, cfg)
    validation_data[EXAMPLE_PREDS_COL] = example_validation_preds["prediction"]

    validation_stats = validation_metrics(  
        validation_data, ["prediction"], example_col=EXAMPLE_PREDS_COL, fast_mode=False)
    validation_stats.to_csv(os.path.join(
        cfg.OUTPUT_DIR, "varidation_stats.csv"), header=True)

    validation_data["prediction"].to_csv(os.path.join(
        cfg.OUTPUT_DIR, "validation.csv"), header=True)


def create_submission(args: argparse.Namespace, cfg: CfgNode) -> str:
    """inferece models and save prediction

    Args:
        args (argparse.Namespace): argparse namespace
        cfg (CfgNode): cfg for parameters

    Returns:
        str: path to the saved prediction
    """
    napi = create_api()
    if args.config_file in ["config/example"]:
        n_round = napi.get_current_round()
        path = os.path.join(cfg.DATASET_DIR, f"numerai_dataset_{n_round}", "example_predictions.csv")
        return path

    path = os.path.join(cfg.OUTPUT_DIR, f"submission.csv")

    models = load_models(args, cfg)
    riskiest_features = load_riskiest_features(napi, args, cfg)

    # Save predictions as a CSV and upload to https://numer.ai
    tournament_data, feature_names = load_data(
        cfg, split="tournament", inference=True)
    test_loader = numerai_loader(cfg, split="test", rand=False)
    test_preds = [inference_on_data(cfg, model, data_loader=test_loader)
                  for model in tqdm(models)]
    tournament_data = ensemble_on_era(
        tournament_data, test_preds, riskiest_features)
    tournament_data["prediction"].to_csv(path, header=True)

    return path

def fix_seed(cfg: CfgNode):
    """fix seed for reproducibility

    Args:
        cfg (CfgNode): cfg for parameters

    Raises:
        Exception: No GPU
    """
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed(cfg.SEED)
    else:
        raise Exception("GPU not found")


def submit(args: argparse.Namespace):
    """
    create submission
    choose good performance model based on past rounds
    submit via numerapi

    please fill "model{i}" with your model name

    Args:
        args (argparse.Namespace): argparse namespace
    """
    
    dict_model = {"model3": "config/mlp.yml",
                  "model4": "config/gbdt.yml",
                  "model5": "config/example"}
    
    dict_path = {}
    for k in dict_model:
        args.config_file = dict_model[k]
        cfg = load_config(args)
        path = create_submission(args, cfg) 
        dict_path[k] = path
    
    napi = NumerAPI(public_id=cfg.NUMERAI.ID, secret_key=cfg.NUMERAI.KEY)
    dict_performance, dict_performance_2xmmc = {}, {}
    for k in dict_model:
        performance = np.mean([round['corr'] for round in napi.round_model_performances(k)[:4] if round['payout'] is not None])
        performance_2xmmc = np.mean([round['corr'] + 2*round['mmc'] for round in napi.round_model_performances(k)[:4] if round['payout'] is not None])
        dict_performance[k] = performance if not np.isnan(performance) else -0.25
        dict_performance_2xmmc[k] = performance_2xmmc if not np.isnan(performance_2xmmc) else -0.25
    dict_path["model1"] = dict_path[sorted(dict_performance_2xmmc, key=dict_performance_2xmmc.get)[-1]]
    dict_path["model2"] = dict_path[sorted(dict_performance, key=dict_performance.get)[-1]]
    
    model_ids = napi.get_models()

    for k in dict_path:
        model_id = model_ids[k]
        napi.upload_predictions(dict_path[k], model_id=model_id, version=2)
            
        
def tune(args: argparse.Namespace, cfg: CfgNode):
    """tuning hyperparameters by optuna

    Args:
        args (argparse.Namespace): argparse namespace
        cfg (CfgNode): cfg for parameters

    Raises:
        ValueError: unknown model type
    """
    import optuna

    def objective_with_arg(args: argparse.Namespace, cfg: CfgNode):
        """objective function with arg

        Args:
            args (argparse.Namespace): argparse namespace
            cfg (CfgNode): cfg for parameters

        Raises:
            ValueError: unknown model type
        """
        _cfg = cfg.clone()
        _cfg.defrost()

        def objective(trial):
            if cfg.MODEL.TYPE == "mlp":
                _cfg.MODEL.NUM_STAGE = trial.suggest_int("MODEL.NUM_STAGE", 1, 8)
                _cfg.MODEL.NUM_HIDDEN = trial.suggest_int("MODEL.NUM_HIDDEN", 1, 4)
                _cfg.MODEL.DIM_HIDDEN = trial.suggest_int(
                    "MODEL.DIM_HIDDEN", 256, 1024)
                _cfg.MODEL.P_DROPOUT = trial.suggest_float(
                    "MODEL.P_DROPOUT", 0.0, 0.3)
                _cfg.MODEL.REG_WEIGHT = trial.suggest_float(
                    "MODEL.REG_WEIGHT", 1e-8, 0.0001, log=True)
                _cfg.MODEL.PRED_LOSS = trial.suggest_categorical(
                    "MODEL.PRED_LOSS", ["mse", "l1", "bce"])
                _cfg.SOLVER.LR = trial.suggest_float(
                    "SOLVER.LR", 1e-5, 0.001, log=True)
                if cfg.FE.APPLY:
                    _cfg.FE.NUM_HIDDEN = trial.suggest_int("FE.NUM_HIDDEN", 1, 4)
                    _cfg.FE.DIM_HIDDEN = trial.suggest_int("FE.DIM_HIDDEN", 128, 512, step=16)
                    _cfg.FE.DIM_OUT = trial.suggest_int("FE.DIM_OUT", 16, 128, step=16)
            elif cfg.MODEL.TYPE == "gbdt":
                _cfg.MODEL.DIM_OUT = trial.suggest_categorical("MODEL.DIM_OUT", [1, 3, 20])
                _cfg.MODEL.GBDT.LR = trial.suggest_float("MODEL.GBDT.LR", 0.01, 0.3)
                _cfg.MODEL.GBDT.MAX_DEPTH = trial.suggest_int("MODEL.GBDT.MAX_DEPTH", 3, 12)
                _cfg.MODEL.GBDT.NUM_LEAVES = trial.suggest_int("MODEL.GBDT.NUM_LEAVES", 20, 3000, step=20)
                _cfg.MODEL.GBDT.N_ESTIMATORS = trial.suggest_categorical("MODEL.GBDT.N_ESTIMATORS", [300, 2000, 10000])
                _cfg.MODEL.GBDT.REG_ALPHA = trial.suggest_float('MODEL.GBDT.REG_ALPHA', 0.0, 1000.0)
                _cfg.MODEL.GBDT.REG_LAMBDA = trial.suggest_float('MODEL.GBDT.REG_LAMBDA', 0.0, 1000.0)

            else:
                raise ValueError(f"unknown model type {cfg.MODEL.TYPE}")
        
            #model = train(args=args, cfg=_cfg, data_loader=dataloader, model=Build_Model(args, cfg))
            # return val(args, _cfg, model)
            return time_series_CV(args, _cfg, save_model=False)

        return objective

    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.HyperbandPruner()

    study = optuna.create_study(sampler=sampler, pruner=pruner,
                                direction="maximize",
                                storage=os.path.join(
                                    "sqlite:///", cfg.OUTPUT_DIR, "optuna.db"),
                                study_name="my_opt",
                                load_if_exists=True)
    study.optimize(objective_with_arg(args, cfg), n_trials=200, gc_after_trial=True)

    trial = study.best_trial

    print(trial.value, trial.params)


def main() -> None:
    args = parse_args()

    if args.mode == "submit":
        submit(args)
    else:
        cfg = load_config(args)

    
    if args.mode == "test":
        test(args, cfg)
    elif args.mode == "train":
        if cfg.MODEL.ERABOOST:
            era_boost_train(args, cfg)
        elif cfg.MODEL.TIME_SERIES_CV:
            time_series_CV(args, cfg)
        else:
            train_one_model(args, cfg)
    elif args.mode == "tune":
        tune(args, cfg)


if __name__ == "__main__":
    main()
