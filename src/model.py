#!/usr/bin/python
# -*- coding: utf-8 -*-
"""models for NN using pytorch

current features
* default MLP model
* feature engineering using AutoEncoder
* several dropout function for numerai tournament

"""
from yacs.config import CfgNode
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

numerai_input_size = 1050

class ModelWithLoss(nn.Module):
    """Pytorch NN model with loss functions
    Forward computation accept a input dict and returns output dict and "loss".
    This feature makes loss definition independent from training loop.
    Instead of forward, "predict" method performs inference.
    """
    def __init__(self, cfg: CfgNode):
        """__init__ function 

        Args:
            cfg (CfgNode): yacs CfgNode for parameters
        """
        super(ModelWithLoss, self).__init__()
        
        input_size = numerai_input_size
        if cfg.FS.APPLY:
            input_size = cfg.FS.FEATURE_NUM

        self.fe_model = None
        if cfg.FE.APPLY:
            self.fe_model = FEModel(cfg)
            input_size = self.fe_model.dim_out
        
        if cfg.MODEL.TYPE == "mlp":
            self.model = MLP(cfg, input_size=input_size)
            
        self.build_loss(cfg)
        
    def forward(self, data_dict: Dict) -> Tuple[Dict, torch.Tensor]:
        """forward method returning loss

        Args:
            data_dict (Dict): input data_dict from dataloader

        Returns:
            Tuple[Dict, torch.Tensor]: output dict and loss
        """
        loss = 0
        
        if self.fe_model is not None:
            data_dict, loss_fe = self.fe_model(data_dict)
            loss += loss_fe
        
        x = data_dict["input"].float().cuda()
        pred = self.model(x)
        data_dict["pred"] = torch.sigmoid(pred)
        loss += self.loss(data_dict)
        if torch.isnan(data_dict["pred"]).any():
            import pdb;pdb.set_trace()
        return data_dict, loss
    
    def predict(self, data_dict: Dict) -> torch.Tensor:
        """predict method for inference

        Args:
            data_dict (Dict): input data_dict from dataloader

        Returns:
            torch.Tensor: prediction
        """
        return self.forward(data_dict)[0]['pred']
    
    def loss(self, data_dict: Dict) -> torch.Tensor:
        """loss calculation based on specified loss type

        Args:
            data_dict (Dict): output dict

        Returns:
            torch.Tensor: loss value for backward
        """
        loss = 0
        loss += self.pred_loss(data_dict["pred"],
                               data_dict["gt"].float().cuda())
        
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                loss += torch.norm(param, self.reg_norm) * self.reg_weight
                    
        return loss

    def build_loss(self, cfg: CfgNode):
        """build loss function based on specified loss type

        Args:
            cfg (CfgNode): yacs CfgNode for parameters
        """
        self.pred_loss = choose_loss(cfg.MODEL.PRED_LOSS)
        self.reg_norm = 1 if cfg.MODEL.REG_TYPE == "l1" else 2
        self.reg_weight = cfg.MODEL.REG_WEIGHT
        
        
class MLP(nn.Module):
    def __init__(self, cfg, input_size=numerai_input_size):
        super(MLP, self).__init__()

        dim_o = cfg.MODEL.DIM_OUT

        num_s = cfg.MODEL.NUM_STAGE
        dim_h = cfg.MODEL.DIM_HIDDEN
        num_l = cfg.MODEL.NUM_HIDDEN
        coef = cfg.MODEL.DIM_HIDDEN_FACT

        self.act = choose_activation(cfg.MODEL.ACTIVATION)
        self.noise = FeatureReversalNoise()
        self.gaussian_dropout = CoupledGaussianDropout()
        
        self.residual = cfg.MODEL.RESIDUAL
        self.h_size = [input_size]
        for _ in range(num_s):
            for _ in range(num_l):
                self.h_size.append(dim_h)
            dim_h = int(dim_h * coef)
        self.h_size.append(dim_o)
        
        self.layers = make_mlp(self.h_size, self.act, self.residual)
        
    def forward(self, x):
        x = self.noise(x)
        x = self.gaussian_dropout(x)
        
        x = self.layers(x)
        return x
    

class FEModel(nn.Module):
    """feature engineering model used in ModelWithLoss
    overwrite the input features by FE features in forward method
    """
    def __init__(self, cfg: CfgNode):
        super(FEModel, self).__init__()
        
        input_size = numerai_input_size
        if cfg.FS.APPLY:
            input_size = cfg.FS.FEATURE_NUM
        
        if cfg.FE.TYPE == "autoencoder":
            self.model = AE(cfg, input_size=input_size)
            self.dim_out = cfg.FE.DIM_OUT + input_size
            
        self.build_loss(cfg)
        
    def forward(self, data_dict: Dict) -> Tuple[Dict, torch.Tensor]:
        x = data_dict["input"].float().cuda()
        feature, x_ = self.model(x)
        loss = self.loss(x, x_)
        
        data_dict["input"] = torch.cat((x, feature), axis=2)

        return data_dict, loss
        
    def loss(self, x: torch.Tensor, x_: torch.Tensor) -> torch.Tensor:
        loss = 0
        loss += self.pred_loss(x_, x.detach())
        
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                loss += torch.norm(param, self.reg_norm) * self.reg_weight
                    
        return loss
        
    def build_loss(self, cfg: CfgNode):
        self.pred_loss = choose_loss(cfg.FE.PRED_LOSS)
        self.reg_norm = 1 if cfg.FE.REG_TYPE == "l1" else 2
        self.reg_weight = cfg.FE.REG_WEIGHT
    
    
class AE(nn.Module):
    def __init__(self, cfg, input_size=numerai_input_size):
        super(AE, self).__init__()
        
        dim_h = cfg.FE.DIM_HIDDEN
        num_l = cfg.FE.NUM_HIDDEN
        dim_o = cfg.FE.DIM_OUT
        coef = cfg.FE.DIM_HIDDEN_FACT
        
        self.h_size_enc = [input_size]
        for _ in range(num_l):
            self.h_size_enc.append(dim_h)
            dim_h = int(dim_h * coef)
        self.h_size_enc.append(dim_o)
            
        self.h_size_dec = []    
        self.h_size_dec.extend(list(reversed(self.h_size_enc)))
        self.act = nn.LeakyReLU(0.01)
        
        self.enc = make_mlp(self.h_size_enc, self.act, residual=False)
        self.dec = make_mlp(self.h_size_dec, self.act, residual=False)
        self.drop = nn.Dropout(p=cfg.FE.P_DROPOUT)
        self.noise = FeatureReversalNoise()
        self.gaussian_dropout = CoupledGaussianDropout()
        

    def forward(self, x):
        feature = self.encode(x)
        x_ = self.decode(feature)

        return feature, x_
        
    def encode(self, x):
        #x = self.noise(x)
        x = self.gaussian_dropout(x)
        #x = self.drop(x)
        
        feature = self.enc(x)
        return feature

    def decode(self, feature):        
        x_ = self.dec(feature)
        return x_
    
    
def make_mlp(h_size: List, act: nn.Module, residual: bool):
    layers = []
    for k in range(len(h_size)-2):
        module = nn.Sequential(
            nn.Linear(h_size[k], h_size[k+1]),
            act)
        
        if residual and h_size[k] == h_size[k+1]:
            layers.append(Residual(module))
        else:
            layers.append(module)
    layers.append(nn.Linear(h_size[-2], h_size[-1]))
    layers = nn.Sequential(*layers)
    return layers


class Residual(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs


class FeatureReversalNoise(nn.Module):
    """
    feature reversal noise from follows.
    https://forum.numer.ai/t/feature-reversing-input-noise/1416
    """
    def __init__(self, p=0.25):
        super(FeatureReversalNoise, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p

    def forward(self, x):
        if self.training:
            binomial = torch.distributions.binomial.Binomial(probs=1-self.p)
            noise = 2*binomial.sample((1,1,x.shape[2])) - 1
            return x * noise.cuda()
        else:
            return x


class CoupledGaussianDropout(nn.Module):
    """
    gaussian dropout from follows.
    https://forum.numer.ai/t/feature-reversing-input-noise/1416
    """
    def __init__(self, alpha=1.0):
        super(CoupledGaussianDropout, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        if self.training:
            stddev = torch.sqrt(torch.clamp(torch.abs(x), min=1e-6)).detach()
            epsilon = torch.randn_like(x) * self.alpha

            epsilon = epsilon * stddev

            return x + epsilon
        else:
            return x


def choose_loss(loss_name):
    if loss_name == "l1":
        return nn.L1Loss()
    elif loss_name == "mse":
        return nn.MSELoss()
    elif loss_name == "bce":
        return nn.BCELoss()
    elif loss_name == "corr":
        return corr_loss
    else:
        raise ValueError(f"unknown loss function {loss_name}")


def choose_activation(activ_name):
    if activ_name == "relu":
        return nn.ReLU()
    elif activ_name == "leaky-relu":
        return nn.LeakyReLU(0.01)
    elif activ_name == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"unknown activation fuction {activ_name}")


# use correlation as the measure of fit
def corr_loss(pred, target):
    
    pred_n = pred - pred.mean(dim=1)
    pred_n = pred_n / pred_n.norm(dim=1)

    target_n = target - target.mean(dim=1)
    target_n = target_n / target_n.norm(dim=1)

    pred_n = pred_n.squeeze()
    target_n = target_n.squeeze()
    l = torch.inner(pred_n, target_n)
    return torch.mean(1-l)
