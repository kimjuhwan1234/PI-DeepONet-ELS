# -*- coding: utf-8 -*-
"""stage-2 가우시안 NLL 신뢰구간 하이브리드 (OptionNet, Lin et al. 2023 의 CI 요소 이식).
 기존 deeponet_hybrid_s2don 과 동일한 stage1 MC 앵커 + DeepONet stage2 이되,
 stage2 가 평균 μ + 로그분산 logσ² 를 함께 예측(가우시안 NLL)해 [μ−2s, μ+2s] 신뢰구간을 출력.
 논문의 MRC(멀티스케일 conv)/LSTM 은 횡단면 ELS 데이터에 시계열 축이 없어 제외; CI 만 이식.

 산출: run(D,cfg) -> {"deeponet_hybrid_ci": DataFrame}
   컬럼: ITEM_CD, isu_ord, y_true, y_pred, mc_true, mc_pred, resid_true, resid_pred,
         resid_std, y_std, y_lo, y_hi   (하이브리드 8 + CI 4)
   기존 평가/그림 코드는 여분 컬럼을 무시하므로 안전. CI 리포팅(커버리지 p·밴드)은 이 CSV 로 별도 수행(task#2).
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from module.data import to_tensor, zstats, time_weights
from module.networks import mlp
from module.train import _EarlyStop, _opt, train_curve
from module.pipeline import hybrid_residual_target
from util import file_manager as fm

LAM = 1e-6        # 논문 식17 의 분산 하한 floor: logσ² >= log λ
LV_MAX = 10.0     # 로그분산 상한 (수치 안정)
Z = 2.0           # 신뢰구간 배수 (≈95%, 논문 [μ±2s])


class MarginOperatorNLL(nn.Module):
    """stage-2 잔차 DeepONet(내적) 2-헤드: 평균 μ, 로그분산 logσ².
     branch=vol·corr·sig_eff, trunk=[곡선|계약]. MarginOperator 를 2-헤드로 확장."""
    def __init__(self, nb, nt, P):
        super().__init__()
        self.b_mu = mlp(nb, P); self.t_mu = mlp(nt, P); self.b0_mu = nn.Parameter(torch.zeros(1))
        self.b_lv = mlp(nb, P); self.t_lv = mlp(nt, P); self.b0_lv = nn.Parameter(torch.zeros(1))

    def forward(self, bx, tx):
        mu = (self.b_mu(bx) * self.t_mu(tx)).sum(-1) + self.b0_mu
        lv = (self.b_lv(bx) * self.t_lv(tx)).sum(-1) + self.b0_lv
        return mu, lv


def _nll_loss(mu, lv, y, w=None):
    """가우시안 음의 로그우도(표준화 공간). lv=logσ² 를 [log LAM, LV_MAX] 로 clamp(식17 floor).
     w: 표본가중(없으면 균등). 상수항 0.5·log(2π) 는 최적화에 무관해 생략."""
    lvc = lv.clamp(float(np.log(LAM)), LV_MAX)
    per = 0.5 * (lvc + (y - mu) ** 2 * torch.exp(-lvc))
    if w is None:
        return per.mean()
    return (per * w).sum() / w.sum()
