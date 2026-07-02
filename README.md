# Physics-Informed DeepONet for Autocallable ELS Fair-Value Pricing

Operator-learning surrogates — **DeepONet** and **Physics-Informed DeepONet (PI-DeepONet)** — for the
fair-value pricing of Korean **3-underlying worst-of autocallable Equity-Linked Securities (ELS)**,
trained on regulatory disclosures (DART) combined with market data, and evaluated under a
leakage-free walk-forward protocol against strong tabular machine-learning baselines.

---

## Abstract

Autocallable ELS are among the most widely issued structured products in Korea, yet their reported
**fair value** embeds both an arbitrage-free (no-arbitrage) component and an issuer-specific margin
that drifts over time and across market regimes. Pure Monte-Carlo (MC) valuation captures the former
but not the latter, while black-box regression on fair value alone overfits temporal artifacts and
fails out-of-sample. We cast ELS valuation as an **operator-learning problem** and study a family of
DeepONet surrogates that map contract- and market-condition functions to price. Our central proposal
is a **hybrid decomposition**

> `FAIR = MC_hat (physics anchor) + recent_margin (anchor) + residual_hat (margin model)`,

where the arbitrage-free anchor `MC_hat` is produced by a (PI-)DeepONet whose training is biased
toward the **Black-Scholes PDE** `-V_τ + ½σ²S²V_SS + rSV_S - rV = 0` and the worst-of autocall/knock-in
terminal condition via soft-penalty physics losses, and the issuer margin is captured by a gradient-boosted
residual model. On 23,151 3-star Korean ELS contracts under an expanding-window walk-forward split
(test = future 20%), the hybrid operator models attain **R² ≈ 0.72–0.73** and Spearman ρ ≈ 0.86,
**substantially outperforming direct fair-value regression** (Ridge 0.56; gradient boosting 0.26–0.32;
direct DeepONet 0.06). The physics-informed anchor additionally yields an arbitrage-consistent price
surface with interpretable Greeks. The full pipeline — data construction, training, evaluation, and
provenance/consistency validation — is reproducible from four top-level notebooks.

---

## Highlights

- **Operator learning for structured-product pricing.** DeepONet / PI-DeepONet map the yield curve and
  contract parameters to a full price surface `V(S, τ, I)` (I = knock-in state), rather than a single scalar.
- **Physics as regularization.** The PI-DeepONet imposes the Black-Scholes PDE residual and the worst-of
  autocall + knock-in terminal payoff through automatic differentiation, following Wang et al. (2021).
- **Hybrid anchor + margin decomposition.** Separating the no-arbitrage price from the issuer margin lifts
  out-of-sample R² from negative/low (direct regression) to ~0.72, and the gain is robust to the choice of anchor.
- **Leakage-free evaluation.** Expanding-window walk-forward with train-only preprocessing and time-decayed
  sample weights; random splits are shown (in prior iterations) to inflate scores via issuer×year memorization.
- **Two branch architectures compared.** A compact MLP branch and a 1D-CNN + FiLM *curve* branch that ingests
  the full US-Treasury yield curve; each is provided in **direct** and **hybrid** modes.
- **Fully reproducible.** Single `config.yaml`, centralized path management, and a validation notebook that
  checks (a) dataset provenance against the source cache and (b) numerical agreement with the prior benchmark.

---

## Results (walk-forward OOS, 3-star Korean ELS, n = 23,151)

| Model | Family | Mode | R² | MAE | Spearman ρ |
|---|---|---|---:|---:|---:|
| **DeepONet (MLP)**        | DeepONet     | hybrid | **0.727** | 0.0210 | 0.864 |
| **PI-DeepONet (MLP)**     | PI-DeepONet  | hybrid | **0.724** | 0.0213 | 0.866 |
| **DeepONet-Curve**        | DeepONet     | hybrid | 0.721 | 0.0210 | 0.858 |
| Ridge                     | tabular      | direct | 0.561 | 0.0269 | 0.762 |
| XGBoost                   | tabular      | direct | 0.320 | 0.0352 | 0.539 |
| CatBoost                  | tabular      | direct | 0.316 | 0.0357 | 0.532 |
| LightGBM                  | tabular      | direct | 0.311 | 0.0353 | 0.540 |
| GBM (HistGB)              | tabular      | direct | 0.257 | 0.0371 | 0.504 |
| DeepONet-Curve            | DeepONet     | direct | 0.056 | 0.0394 | 0.433 |
| DeepONet (MLP)            | DeepONet     | direct | −0.239 | 0.0493 | −0.047 |

**Per-stage R² (hybrids).** Stage-1 (MC reproduction) 0.92–0.96 · Stage-2 (residual margin) 0.49 ·
Final (FAIR) 0.72–0.73. The hybrids cluster tightly regardless of anchor, indicating that the
*decomposition structure* — arbitrage-free anchor + recent-margin + learned residual — is the decisive
factor, while the physics anchor's distinctive value lies in arbitrage consistency and Greeks rather than
in the headline R². Metrics: `result/statistics/`; figures: `result/image/`.

---

## Repository structure

```
config.yaml            # single source of all hyperparameters
0_data.ipynb           # build 3 modeling datasets -> data/
1_run.ipynb            # train all models (walk-forward) -> result/predictions/
2_evaluate.ipynb       # metrics + figures -> result/statistics/, result/image/
3_validate.ipynb       # (A) dataset-provenance check  (B) result match vs prior benchmark

model/                 # deeponet.py, deeponet_curve.py, pi_deeponet.py, benchmark.py
module/                # data, networks, physics (BS-PDE + RBF tools), tabular, train, pipeline
util/                  # file_manager (paths), utils (config/seed/device), metric, plot
data/                  # ml.parquet, deeponet.parquet, deeponet_curve.parquet (+ cache/, raw/*)
result/                # predictions/, image/, statistics/
paper/                 # reference papers [1]-[5]
```

Datasets: **`ml.parquet`** (tabular benchmark + hybrid margin stage), **`deeponet.parquet`**
(MLP branch), **`deeponet_curve.parquet`** (yield-curve + FiLM branch). All three carry `fair`, `mc`,
and `recent_margin`, supporting both direct and hybrid modes. Raw DART source CSVs live in `data/raw/`
and are **git-ignored** (large); the feature-complete source is `data/cache/els3_dataset.parquet`.

---

## Reproduction

```bash
pip install -r requirements.txt          # Python 3.11; see note on PyTorch/CUDA in the file
# then run the notebooks in order:
jupyter nbconvert --to notebook --execute --inplace 0_data.ipynb     # -> data/*.parquet
jupyter nbconvert --to notebook --execute --inplace 1_run.ipynb      # -> result/predictions/
jupyter nbconvert --to notebook --execute --inplace 2_evaluate.ipynb # -> result/{image,statistics}/
jupyter nbconvert --to notebook --execute --inplace 3_validate.ipynb # provenance + result check
```

All hyperparameters (seed, iterations, learning rate, tree parameters, walk-forward boundaries, RBF grid)
are edited in **`config.yaml`** only. Paths are resolved through **`util/file_manager.py`**.

---

## Methodology in brief

- **Product.** Worst-of, step-down autocallable with knock-in (KI); terminal payoff
  `1 + coupon·tenor` if the worst performer is above the redemption barrier, else `1` (no KI) or `S` (KI).
- **Physics.** Black-Scholes PDE residual and terminal conditions for `I ∈ {0,1}` imposed as soft penalties;
  derivatives via `torch.autograd`. An **RBF differentiation-matrix** variant (fixed collocation grid,
  inverse-quadratic kernel) is retained in `module/physics.py` for autograd-vs-RBF studies.
- **Anchoring & margin.** `recent_margin` is a causal 90-day mean of `(FAIR − MC)`; the residual
  `(FAIR − MC) − recent_margin` is learned by XGBoost on contract features.
- **Evaluation.** Expanding-window walk-forward; train-only fit of all scalers/encoders; time-decayed weights;
  metrics R², MAE, RMSE, and Spearman ρ (rank quality — the operative criterion for relative-value pricing).

---

## References

The methods build on the following works (PDFs in `paper/`):

1. L. Lu, P. Jin, G. Pang, Z. Zhang, G. E. Karniadakis. *Learning nonlinear operators via DeepONet based on
   the universal approximation theorem of operators.* **Nature Machine Intelligence** 3(3):218–229, 2021.
2. S. Wang, H. Wang, P. Perdikaris. *Learning the solution operator of parametric partial differential
   equations with physics-informed DeepONets.* **Science Advances** 7(40):eabi8605, 2021.
3. S. Lee, J. Huh, S. Jeong. *DeepONet-based surrogate modeling for bond option pricing.*
   **AIMS Mathematics** 11(3):5853–5896, 2026.
4. P. C. Andreou, C. Han, N. Li. *Stock options pricing via machine learning methods combined with firm
   characteristics.* Working paper, 2023.
5. A. Jiao, Q. Yan, J. Harlim, L. Lu. *Solving forward and inverse PDE problems on unknown manifolds via
   physics-informed neural operators.* **arXiv preprint**, 2024.

```bibtex
@article{lu2021learning,
  author  = {Lu, Lu and Jin, Pengzhan and Pang, Guofei and Zhang, Zhongqiang and Karniadakis, George Em},
  title   = {Learning nonlinear operators via {DeepONet} based on the universal approximation theorem of operators},
  journal = {Nature Machine Intelligence},
  volume  = {3}, number = {3}, pages = {218--229}, year = {2021},
  doi     = {10.1038/s42256-021-00302-5}
}

@article{wang2021learning,
  author  = {Wang, Sifan and Wang, Hanwen and Perdikaris, Paris},
  title   = {Learning the solution operator of parametric partial differential equations with physics-informed {DeepONets}},
  journal = {Science Advances},
  volume  = {7}, number = {40}, pages = {eabi8605}, year = {2021},
  doi     = {10.1126/sciadv.abi8605}
}

@article{lee2026deeponet,
  author  = {Lee, Sanghyun and Huh, Jeonggyu and Jeong, Seungwon},
  title   = {{DeepONet}-based surrogate modeling for bond option pricing},
  journal = {AIMS Mathematics},
  volume  = {11}, number = {3}, pages = {5853--5896}, year = {2026},
  doi     = {10.3934/math.2026242}
}

@article{andreou2023stock,
  author  = {Andreou, Panayiotis C. and Han, Chulwoo and Li, Nan},
  title   = {Stock options pricing via machine learning methods combined with firm characteristics},
  year    = {2023},
  note    = {Working paper}
}

@article{jiao2024solving,
  author  = {Jiao, Anran and Yan, Qile and Harlim, John and Lu, Lu},
  title   = {Solving forward and inverse {PDE} problems on unknown manifolds via physics-informed neural operators},
  journal = {arXiv preprint},
  year    = {2024}
}
```
