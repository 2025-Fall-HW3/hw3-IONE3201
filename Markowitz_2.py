"""
Package Import
"""
from signal import signal
from unicodedata import name
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=40, gamma=40):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """

        use_pca = True
        n_components = 3
        max_asset_weight = 0.5  # 單一資產上限 50%

        # 防禦模式 A：弱勢時配置的防禦資產
        defensive_assets = ["XLU", "XLP"]

        for i in range(self.lookback + 1, len(self.price)):
            date = self.price.index[i]

            # =========================
            # STEP 0: 市場絕對動量（SPY 200 日）
            # =========================
            '''risk_on = True
            if "SPY" in self.returns.columns and i > 200:
                spy_200_ret = (1 + self.returns["SPY"].iloc[i - 200 : i]).prod() - 1
                if spy_200_ret <= 0:
                    risk_on = False'''

            risk_on = True

            spy_200_ret = (1 + self.returns["SPY"].iloc[i - 200 : i]).prod() - 1
            spy_vol = self.returns["SPY"].iloc[i - 60 : i].std() * np.sqrt(252)

            if (spy_200_ret <= 0) and (spy_vol > 0.25):
                risk_on = False

            

            # =========================
            # 若 risk-off → 防禦模式 A (XLU / XLP)
            # =========================
            if not risk_on:
                weights = np.zeros(len(assets))
                idx_def = [j for j, a in enumerate(assets) if a in defensive_assets]

                if len(idx_def) > 0:
                    # XLU / XLP 等權
                    for j in idx_def:
                        weights[j] = 1.0 / len(idx_def)
                else:
                    # 找不到防禦資產就等權全部
                    weights[:] = 1.0 / len(assets)

                self.portfolio_weights.loc[date, assets] = weights
                continue  # 當天不用再做 MVO，直接下一天

            # =========================
            # STEP 1: 協方差估計（shrinkage + PCA 降噪）
            # =========================
            R_n = self.returns[assets].iloc[i - self.lookback : i]

            Sigma_sample = R_n.cov().values
            variances = np.diag(Sigma_sample)
            Sigma_target = np.diag(variances)
            shrinkage_factor = 0.15
            Sigma = (1 - shrinkage_factor) * Sigma_sample + shrinkage_factor * Sigma_target

            if use_pca:
                try:
                    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
                    idx = eigenvalues.argsort()[::-1]
                    eigenvalues = eigenvalues[idx]
                    eigenvectors = eigenvectors[:, idx]
                    eigenvalues_reduced = eigenvalues[:n_components]
                    eigenvectors_reduced = eigenvectors[:, :n_components]
                    Sigma = eigenvectors_reduced @ np.diag(eigenvalues_reduced) @ eigenvectors_reduced.T
                except Exception:
                    pass

            Sigma += np.eye(len(Sigma)) * 1e-5

            # =========================
            # STEP 2: Multi-timeframe momentum 作為 alpha
            # =========================
            mom_15 = self.returns[assets].iloc[i - 15 : i].mean().values
            mom_30 = self.returns[assets].iloc[i - 30 : i].mean().values
            mom_40 = R_n.mean().values

            momentum_raw = 0.50 * mom_15 + 0.30 * mom_30 + 0.20 * mom_40

            # 再加一個 126-day 絕對動量（較長期趨勢）
            if i > 126:
                R_126 = self.returns[assets].iloc[i - 126 : i]
                abs_mom_126 = (1 + R_126).prod().values - 1
            else:
                abs_mom_126 = np.zeros(len(assets))

            
            # 組合 alpha：短中期 + 長期絕對動量
            momentum_score = 0.7 * momentum_raw + 0.3 * abs_mom_126
            

            # =========================
            # STEP 3: 用波動做 scaling（reward per unit risk）
            # =========================
            vol = R_n.std().values * np.sqrt(252)  # 年化波動
            vol[vol == 0] = 1e-5
            mu_scaled = (momentum_score / vol) * 1.5  # 放大 alpha

            # =========================
            # ⭐ STEP 3.5: 對 XLK 做 growth boost
            # =========================
            asset_names = np.array(assets)
            growth_boost = np.ones(len(assets))
            for j, name in enumerate(asset_names):
                if name == "XLK":
                    # 科技股我看好你
                    # 高看你了以為你會帶我過1
                    # 但你還是很棒還是誇獎你一下
                    mom_XLK = momentum_score[j]

                    if mom_XLK > 0:
                        growth_boost[j] =4.0
                    else:
                        growth_boost[j] = 2.0
                elif name == "XLC":
                    # 消費選擇、通訊也偏成長，給一點但不要太多
                    growth_boost[j] = 0.6
                elif name == "XLY":
                    # 消費選擇、通訊也偏成長，給一點但不要太多
                    mom_XLY = momentum_score[j]
                    if mom_XLY > 0:
                        growth_boost[j] = 1.4
                    #growth_boost[j] = 1.2
            mu_scaled *= growth_boost


            

            inv_vol = 1 / (vol + 1e-6)
            inv_vol /= inv_vol.sum()

            # =========================
            # STEP 4: 選 Top 3 資產做 MVO
            # =========================
            #top_idx = np.argsort(mu_scaled)[-3:]
            signal = mu_scaled * inv_vol     # 動量 × 低風險
            top_idx = np.argsort(signal)[-3:]
            mu_top = mu_scaled[top_idx]

            R_top = R_n.iloc[:, top_idx]
            Sigma_top = R_top.cov().values
            # 再提供一點 shrinkage，避免 3x3 協方差過度極端
            Sigma_top = (1 - 0.1) * Sigma_top + 0.1 * np.diag(np.diag(Sigma_top))
            Sigma_top += np.eye(3) * 1e-5

            # =========================
            # STEP 5: Gurobi 上做 mean-variance 最佳化
            # =========================
            try:
                with gp.Env(empty=True) as env:
                    env.setParam("OutputFlag", 0)
                    env.setParam("DualReductions", 0)
                    env.start()

                    with gp.Model(env=env, name="portfolio") as model:
                        w = model.addMVar(3, name="w", lb=0, ub=max_asset_weight)
                        port_return = mu_top @ w
                        #port_var = w @ Sigma_top @ w
                        port_var = 252 * (w @ Sigma_top @ w) 
                        #reg = 0.01 * (w @ w)
                        reg = 0.001 * (w @ w)

                        model.setObjective(
                            port_return - (self.gamma / 2) * port_var - reg,
                            gp.GRB.MAXIMIZE,
                        )
                        model.addConstr(w.sum() == 1, name="budget")
                        model.optimize()

                        if model.status in [gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL]:
                            weights_top = np.array(
                                [model.getVarByName(f"w[{j}]").X for j in range(3)]
                            )
                        else:
                            weights_top = np.array([0.4, 0.3, 0.3])
            except Exception:
                weights_top = np.array([0.4, 0.3, 0.3])


            # =========================
            # STEP 6: 映射回整個資產空間
            # =========================
            weights = np.zeros(len(assets))
            weights[top_idx] = weights_top
            self.portfolio_weights.loc[date, assets] = weights
        
        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)
