"""P18: 신경망(MLP·LSTM) 5-시드 재훈련 — '신경망 실패' 주장의 시드 강건성.

E10 미실시 목록의 항목. p2_forecast_race.py와 동일한 확장 윈도우·피처·평가에서
MLP·LSTM만 시드 {0, 7, 13, 21, 42}로 재훈련해 QLIKE·표본외 R²의 시드 분산을 본다.
주장: §6 "신경망은 실패한다"가 seed=42 단일 시드의 우연이 아님.

출력: output/tables/p18_nn_seeds.csv
게이트: 5시드 전부에서 MLP·LSTM의 QLIKE > RW126(0.334) 이면 PASS(실패 주장 강건).
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/mnt/20t/졸업논문")
ds = pd.read_csv(ROOT / "data/processed/ml_dataset.csv", index_col=0)
ds.index = pd.PeriodIndex(ds.index, freq="M")
ds = ds.dropna()

FEATURES = [c for c in ds.columns if not c.startswith("tgt_")]
y_var = ds["tgt_var_next"].values
y_log = np.log(y_var)
n = len(ds)
FIRST_TRAIN, RETRAIN = 120, 12
ml_X = ds[FEATURES].values

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

def smear(pred_log, resid):
    return np.exp(pred_log + resid.var() / 2.0)

class LSTMNet(nn.Module):
    def __init__(self, d, h=24):
        super().__init__()
        self.lstm = nn.LSTM(d, h, batch_first=True)
        self.head = nn.Linear(h, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)

SEQ = 12

def fit_lstm(tr_end, seed):
    torch.manual_seed(seed)
    sc = StandardScaler().fit(ml_X[:tr_end])
    Z = sc.transform(ml_X)
    seqs = np.stack([Z[i - SEQ:i] for i in range(SEQ, n)])
    idx = np.arange(SEQ, n)
    tr_mask = idx < tr_end
    Xtr = torch.tensor(seqs[tr_mask], dtype=torch.float32)
    ytr = torch.tensor(y_log[idx[tr_mask]], dtype=torch.float32)
    net = LSTMNet(Z.shape[1]); opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    for _ in range(150):
        opt.zero_grad(); loss = nn.functional.mse_loss(net(Xtr), ytr); loss.backward(); opt.step()
    with torch.no_grad():
        resid = (ytr - net(Xtr)).numpy()
    def predict(te):
        te = np.asarray(te)
        ok = te >= SEQ
        out = np.full(len(te), np.nan)
        if ok.any():
            Xte = torch.tensor(seqs[te[ok] - SEQ], dtype=torch.float32)
            with torch.no_grad():
                out[ok] = smear(net(Xte).numpy(), resid)
        return out
    return predict

def qlike(f, a):
    r = a / f
    return r - np.log(r) - 1.0

SEEDS = [0, 7, 13, 21, 42]
oos = np.arange(FIRST_TRAIN, n)
bench = y_var[oos]
# 표본외 R² 벤치마크: p2와 동일(ExpandingMean)
em = np.full(n, np.nan)
for t0 in range(FIRST_TRAIN, n, RETRAIN):
    te = np.arange(t0, min(t0 + RETRAIN, n))
    em[te] = y_var[:t0].mean()
sse_bench = ((bench - em[oos]) ** 2).sum()

rows = []
for seed in SEEDS:
    np.random.seed(seed)
    F = {"MLP": np.full(n, np.nan), "LSTM": np.full(n, np.nan)}
    for t0 in range(FIRST_TRAIN, n, RETRAIN):
        tr = np.arange(t0)
        te = np.arange(t0, min(t0 + RETRAIN, n))
        sc = StandardScaler().fit(ml_X[tr]); Z = sc.transform(ml_X)
        m = MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-2,
                         max_iter=3000, random_state=seed).fit(Z[tr], y_log[tr])
        resid = y_log[tr] - m.predict(Z[tr])
        F["MLP"][te] = smear(m.predict(Z[te]), resid)
        F["LSTM"][te] = fit_lstm(t0, seed)(te)
    for name in ("MLP", "LSTM"):
        f = F[name][oos]; ok = np.isfinite(f)
        ql = qlike(f[ok], bench[ok]).mean()
        r2 = 1 - ((bench[ok] - f[ok]) ** 2).sum() / sse_bench
        rows.append(dict(model=name, seed=seed, QLIKE=ql, OOS_R2=r2 * 100, n=int(ok.sum())))
        print(f"seed={seed} {name}: QLIKE={ql:.3f} R2={r2*100:.1f}%", flush=True)

res = pd.DataFrame(rows)
summ = res.groupby("model").agg(QLIKE_mean=("QLIKE", "mean"), QLIKE_min=("QLIKE", "min"),
                                QLIKE_max=("QLIKE", "max"), R2_mean=("OOS_R2", "mean"),
                                R2_min=("OOS_R2", "min"), R2_max=("OOS_R2", "max"))
out = ROOT / "output/tables/p18_nn_seeds.csv"
res.to_csv(out, index=False)
print("\n===== P18 NN 시드 강건성 =====")
print(summ.round(3).to_string())

RW126_QLIKE = 0.334   # p2_leaderboard 고정값
worst = res.groupby("model").QLIKE.min()
gate = (worst > RW126_QLIKE).all()
print(f"\n[게이트] 5시드 전부 QLIKE > RW126({RW126_QLIKE})? "
      f"MLP min={worst['MLP']:.3f}, LSTM min={worst['LSTM']:.3f} → {'PASS (실패 주장 강건)' if gate else 'FAIL (시드 의존!)'}")
