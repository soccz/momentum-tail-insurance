"""
P12 — 무위험수익률 실질화 (심사자 must-fix #7).

문제: RF≈0 가정이 90년대 한국(콜금리 10~20%)에서 시장 샤프를 부풀림 → C1("시장의 절반") 오염.
해법: 한국은행 콜금리(BOK/OECD 경유 FRED IRSTCI01KRM156N, 1991-01~) → RMRF를 진짜 초과수익으로.
      CD 3개월(IR3TIB01KRM156N)은 강건성. WML·SMB·HML은 self-financing이라 불변.
게이트: RF가 표본 전 기간 커버 · 초과수익 재계산 후 순위 논리 일관.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT=Path("/mnt/20t/졸업논문")
sr=lambda r: np.sqrt(12)*r.mean()/r.std()

# ---- RF 저장 (월별 소수) ----
def load_rf(fp, col):
    d=pd.read_csv(fp, parse_dates=["observation_date"])
    d["month"]=d["observation_date"].dt.to_period("M")
    d["rf_m"]=(1+d[col]/100)**(1/12)-1
    return d.set_index("month")[["rf_m"]].rename(columns={"rf_m":col})
rf=load_rf("/tmp/rf_IRSTCI01KRM156N.csv","IRSTCI01KRM156N").join(
   load_rf("/tmp/rf_IR3TIB01KRM156N.csv","IR3TIB01KRM156N"))
rf.columns=["rf_call","rf_cd3m"]
rf.to_csv(ROOT/"data/processed/rf_monthly.csv")
print(f"[rf] {rf.index[0]}~{rf.index[-1]}  콜금리 연평균 {((1+rf.rf_call)**12-1).mean()*100:.2f}%  "
      f"(90년대 {((1+rf.loc['1991':'1999','rf_call'])**12-1).mean()*100:.1f}%)")

# ---- 시장 초과수익 재계산 ----
ffm=pd.read_csv(ROOT/"data/processed/ff_own_monthly.csv",parse_dates=["month"]).set_index("month")
wm=pd.read_csv(ROOT/"data/processed/wml_monthly.csv",parse_dates=["month"]).set_index("month")
ffm.index=ffm.index.to_period("M"); wm.index=wm.index.to_period("M")
df=pd.concat([ffm[["rmrf","smb","hml"]],wm["wml"],rf["rf_call"]],axis=1).dropna()
df["mkt_excess"]=df["rmrf"]-df["rf_call"]

print(f"\n===== Table 1 공통표본({df.index[0]}~{df.index[-1]}, n={len(df)}) — gross vs 초과 =====")
rows={}
for nm,s in [("시장 gross (기존)",df["rmrf"]),("시장 초과 (콜금리 차감)",df["mkt_excess"]),
             ("WML (self-financing=이미 초과)",df["wml"]),("SMB",df["smb"]),("HML",df["hml"])]:
    rows[nm]=dict(mean_ann=s.mean()*1200,sharpe=sr(s),t=s.mean()/(s.std()/np.sqrt(len(s))))
tab=pd.DataFrame(rows).T
print(tab.round(3).to_string())

# OOS 창(2001-12+) 참고
oos=df.loc["2001-12":]
print(f"\n[OOS 2001-12+] 시장 초과 샤프 {sr(oos['mkt_excess']):.3f} (gross {sr(oos['rmrf']):.3f}) · WML {sr(oos['wml']):.3f}")

tab.round(4).to_csv(ROOT/"output/tables/p12_rf_corrected.csv")
gap_old=rows["WML (self-financing=이미 초과)"]["sharpe"]/rows["시장 gross (기존)"]["sharpe"]
gap_new=rows["WML (self-financing=이미 초과)"]["sharpe"]/rows["시장 초과 (콜금리 차감)"]["sharpe"]
print(f"\n[C1 수정] WML/시장 샤프비: gross 기준 {gap_old:.2f} → 초과 기준 {gap_new:.2f}")
assert rf.loc["1991-02":"2026-03"].notna().all().all(), "RF 커버리지 구멍"
print("[GATES] RF 전기간 커버 ✓ · 표 저장 ✓")
