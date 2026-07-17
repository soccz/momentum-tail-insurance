# -*- coding: utf-8 -*-
"""
STEP 1. 논문 핵심 결과 재현 — 실제 데이터에서 표 1·표 3을 처음부터 끝까지
=========================================================================
이 스크립트 하나가 논문의 뼈대를 재현합니다:

  [A] 한국 모멘텀 포트폴리오(WML)를 원시 주가에서 직접 구축      (논문 §4.1)
  [B] 표 1: WML 샤프 0.17 · 왜도 −1.17 · 최악월 1998-10 −94.9%   (논문 §3.3)
  [C] BS15 위험관리: σ타깃/σ예측 스케일링                        (논문 §4.2, 식 5·6)
  [D] 표 3: WML* 샤프 0.26 · IR 0.23 · 최악월 −20%               (논문 §5.3)
  [E] 검증 게이트: 위 수치가 허용오차 안이면 PASS

사용하는 것: pandas와 numpy뿐. 머신러닝 없음. 함수 2개, 나머지는 위→아래 직진.
실행:  python3 reproduce/step1_reproduce.py     (소요 약 2~4분)

논문과의 관계: 이 코드는 code/build_factor.py(레퍼런스 파이프라인)와 같은
알고리즘을 교육용으로 풀어쓴 것입니다. 수치는 게이트 오차 안에서 일치합니다.
"""
import numpy as np
import pandas as pd

# ════════════════════════════════════════════════════════════════════
# [A-1] 데이터 로드
# ────────────────────────────────────────────────────────────────────
# mom_prices.parquet: 한 행 = (종목, 날짜, 수정주가, 시가총액).
# "수정주가"란 액면분할·배당 등을 소급 보정한 가격 — 수익률 계산의 표준.
# 상장폐지 종목도 상폐일까지 들어있다(생존편향 제거 — 논문 §3.1의 핵심).
# ════════════════════════════════════════════════════════════════════
SRC = "/mnt/20t/study/mom_paper_test/data/external/mom_prices.parquet"
print("[1/7] 데이터 로드 …")
df = pd.read_parquet(SRC, columns=["Date", "Code", "AdjClose", "MarketCap"])
df["Date"] = pd.to_datetime(df["Date"])

# 유효한 가격만 남긴다 (결측·0원 제거). 논문 §3.1과 동일한 필터.
df = df.dropna(subset=["AdjClose"])
df = df[df["AdjClose"] > 0]

# ════════════════════════════════════════════════════════════════════
# [A-2] "와이드" 표로 변환: 행=날짜, 열=종목
# ────────────────────────────────────────────────────────────────────
# 이렇게 하면 "어느 날 모든 종목의 가격"이 한 행이 되어 계산이 단순해진다.
# ════════════════════════════════════════════════════════════════════
price = df.pivot_table(index="Date", columns="Code", values="AdjClose", aggfunc="last").sort_index()
mcap  = df.pivot_table(index="Date", columns="Code", values="MarketCap", aggfunc="last").reindex_like(price)

# 표본 시작: 1990년. (1989년 이전은 원자료 품질이 나빠 사용하지 않음 — E27 감사)
price = price.loc[price.index >= "1990-01-01"]
mcap  = mcap.loc[price.index]
print(f"      {price.shape[0]:,}거래일 × {price.shape[1]:,}종목, "
      f"{price.index.min().date()} ~ {price.index.max().date()}")

# ════════════════════════════════════════════════════════════════════
# [A-3] 일별 수익률 + 이상치 필터
# ────────────────────────────────────────────────────────────────────
# 수익률 = 오늘가격/어제가격 − 1.
# |수익률| > 50%는 수정주가 산정 오류일 가능성이 높아 결측 처리(논문 §3.1).
# 이 필터를 없애도 결론이 유지됨은 논문 §9.3·부록 B에서 확인됨.
# ════════════════════════════════════════════════════════════════════
ret = price.pct_change(fill_method=None)
ret = ret.where(ret.abs() <= 0.50)

# ════════════════════════════════════════════════════════════════════
# [A-4] 매월 말: 모멘텀 신호로 10분위 → 승자·패자 포트폴리오
# ────────────────────────────────────────────────────────────────────
# 신호(논문 §4.1): t−12개월 말 → t−1개월 말의 누적수익률.
#   · 직전 1개월(t월)은 제외 = "최근월 스킵" (단기반전 회피, Jegadeesh–Titman)
# 승자 = 신호 상위 10%, 패자 = 하위 10%. 각 분위 안은 시가총액 가중(VW).
# 보유: 다음 1개월 동안 buy-and-hold (중간 매매 없음, 가중치 자연 표류).
# ════════════════════════════════════════════════════════════════════
month_end = pd.DatetimeIndex(                      # 매월의 마지막 거래일 목록
    pd.Series(price.index, index=price.index).groupby(
        [price.index.year, price.index.month]).last().values)

def leg_return_daily(daily_ret, w0):
    """한 다리(승자 or 패자)의 보유월 일별 수익률 — buy-and-hold 가치가중.

    daily_ret: 보유월의 (일×구성종목) 수익률.  w0: 형성일 가중치(합=1).
    각 종목의 가치 = w0 × (1+r)의 누적곱. 오늘 수익 = Σ(어제가치×오늘수익)/Σ(어제가치).
    수익률이 결측인 날(거래정지·상폐 후)은 그 종목을 분자·분모에서 제외
    → 남은 종목이 자동으로 재정규화된다. 이것이 상폐 처리 'del-a'(가치 동결).
    """
    growth = (1 + daily_ret.fillna(0)).cumprod()      # 각 종목 가치의 누적 성장
    value_end = growth.mul(w0, axis=1)                # 매일 장마감 가치
    value_prev = value_end.shift(1)
    value_prev.iloc[0] = w0.values                    # 첫날의 '어제 가치' = 형성 가중치
    valid = daily_ret.notna()
    num = (value_prev.where(valid) * daily_ret).sum(axis=1)
    den = value_prev.where(valid).sum(axis=1)
    return (num / den).where(den > 0)

print("[2/7] 월별 포트폴리오 형성 (약 420개월 루프) …")
daily_parts, monthly_rows = [], []
for k in range(12, len(month_end) - 1):
    T = month_end[k]                                  # 형성일 = t월 말
    signal = price.loc[month_end[k - 1]] / price.loc[month_end[k - 12]] - 1  # §4.1 신호
    cap_T = mcap.loc[T]
    eligible = signal.notna() & cap_T.notna() & (cap_T > 0)   # 신호·시총 둘 다 있어야
    s = signal[eligible]
    if len(s) < 30:                                   # 10분위를 만들기엔 종목이 너무 적음
        continue
    decile = pd.qcut(s, 10, labels=False, duplicates="drop")  # 0=패자 … 9=승자
    winners = s.index[decile == decile.max()]
    losers  = s.index[decile == decile.min()]

    hold_days = price.index[(price.index > T) & (price.index <= month_end[k + 1])]
    if len(hold_days) == 0:
        continue
    r_hold = ret.reindex(hold_days)

    w_win = cap_T[winners] / cap_T[winners].sum()     # 시가총액 가중치
    w_los = cap_T[losers]  / cap_T[losers].sum()
    rW = leg_return_daily(r_hold[list(winners)], w_win)
    rL = leg_return_daily(r_hold[list(losers)],  w_los)
    daily_parts.append(pd.DataFrame({"win": rW, "los": rL}))

    # 월별 WML = 승자 월복리 − 패자 월복리 (Kenneth French 팩터 규약, 논문 §4.1)
    monthly_rows.append({
        "month": month_end[k + 1].to_period("M"),
        "wml": (1 + rW.dropna()).prod() - 1 - ((1 + rL.dropna()).prod() - 1),
    })

daily = pd.concat(daily_parts).sort_index()
daily = daily[~daily.index.duplicated()]
daily["wml"] = daily["win"] - daily["los"]            # 일별 WML = 일별 승자−패자
wml_m = pd.DataFrame(monthly_rows).set_index("month")["wml"]
print(f"      월별 WML {len(wml_m)}개월, 일별 WML {len(daily):,}일")

# ════════════════════════════════════════════════════════════════════
# [B] 표 1 통계 — "약하지만 위험한 팩터"
# ════════════════════════════════════════════════════════════════════
print("[3/7] 표 1 통계 …")
ann_mean  = wml_m.mean() * 12                         # 연율 평균
ann_std   = wml_m.std() * np.sqrt(12)                 # 연율 표준편차
sharpe    = ann_mean / ann_std                        # 샤프비율 (WML은 자기자금조달 → RF 차감 불필요)
skew      = wml_m.skew()                              # 왜도 (음수 = 왼쪽 꼬리)
kurt      = wml_m.kurt()                              # 초과첨도 (클수록 극단값 빈발)
worst_val = wml_m.min()
worst_mon = wml_m.idxmin()

print(f"      WML: 샤프 {sharpe:.3f} · 왜도 {skew:.2f} · 초과첨도 {kurt:.1f} · "
      f"최악월 {worst_mon} ({worst_val:.1%})")

# ════════════════════════════════════════════════════════════════════
# [C] BS15 위험관리 (논문 §4.2, 식 5·6)
# ────────────────────────────────────────────────────────────────────
# 식(5)  σ̂²_t = 21 × (1/126) Σ_{j=0..125} r²_{t−1월 마지막 거래일−j}
#        → "직전 126거래일(≈6개월)의 일별 WML 제곱합"으로 다음 달 월분산을 추정.
#        정보는 t−1월 말까지만 사용 → 미래 정보 유출(look-ahead) 없음.
# 식(6)  w_t = σ_target / σ̂_t,  σ_target = 연 12%
#        → 위험이 높게 예측되면 포지션 축소, 낮으면 확대.
# ════════════════════════════════════════════════════════════════════
print("[4/7] 실현변동성 스케일링 …")
r2 = daily["wml"] ** 2                                # 일별 WML의 제곱
sigma2_m = 21 * r2.rolling(126).mean()                # 식(5): 월분산 추정치 (일 단위 시계열)

scale = {}
for m in wml_m.index:
    last_day_prev = daily.index[daily.index <= (m - 1).to_timestamp("M")]  # t−1월 말까지
    if len(last_day_prev) == 0 or pd.isna(sigma2_m.get(last_day_prev[-1], np.nan)):
        continue
    sigma_ann = np.sqrt(12 * sigma2_m[last_day_prev[-1]])  # 월분산 → 연율 변동성
    scale[m] = 0.12 / sigma_ann                            # 식(6): 연 12% 타깃
scale = pd.Series(scale)

common = wml_m.index.intersection(scale.index)        # 스케일 가능한 월만
wml_star = wml_m[common] * scale[common]              # WML* = 스케일 × WML
wml_base = wml_m[common]

# ════════════════════════════════════════════════════════════════════
# [D] 표 3 통계 — 위험관리의 효과
# ════════════════════════════════════════════════════════════════════
print("[5/7] 표 3 통계 …")
star_sharpe = wml_star.mean() * 12 / (wml_star.std() * np.sqrt(12))
star_worst  = wml_star.min()
star_kurt   = wml_star.kurt()

# 정보비율(IR): 두 전략을 각자 변동성으로 정규화한 뒤의 차이가 얼마나 꾸준한가.
# (목표변동성 12%라는 선택에 영향받지 않는 비교 — 논문 표 3 주석)
d = wml_star / wml_star.std() - wml_base / wml_base.std()
ir = d.mean() / d.std() * np.sqrt(12)

print(f"      WML*: 샤프 {star_sharpe:.3f} · 최악월 {star_worst:.1%} · "
      f"초과첨도 {star_kurt:.1f} · IR {ir:.3f}")

# ════════════════════════════════════════════════════════════════════
# [E] 검증 게이트 — 논문 헌법 §3의 고정값과 대조
# ════════════════════════════════════════════════════════════════════
print("[6/7] 검증 게이트 …\n")
gates = [
    # (항목,            재현값,        기준값,   허용오차,  비교방식)
    ("WML 샤프",        sharpe,       0.165,    0.02,     "abs"),
    ("WML 왜도",        skew,         -1.16,    0.10,     "abs"),
    ("WML 초과첨도",     kurt,         10.5,     1.0,      "abs"),
    ("최악월 = 1998-10", str(worst_mon), "1998-10", None,   "eq"),
    ("최악월 수익률",     worst_val,    -0.949,   0.01,     "abs"),
    ("WML* 샤프",       star_sharpe,  0.26,     0.02,     "abs"),
    ("WML* IR",         ir,           0.23,     0.03,     "abs"),
]
all_pass = True
print(f"      {'항목':<16}{'재현값':>10}{'기준값':>10}   판정")
for name, got, want, tol, mode in gates:
    if mode == "eq":
        p = (got == want)
        print(f"      {name:<16}{got:>10}{want:>10}   {'PASS' if p else 'FAIL'}")
    else:
        p = abs(got - want) <= tol
        print(f"      {name:<16}{got:>10.3f}{want:>10.3f}   {'PASS' if p else 'FAIL'}")
    all_pass &= p

print()
print("[7/7] " + ("✅ 전체 PASS — 논문 표 1·표 3이 원시 데이터에서 재현되었습니다."
                  if all_pass else "❌ FAIL 항목 있음 — 데이터/환경을 확인하세요."))
