# 재현 스펙 — Barroso & Santa-Clara (2015) 한국 데이터

> 이 문서는 재현의 **모든 방법론 선택을 고정**한다. 코드는 이 스펙을 구현한다.
> 재현 대상: **Fig 1, 2, 3, 6 / Table 1, 3** (사용자 지정).

## 0. 확정된 설계 결정 (2026-07-06, rev2)

> **rev2 (사용자 지시):** ① **20년+ 장기 데이터를 본편**으로 — `mom_prices` 1990–2026(35년)이 본편, `prices_adj` 2005–2025는 clean 부표본. ② **외부 FF3 팩터 파일 완전 폐기** — RMRF·SMB·HML을 한국 원천 주가+PBR에서 **직접** 구성(`build_ff_factors.py`). (게다가 그 FF3 파일은 HML −42%/yr로 손상돼 있었음.)

| 항목 | 결정 | 근거 |
|---|---|---|
| 표본·패널 | **본편 1990–2026** (`mom_prices.parquet`, 35년) + **clean 2005–2025** (`prices_adj.parquet`) | 사용자 지시 "20년+ 장기". 본편은 IMF·닷컴·GFC 크래시 전부 포함, clean은 시장라벨·전컬럼 완비 |
| **팩터 소스** | **한국 원천 데이터만** — RMRF=전종목 VW 시장수익(RF=0), SMB/HML=FF 2×3 자체구성(B/M=1/PBR) | 사용자 지시 "한국 데이터 기반으로만". 외부 FF3 파일 폐기 |
| 무위험 RF | **0** (디스크에 한국 RF 시계열 없음) → RMRF=gross VW 시장수익 | 한국-데이터-온리. WML·SMB·HML은 self-financing이라 무관, RMRF 레벨만 영향(후속 국고채 보강 여지) |
| B/M 소스 | `PBR.xlsx`(FnGuide, 1989+ 일간 PBR, 코스피+코스닥+상폐) → 월말 parquet, B/M=1/PBR | 유일한 장기 장부가 소스(ALL_STOCKS_HISTORY PBR은 2013+만 조밀) |
| 유니버스 | **전종목** (KOSPI+KOSDAQ, survivorship-free, 상폐 포함) | 사용자 결정 "전종목". Barroso "all stocks" |
| 분위 | **10분위 (decile)** | 사용자 결정 "US 본표식". Barroso Table 1·3은 decile |
| 가중 | **가치가중 (value-weight)**, 형성시점 시총 | Barroso 전 표 VW |
| 신호 | 누적수익 t−12 ~ t−2 (**최근 1개월 스킵**) | Barroso "returns from month t−12 to t−2, formation excludes preceding month" |
| 리밸런스 | **월별** (매월 말 형성 → 다음 달 보유) | Barroso monthly |
| WML | D10(winner) − D1(loser), 롱숏 self-financing | Barroso |
| 목표변동성 σ_target | **연 12%** | Barroso. 샤프·IR엔 불변(스케일 상수) |
| RV 창(진단) | 21거래일 (Eq 2) | Fig 2·Table 2용 |
| RV 창(전략) | 126거래일=6개월 (Eq 5) | 스케일링용 |
| 상폐처리 | **del-a**(마지막 유효가에서 제외, 무패널티) 본편 / del-b(−100%) 강건성 | 한국은 delisting return 부재 → del-a 보수적·투명. del-b는 short-leg 이득 과장 위험 |
| 이상치 캡 | 일별 종목수익 \|r\|>0.50 → NaN | 한국 가격제한 ±15%(∼2015.6)/±30% 초과는 데이터 아티팩트 |
| 무위험 RF | ff3의 상수 placeholder 사용, RMRF는 MKT_RF | WML·SMB·HML은 self-financing이라 샤프 무관; RMRF 레벨만 영향(2차적). 후속 국고채 보강 여지 |

**나중에 붙이는 강건성 변형(요청 시):** KOSPI-only, 시총 상위절반 필터, 5분위, 동일가중, del-b, KOSPI-cutoff 브레이크포인트.

## 1. 데이터 소스 (경로 고정)

- 본편 패널: `/mnt/20t/study/mom_paper_test/data/processed/prices_adj.parquet`
  - 컬럼: `Date, Close, Open, High, Low, Volume, MarketCap, Shares, Code, Name, Market, Status, AdjFactor, AdjClose` (일별, 2005-01-03~2025-12-30, 3,675종목, 상폐포함)
- 장기 패널: `/mnt/20t/study/mom_paper_test/data/external/mom_prices.parquet`
  - 컬럼: `Code, Date, AdjClose, RawClose, MarketCap, Name, CodeType` (1979~2026)
- 팩터: `/mnt/20t/main/FF3/ff3_factors/ff3_factors_korea.csv`
  - 컬럼: `Date, MKT, SMB, HML, RF, MKT_RF` (일별, 2002-04-24~2025)
- 상장/상폐: `/mnt/20t/study/mom_paper_test/data/processed/listings.csv`

## 2. 팩터 구성 알고리즘 (WML)

```
입력: 일별 패널 (AdjClose, MarketCap, Code, Date)
매월 말일 T (형성일):
  1) 후보 = AdjClose(T), AdjClose(T-12M), AdjClose(T-1M) 모두 유효 & MarketCap(T) 유효한 종목
  2) 신호 S_i = AdjClose_i(T-1M) / AdjClose_i(T-12M) - 1        # t-12~t-2, 최근월 스킵
  3) S 기준 10분위 컷 (all-stock). D10=winner, D1=loser
  4) 각 분위 가중 w_i = MarketCap_i(T) / Σ MarketCap(T)         # 형성시점 VW
  5) 보유기간 (T, 다음 월말 T'] 동안 각 leg의 일별 buy-and-hold VW 수익률:
       v_i 초기화 = w_i (형성일)
       거래일 d: r_i(d) = AdjClose_i(d)/AdjClose_i(d-1) - 1  (|r|>0.5 → NaN, skip)
                 r_leg(d) = Σ_i v_i(d-1)·r_i(d) / Σ_i v_i(d-1)
                 v_i(d) = v_i(d-1)·(1+r_i(d))                  # 가중 표류
       상폐(del-a): 가격 소멸 시 v_i 제거·재정규화
  6) WML_daily(d) = r_D10(d) - r_D1(d)
출력: wml_daily.csv (Date, ret), wml_monthly.csv (월말, ∏(1+r)-1)
```

- **두 개의 WML 시계열(의도된 설계, Barroso 방식):**
  - **월별 WML**(Table 1·3, Fig 1·6용) = **leg별 월복리 차**: `(∏(1+r_D10)−1) − (∏(1+r_D1)−1)`. 이는 Kenneth French의 모멘텀 팩터 정의(승자 decile 월수익 − 패자 decile 월수익)와 동일 — Barroso가 Table 1·3에서 쓰는 그 방식. → `wml_monthly.csv`.
  - **일별 WML**(실현변동성 Eq 2·5, WML* 스케일용) = **일별 차** `r_D10(d) − r_D1(d)`. Barroso도 실현분산은 일별 WML 수익으로 계산. → `wml_daily.csv`.
  - **두 시계열은 정의상 다르다**(일별차 복리 ≠ leg별 월복리 차; 교차항 차이). 이는 버그가 아니라 Barroso 방법 자체의 구조 — 월별 전략수익(French)과 일별기반 위험추정(RV)이 별개다. WML*ₜ = scale(t)·월별WMLₜ, scale은 일별WML의 6개월 RV로 산출.
- **RMRF/SMB/HML 월별** = 일별 팩터를 월말 복리 리샘플.

## 3. 실현변동성 & 위험관리

- **Eq 2 (Fig 2용):** `RV_t = Σ_{j=0}^{20} r²_{WML}(d_t−j)` (월말 직전 21거래일). 연율화 변동성 = `√(12·RV_t)·100`.
- **Eq 5 (스케일용):** `σ̂²_t = 21 · Σ_{j=0}^{125} r²_{WML}(d_{t−1}−j) / 126` (직전 6개월, look-ahead 없음).
- **Eq 6:** `WML*_t = (σ_target,m / σ̂_t)·WML_t`, `σ_target,m = 0.12/√12` (월 목표), `σ̂_t=√(σ̂²_t)` (월 단위).

## 4. 산출물 정의

- **Table 1:** {RMRF, SMB, HML, WML} × {Max, Min(월%), Mean(연율%), Std(연율%), 초과Kurt, Skew, Sharpe(연율)}. Max/Min=월수익 최대/최소×100. Mean=월평균×12×100. Std=월std×√12×100. Sharpe=월평균/월std×√12.
- **Table 3:** {WML, WML*} × 위 7열 + **정보비율(IR)**. IR: WML·WML*를 각자 std로 나눠 정규화 후 `mean(WML*_n − WML_n)/std(WML*_n − WML_n)·√12`.
- **Fig 1:** 2패널, 한국 2대 격변기(데이터 기반 최악 12개월 드로다운으로 선정). 각 패널 WML·RMRF 누적 `∏(1+r)`.
- **Fig 2:** WML 월별 연율 실현변동성 시계열 (전 표본).
- **Fig 3:** 각 월을 직전 6개월 RV로 5분위 분류(팩터별). 분위별 이후 12개월 (A)실현변동성 (B)누적수익%p (C)샤프. WML·RMRF 막대.
- **Fig 6:** Fig 1과 동일 2패널, WML vs WML* 누적.

## 5. 검증 게이트 (완료 보고 전 필수)

1. WML 월별 통계가 mom_paper_test 선행결과와 정합(한국 모멘텀 약함: 샤프 낮음, 음의 왜도, 큰 MDD)인가?
2. WML* 실현 std ≈ 12~17%(타깃 근처), 샤프·왜도·첨도가 WML 대비 개선되는가? (Barroso 방향성)
3. 스케일 σ̂_t가 look-ahead 없이 직전월까지 정보만 쓰는가? (코드 검토)
4. survivorship: 상폐종목이 보유기간 중 정상 반영·이후 제외되는가?
5. 각 Fig가 렌더되고 축·범례·기간이 정의대로인가? (PNG 육안)
6. 적대적 검증 에이전트가 수식·정합성 통과 판정.

> **한국 특이성 예상:** raw WML 샤프가 미국(0.53)보다 훨씬 낮을 것(선행 ≈0.13). 재현의 관전 포인트는 **risk-managed가 약한 한국 모멘텀에서도 샤프·꼬리위험을 개선하는가**(Barroso 일본 사례 0.08→0.24 유형). 숫자를 미국과 맞추는 게 목표가 아니라 **메커니즘 전이 여부**를 보는 것.
