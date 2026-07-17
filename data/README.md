# 데이터 출처와 공개 범위

## 원칙

**종목 수준 원시 데이터는 공개하지 않는다** (FnGuide DataGuide 구독 라이선스).
공개하는 것은 그로부터 변형·집계된 **팩터/포트폴리오 수준 시계열**과 **공개 출처 데이터**뿐이다 —
Kenneth French Data Library가 CRSP 파생 팩터 수익률을 공개하는 것과 같은 관행이다.

## 파일별 내역

| 파일 | 내용 | 수준 | 원출처 |
|---|---|---|---|
| `processed/wml_daily.csv` · `wml_monthly.csv` (+`_clean`) | 한국 모멘텀 WML 수익률 (일별/월별, 본편·청정 패널) | 포트폴리오 집계 | FnGuide 파생 |
| `processed/wml_2x3_daily.csv` · `wml_2x3_monthly.csv` | 한국 2×3 분산 구성 WML (절제 실험 E22) | 포트폴리오 집계 | FnGuide 파생 |
| `processed/ff_own_*.csv` · `factors_*.csv` | 한국 시장·SMB·HML·WML 팩터 세트 | 포트폴리오 집계 | FnGuide 파생 |
| `processed/ml_dataset.csv` | ML 피처 20종 + 타깃 (전부 포트폴리오/시장 수준 집계) | 집계 | FnGuide 파생 |
| `processed/ml_forecasts.csv` | 12개 예측기의 표본외 분산 예측치 | 모델 산출물 | — |
| `processed/p3_weights.csv` | 예측기별 월간 스케일 가중치 | 모델 산출물 | — |
| `processed/rf_monthly.csv` | 무위험수익률 (한국은행 콜금리·CD) | 공개 | FRED `IRSTCI01KRM156N` (라이브 대조 내장) |
| `us/`, `intl/` | 미국·국제 모멘텀/팩터 일별 | 공개 | Kenneth R. French Data Library |
| ~~종목별 주가·시가총액·PBR~~ | **미공개** | 종목 수준 | FnGuide (구독) — `reproduce/DATA_MANIFEST.txt`의 SHA256으로 동일성 검증 |

## 재현 범위

- **이 저장소만으로**: 예측 경기(p2)부터 최종 검정(p30)까지 분석 파이프라인 전체 + 국제 병렬 전체.
- **FnGuide 원시 패널 필요**: `build_*.py`(팩터 구축)와 `reproduce/step1`(원시 주가→표 1·3) — 데이터 보유 기관에서 매니페스트 대조로 재현.
