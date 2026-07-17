# Momentum Tail Insurance — 위험 예측이 좋아지면 무엇이 좋아지는가

> **약한 모멘텀 시장에서 머신러닝 변동성 예측과 꼬리 보험**
> What Improves When Risk Forecasts Improve: Machine-Learned Volatility and Tail Insurance in a Weak Momentum Market

Barroso and Santa-Clara (2015, JFE)의 위험관리 모멘텀에서 **위험 예측기 하나만 체계적으로 교체**하는 설계로,
"예측이 좋아지면 포트폴리오의 무엇이 좋아지는가"를 한국 전종목 35년(1991–2026, 상장폐지 포함)에서 측정한 연구의
**코드·결과 원장·재현 킷** 저장소입니다.

- 🧭 **연구 여정 (성공·실패·철회 전체 기록)**: https://soccz.github.io/projects/momentum-journey/
- 🎛️ **원논문(BS15) 재현 인터랙티브 데모**: https://soccz.github.io/projects/momentum-barroso/

## 세 발견

1. **예측은 좋아지지만 샤프비율은 따라오지 않는다** — 앙상블이 QLIKE를 24% 줄여도(중첩보정 CW t=3.4),
   전이곡선의 샤프–정확도 기울기는 0과 구분되지 않는다 (조인트 블록 부트스트랩 CI [−0.006, +0.006]).
2. **개선은 꼬리로 간다** — 월간 기대꼬리손실(ES5)이 0.86%p 얕아지고, 사전 지정 예측기(HAR·Lasso)에서
   Romano–Wolf 다중검정 보정을 통과하며, 극값이론으로 1% 꼬리까지, 거래비용을 부과한 순수익에서도 남는다.
3. **보험이 성립하는 곳은 좁고, 보험을 파는 것은 정확도가 아니라 구조다** — 효과는 약한 팩터 × 극단 구성의
   교집합(한국 극단 10분위)에서만 성립하고(한국−미국 차이 검정 D=+2.01%p, p=0.001),
   꼬리 증분의 대부분은 장기기억(HAR)·시장상태 구조가 확보하며 순수 ML 결합의 추가분은 0이다.

## 저장소 구조

```
code/                  분석 파이프라인 (실행 순서 = 번호 순서)
  build_*.py           팩터·데이터셋·표·그림 빌드
  p2  예측 경기        p3  전이곡선        p4  메커니즘·플라시보
  p5~p9  강건성 배터리(부트스트랩·비용·구성 그리드·EVT 전 단계)
  p10~p17  심사 대응(계량 수정·RF 실질화·상폐/필터·EVT·Romano-Wolf)
  p18~p24  국제 병렬(us/·intl/ — 미국·4지역·2×3 절제·메커니즘 대리검정)
  p25~p30  최종 게이트 라운드 신규 검정(플라시보 꼬리·CW-SPA·표본창 분해·
           미국 동창·ML−HAR 분해·국제 QLIKE·한국−미국 차이 검정)
docs/
  01_reproduction_spec.md   방법론 고정 사양
  04_research_design.md     연구 설계·정직성 프로토콜
  05_results_ledger.md      ★ 결과 원장 E00~E38 (append-only: 실패·철회 포함 전체 기록)
  06_claims_map.md          ★ 주장–근거 지도 C1~C42 (주장↔수치↔표↔실험↔재현 명령)
notebooks/momentum_korea.ipynb   교육용 최단순 재현 (pandas 기본기만, 수식↔코드 1:1)
reproduce/               초간단 재현 킷 (데이터 무결성 SHA256 + 표 1·3 재현, 7게이트)
output/tables/           논문 전 표의 정본 CSV (본문 수치는 전부 여기로 소급)
output/figures/          논문 그림
data/us/, data/intl/     Ken French 공개 데이터 파생 시계열 (재현 포함)
data/processed/rf_monthly.csv   무위험수익률 (한국은행 콜금리, FRED 경유 재현 가능)
```

## 데이터 정책 (중요)

| 데이터 | 출처 | 이 저장소 |
|---|---|---|
| 한국 주가·시가총액·PBR | FnGuide DataGuide (구독) | **미포함** — 라이선스상 재배포 불가. `reproduce/DATA_MANIFEST.txt`의 SHA256으로 동일성 검증 가능 |
| 무위험수익률 (콜금리) | 한국은행 / FRED `IRSTCI01KRM156N` | 포함 (`reproduce/step0`가 FRED 라이브 대조 — 최대차 1e-16) |
| 미국·국제 모멘텀/팩터 | Kenneth R. French Data Library | 포함 (공개 데이터) |

한국 원시 데이터가 없으면 `code/p2` 이후의 본편 파이프라인은 재실행할 수 없지만,
(1) 모든 결과 수치는 `output/tables/`의 CSV와 `docs/05_results_ledger.md`로 소급되고,
(2) 국제 병렬(`code/us/`, `code/intl/`, p25~p30의 국제 부분)은 이 저장소만으로 완전 재현됩니다.

## 논문

한국어 정본(v4.2)은 학위 심사 절차를 고려해 심사 후 공개 예정입니다.
그 전까지 본문의 모든 정량 주장은 `docs/06_claims_map.md`(주장–근거 지도)와
`output/tables/`(정본 CSV)로 검증할 수 있습니다.

## 정직성 프로토콜

이 연구는 다음 장치로 진행됐습니다 — 자세한 이야기는 [연구 여정 페이지](https://soccz.github.io/projects/momentum-journey/) 참조.

- **사전 지정**: 예측기 12개·평가 프로토콜을 결과 산출 전 커밋으로 고정 (부록 A 타임라인)
- **결과 원장 (append-only)**: 실패·철회·정정까지 E00~E38로 전부 기록 — 원장에 없으면 미완료
- **주장–근거 지도**: 논문의 모든 주장은 수치·표·실험·재현 명령과 연결된 행 하나를 가짐
- **블라인드 게이트**: 5개 렌즈(산문·심사·수치·구조·방어) × 적대적 검증 × 7라운드 — 확정 지적은 계산으로만 닫음
- **사후 개입 공개**: 결과 관찰 후 도입한 수치 안정화 장치 2건의 도입 계기·코드 위치 공개

## License

코드는 MIT License를 따릅니다. Ken French Data Library 파생 데이터의 권리는 원 출처에,
결과 표·그림은 저자에게 있습니다 (학술 인용 환영).
