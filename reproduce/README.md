# 재현 킷 — 논문 핵심 결과를 실제 데이터에서 직접 재현하기

> **공개판 주석**: 본 문서의 `paper/thesis_kr.md` 참조와 버전 표기는 원 연구 레포 기준이다(논문 전문은 심사 후 공개, 현재 정본 v4.2). FnGuide 원시 데이터 2종은 이 레포에 없으며 SHA256 매니페스트로 동일성만 검증한다.

> 논문: "위험 예측이 좋아지면 무엇이 좋아지는가" (`paper/thesis_kr.md` v3.3)
> 이 폴더만으로 **표 1(모멘텀 통계)과 표 3(위험관리 효과)** 을 원시 주가에서 재현한다.
> 필요한 것: Python 3 + pandas + numpy. 머신러닝 라이브러리 불필요.

## 빠른 시작 (2단계)

```bash
python3 reproduce/step0_check_data.py    # ① 데이터가 제자리에·논문과 동일한지 확인 (~10초)
python3 reproduce/step1_reproduce.py     # ② 표 1·표 3 재현 + 게이트 판정 (~2-4분)
```

마지막 줄에 `✅ 전체 PASS`가 나오면 재현 성공이다. (2026-07-14 실행 확인:
샤프 0.160·왜도 −1.16·최악월 1998-10 −94.9%·WML* 샤프 0.260·IR 0.229 — 7게이트 전체 PASS)

## 데이터 출처 — 전부 "내가 만든 값"이 아니라 외부에서 온 실측 자료다

| 파일 | 내용 | 출처 | 직접 얻는 법 |
|---|---|---|---|
| `mom_prices.parquet` (167MB) | KRX 전종목(코스피+코스닥, **상폐 포함**) 일별 수정주가·시가총액, 1979-12~2026-04 | 에프앤가이드 **DataGuide** (구독 서비스) | DataGuide에서 [수정주가]·[시가총액]을 전종목·상장폐지포함 옵션으로 추출. 구독이 없으면 `DATA_MANIFEST.txt`의 SHA256으로 이 파일이 추출 후 변형되지 않았음을 확인 |
| `pbr_monthly.parquet` (1MB) | 월말 PBR (B/M=1/PBR용), 1988+ | 동일 DataGuide (원본 `PBR.xlsx` 1.1GB) | 원본: `/mnt/20t/study/mom_paper_test/data/PBR.xlsx`. ⚠️ 표 1·3 재현에는 불필요(HML 전용) |
| `rf_monthly.csv` (24KB) | 콜금리·CD 3개월 월수익률, 1991+ | **FRED 완전 공개** (키 불필요) | `https://fred.stlouisfed.org/graph/fredgraph.csv?id=IRSTCI01KRM156N` (콜) / `...?id=IR3TIB01KRM156N` (CD). 변환: `(1+연율%/100)^(1/12)−1`. **step0이 매 실행마다 온라인 재대조** — 2026-07-14 확인: 425개월 최대차 1e-16 (완전 일치) |
| (표 9 국제 대조용) | 미국·국제 모멘텀 포트폴리오 | **Ken French Data Library 완전 공개** | `https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html` → "10 Portfolios Formed on Momentum (Daily)" 등. ⚠️ 이 킷의 범위 밖(§8.4는 `code/us/`·`code/intl/`) |

**요점**: 주가 패널은 FnGuide 구독 추출물(저자 소장), 무위험수익률은 지금 이 순간에도
누구나 재다운로드 가능한 공개 데이터다. `DATA_MANIFEST.txt`의 SHA256 지문이 "논문이 쓴
바로 그 파일"임을 보증한다 — 파일이 1비트라도 바뀌면 step0이 FAIL을 낸다.

## step1이 하는 일 (논문 대응)

| 단계 | 내용 | 논문 |
|---|---|---|
| [A] | 원시 주가 → 와이드 표 → 일별수익(±50% 필터) → 매월 말 신호(t−12~t−2)로 10분위, VW 승자−패자, buy-and-hold | §3.1, §4.1 |
| [B] | 표 1: 샤프 0.17 · 왜도 −1.17 · 초과첨도 10.5 · 최악월 1998-10 −94.9% | §3.3 |
| [C] | 식(5) 126일 실현분산 σ̂² (t−1월 말 정보만) → 식(6) 스케일 = 12%/σ̂ | §4.2 |
| [D] | 표 3: WML* 샤프 0.26 · 최악월 −20% · IR 0.23 | §5.3 |
| [E] | 헌법 §3 고정값 대비 7게이트 PASS/FAIL | — |

## 이 킷이 재현하지 **않는** 것

예측 경기(표 4)·전이곡선(표 5~6)·메커니즘(표 7)·강건성(표 8)·국제 대조(표 9)는
머신러닝 학습이 필요해 "아주 쉬운 코드" 범위를 벗어난다. 그 재현 명령은 논문 부록 A
(`code/p2`~`p24`, 각 스크립트가 자체 게이트 출력)와 원장 `docs/05_results_ledger.md` 참조.
교육용 정본 노트북은 `notebooks/momentum_korea.ipynb` (2026-07-14 `nbconvert --execute` 재실행, 6게이트 PASS 확인).
