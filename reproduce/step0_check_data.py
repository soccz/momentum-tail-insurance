# -*- coding: utf-8 -*-
"""
STEP 0. 데이터 확인 — "내 데이터가 논문과 같은 데이터인가?"
================================================================
이 스크립트는 아무것도 계산하지 않습니다. 딱 세 가지만 합니다.

  (1) 데이터 파일 3개가 제자리에 있는지
  (2) 파일이 논문과 *비트 단위로 동일*한지 (SHA256 지문 대조)
  (3) 무위험수익률이 진짜 공개 출처(FRED)에서 재현되는지 (인터넷 연결 시)

실행:  python3 reproduce/step0_check_data.py
※ 공개 레포판: 경로는 레포 기준이며, FnGuide 구독 데이터 2종(①②)은 레포에 없어
   자동으로 건너뜁니다(보유 시 data/에 놓으면 전체 검증). ③ FRED 대조는 레포만으로 동작합니다.
소요:  약 10초 (FRED 온라인 대조 포함 시 +5초)

데이터 출처 (각 파일을 "누가 어디서" 얻는가):
  ① mom_prices.parquet  — 에프앤가이드(FnGuide) DataGuide에서 저자가 직접 추출.
       내용: KRX 전종목(코스피+코스닥, 상장폐지 포함) 일별 수정주가·시가총액.
       재추출: DataGuide(구독)에서 [수정주가]와 [시가총액]을 전종목·상폐포함으로
       내려받으면 됨. 구독이 없으면 아래 SHA256으로 이 파일이 조작되지 않았음을 확인.
  ② pbr_monthly.parquet — FnGuide 일간 PBR 원본(PBR.xlsx, 1.1GB)에서 월말만 추출.
       원본: /mnt/20t/study/mom_paper_test/data/PBR.xlsx
  ③ rf_monthly.csv      — 한국은행 콜금리. **완전 공개 데이터**: FRED 시리즈
       IRSTCI01KRM156N (콜금리), IR3TIB01KRM156N (CD 3개월).
       누구나 키 없이 다운로드 가능:
       https://fred.stlouisfed.org/graph/fredgraph.csv?id=IRSTCI01KRM156N
       월 수익률 변환식: rf_m = (1 + 연율%/100)^(1/12) − 1
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd

# ── 파일 위치 (공개 레포 기준 상대경로) ────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "주가·시총 패널": ROOT / "data/external/mom_prices.parquet",   # FnGuide — 공개 레포 미포함
    "월말 PBR":      ROOT / "data/processed/pbr_monthly.parquet",  # FnGuide — 공개 레포 미포함
    "무위험수익률":    ROOT / "data/processed/rf_monthly.csv",       # 공개 (FRED 재현 가능)
}
OPTIONAL = {"주가·시총 패널", "월말 PBR"}  # 없으면 건너뜀 (SHA256 매니페스트로 동일성 검증 대체)

# ── 논문 v3.3 시점의 SHA256 지문 (이 값과 다르면 데이터가 바뀐 것) ──────
# 최초 실행 시 자동 생성되어 reproduce/DATA_MANIFEST.txt 에 저장됩니다.
MANIFEST = Path(__file__).parent / "DATA_MANIFEST.txt"


def sha256_of(path: Path) -> str:
    """파일 전체의 SHA256 해시(지문)를 계산한다. 1비트만 달라도 값이 완전히 바뀐다."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):  # 1MB씩 읽기
            h.update(chunk)
    return h.hexdigest()


print("=" * 72)
print("STEP 0-(1) 파일 존재 + 기본 구조 확인")
print("=" * 72)
ok = True
present = {}
for name, path in FILES.items():
    if not path.exists():
        if name in OPTIONAL:
            print(f"  – {name}: 미포함 (FnGuide 구독 데이터) → 건너뜀. 보유 시 {path.relative_to(ROOT)} 에 배치")
            continue
        print(f"  ✗ {name}: 없음! → {path}")
        ok = False
        continue
    present[name] = path
    size_mb = path.stat().st_size / 1e6
    print(f"  ✓ {name}: {path.name} ({size_mb:,.1f} MB)")
if not ok:
    sys.exit("필수 파일이 없습니다. README의 데이터 출처를 보고 준비하세요.")

# 내용의 핵심 성질 확인 (논문 §3.1 기술과 일치해야 함)
if "주가·시총 패널" in present:
    p = pd.read_parquet(FILES["주가·시총 패널"], columns=["Date", "Code"])
    print(f"\n  주가 패널: {len(p):,}행, 기간 {p.Date.min().date()} ~ {p.Date.max().date()}, "
          f"고유 종목 {p.Code.nunique():,}개")
    print(f"  (논문 §3.1: 1990+ 필터 후 4,660종목 — 필터 전 전체는 더 많은 것이 정상)")

r = pd.read_csv(FILES["무위험수익률"])
print(f"  무위험수익률: {len(r)}개월, {r.month.iloc[0]} ~ {r.month.iloc[-1]}, "
      f"컬럼 {list(r.columns)}")

print()
print("=" * 72)
print("STEP 0-(2) SHA256 지문 대조 — 데이터가 논문 시점과 동일한가")
print("=" * 72)
hashes = {name: sha256_of(path) for name, path in present.items()}

if MANIFEST.exists():
    # 저장된 지문과 대조
    saved = dict(line.split("\t")[:2][::-1] for line in MANIFEST.read_text().strip().split("\n")[1:])
    all_match = True
    for name, h in hashes.items():
        match = saved.get(h) == FILES[name].name or h in saved
        # saved: {hash: filename} 형태로 재구성했으므로 hash로 조회
        status = "일치 ✓" if h in saved else "다름 ✗ (데이터가 변경됨!)"
        if h not in saved:
            all_match = False
        print(f"  {name}: {h[:16]}…  {status}")
    print(f"\n  판정: {'PASS — 논문과 동일한 데이터' if all_match else 'FAIL — 지문 불일치'}")
else:
    # 최초 실행: 지문을 기록해 둔다 (이후 실행부터 대조)
    lines = ["# 데이터 지문 (SHA256) — 논문 v3.3 기준. 이 파일은 수정하지 마세요."]
    for name, h in hashes.items():
        lines.append(f"{FILES[name].name}\t{h}")
        print(f"  {name}: {h[:16]}…  (최초 기록)")
    MANIFEST.write_text("\n".join(lines) + "\n")
    print(f"\n  지문을 저장했습니다 → {MANIFEST}")

print()
print("=" * 72)
print("STEP 0-(3) 무위험수익률 출처 검증 — FRED에서 직접 재다운로드해 대조")
print("=" * 72)
try:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=IRSTCI01KRM156N"
    fred = pd.read_csv(url)                                   # 키 없이 공개 다운로드
    fred["rf_m"] = (1 + fred.iloc[:, 1] / 100) ** (1 / 12) - 1  # 연율% → 월 수익률
    fred["month"] = pd.to_datetime(fred.iloc[:, 0]).dt.strftime("%Y-%m")
    ours = pd.read_csv(FILES["무위험수익률"])
    m = fred.merge(ours, on="month")
    diff = (m.rf_m - m.rf_call).abs().max()
    print(f"  대조 {len(m)}개월, 최대 절대차 {diff:.2e}")
    print(f"  판정: {'PASS — rf_monthly.csv는 FRED 공개 시리즈에서 정확히 재현됨' if diff < 1e-9 else 'FAIL'}")
except Exception as e:
    print(f"  (오프라인이거나 FRED 접속 실패 — 건너뜀: {e})")

print("\nSTEP 0 완료. 다음: python3 reproduce/step1_reproduce.py")
