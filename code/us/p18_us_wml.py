"""
미국 병렬 1단계: Ken French 일별/월별 10분위 모멘텀 → US WML(VW D10−D1) 구성 + BS15 재현 게이트.
목표: 우리 한국 파이프라인과 동일 규격(VW 10분위, t−12~t−2, 월 리밸런스)의 미국판이
      BS15의 알려진 미국 수치(raw 샤프~0.5, 매우 음의 왜도, RW126 관리가 샤프 대략 2배, 2009 크래시)를 재현하는가.
"""
import numpy as np, pandas as pd
from scipy import stats
D="/mnt/20t/tmp/claude-1001/-mnt-20t-----/da2b9872-77c2-46d2-a247-cd601338a0be/scratchpad/"

def load_deciles(fp, daily):
    lines=open(fp).read().splitlines()
    freq="Daily" if daily else "Monthly"; dlen=8 if daily else 6
    # VW(가치가중) + 해당 빈도 섹션 헤더 (Annual/Equal/Number 제외)
    def is_vw_hdr(l):
        return ("Value" in l and "Weight" in l and freq in l
                and "Annual" not in l and "Equal" not in l and "Number" not in l)
    i0=next(i for i,l in enumerate(lines) if is_vw_hdr(l))
    rows=[]
    for l in lines[i0+1:]:
        t=l.split(","); k=t[0].strip()
        if k.isdigit() and len(k)==dlen:
            try: rows.append([k]+[float(x) for x in t[1:11]])
            except: break
        elif rows: break   # 데이터 시작 후 비-날짜행 → VW 섹션 끝
    df=pd.DataFrame(rows).set_index(0)
    df.index=pd.to_datetime(df.index, format="%Y%m%d" if daily else "%Y%m")
    df=df.replace([-99.99,-999.0], np.nan)/100.0
    df.columns=[f"d{i}" for i in range(1,11)]
    return df

dd=load_deciles(D+"10_Portfolios_Prior_12_2_Daily.csv", True)
dm=load_deciles(D+"10_Portfolios_Prior_12_2.csv", False)
wml_d=(dd["d10"]-dd["d1"]).dropna()            # 일별 WML (실현변동성용)
wml_m=(dm["d10"]-dm["d1"]).dropna()            # 월별 WML (전략용)
print(f"[US WML] 일별 {wml_d.index[0].date()}~{wml_d.index[-1].date()} ({len(wml_d)}일) · 월별 {len(wml_m)}개월")

sr=lambda r: np.sqrt(12)*np.nanmean(r)/np.nanstd(r)
print("\n===== raw US WML (전기간) =====")
print(f"  연평균 {wml_m.mean()*1200:.2f}% · 샤프 {sr(wml_m):.3f} · 왜도 {stats.skew(wml_m):.2f} · 초과첨도 {stats.kurtosis(wml_m):.1f}")
print(f"  최악월 {wml_m.min()*100:.1f}% ({wml_m.idxmin().date()}) · 차악 {wml_m.nsmallest(2).iloc[-1]*100:.1f}% ({wml_m.nsmallest(2).index[-1].date()})")

# BS15 표본(1927-2013)으로도 대조
w=wml_m.loc["1927":"2013"]
print(f"\n===== BS15 표본(1927–2013) 대조 =====")
print(f"  연평균 {w.mean()*1200:.2f}% (BS15 14.46) · 샤프 {sr(w):.3f} (BS15 0.53) · 왜도 {stats.skew(w):.2f} (BS15 −2.47) · 최악 {w.min()*100:.1f}% (BS15 −78.96)")

# RW126 위험관리 재현
me=wml_d.groupby(wml_d.index.to_period("M")).apply(lambda x:x.index[-1])
sq=wml_d**2
fvar=(21*sq.rolling(126,min_periods=126).mean())
fvar_me=fvar.reindex(pd.DatetimeIndex(me.values)).copy(); fvar_me.index=me.index   # 월말 σ̂²
TGT=0.12/np.sqrt(12)
wml_m_p=wml_m.copy(); wml_m_p.index=wml_m_p.index.to_period("M")
L=(TGT/np.sqrt(fvar_me)).shift(1)              # t−1월말 정보로 t월 스케일 (look-ahead 방지)
star=(L*wml_m_p).dropna()
print(f"\n===== RW126 위험관리 (WML*, 12% 타깃) =====")
print(f"  샤프 {sr(star):.3f} (BS15 0.97) · 왜도 {stats.skew(star):.2f} (BS15 −0.42) · 최악 {star.min()*100:.1f}% · 초과첨도 {stats.kurtosis(star):.1f}")
print(f"  IR(vs raw, 각자 표준화 차) {sr((star/star.std()-wml_m_p.reindex(star.index)/wml_m_p.std())):.3f}")

# 게이트: 미국 재현 성립?
ok = (0.35<sr(w)<0.65) and stats.skew(w)<-1.0 and sr(star)>sr(w)*1.3 and w.min()<-0.5
print(f"\n[GATE] BS15 미국 재현 {'PASS' if ok else 'FAIL'} — raw샤프~0.5·강한 음왜도·관리가 크게 개선·큰 크래시")
wml_d.to_csv(D+"us_wml_daily.csv"); wml_m.to_csv(D+"us_wml_monthly.csv")
