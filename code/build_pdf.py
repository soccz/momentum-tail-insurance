"""
제출본 PDF 빌더 — paper/thesis_kr.md → 그림 임베드 학술 PDF.
markdown→HTML(표·각주 포함) → 수식은 유니코드 근사 → Chrome headless --print-to-pdf (오프라인, CDN 없음).
게이트: PDF 생성 + 그림 3장 임베드 + 페이지 수 출력.
"""
import re, base64, subprocess, sys
from pathlib import Path
import markdown

ROOT = Path("/mnt/20t/졸업논문")
SRC = ROOT / "paper/thesis_kr.md"
OUT_HTML = ROOT / "paper/thesis_kr.html"
OUT_PDF = ROOT / "paper/thesis_kr.pdf"

md = SRC.read_text(encoding="utf-8")

# ---- 1) 내부 헤더(상태줄 인용문) 제거: 제출본에는 미표시 ----
md = re.sub(r"^> 한국어 초고 정본.*\n> 상태:.*\n", "", md, flags=re.M)

# ---- 2) LaTeX 수식 → 유니코드/HTML 근사 (원고의 실제 사용분만) ----
MATH = {
    r"$w_t=\sigma_{target}/\hat\sigma_t$": "<i>w<sub>t</sub></i> = σ<sub>target</sub>/σ̂<sub>t</sub>",
    r"$\hat\sigma_t$": "σ̂<sub>t</sub>",
    r"$\sigma_{target}$": "σ<sub>target</sub>",
    r"$$\hat\sigma^2_{t}=21\cdot\frac{1}{126}\sum_{j=0}^{125} r^2_{WML,\,d_{t-1}-j}, \qquad r_{WML^*,t}=\frac{\sigma_{target}}{\hat\sigma_t}\,r_{WML,t}$$":
        '<div class="eq">σ̂²<sub>t</sub> = 21 · (1/126) Σ<sub>j=0..125</sub> r²<sub>WML, d<sub>t−1</sub>−j</sub> &nbsp;&nbsp;&nbsp; r<sub>WML*,t</sub> = (σ<sub>target</sub>/σ̂<sub>t</sub>) · r<sub>WML,t</sub></div>',
    r"$d_{t-1}$": "<i>d<sub>t−1</sub></i>",
    r"$t$": "<i>t</i>",
    r"$T$": "<i>T</i>",
    r"$t-12$": "<i>t</i>−12",
    r"$t-2$": "<i>t</i>−2",
    r"$t-1$": "<i>t</i>−1",
    r"$B/M = 1/PBR$": "B/M = 1/PBR",
    r"$B/M$": "B/M",
    r"$\lambda\in\{0.7,0.8,0.9,0.94\}$": "λ ∈ {0.7, 0.8, 0.9, 0.94}",
    r"$\beta_t^2 \cdot RV_{mkt}$": "β²<sub>t</sub>·RV<sub>mkt</sub>",
    r"$\mu/\sigma^2$": "μ/σ²",
    r"$\mu$": "μ",
    r"$\sigma$": "σ",
    r"$c\times(2\cdot TO\cdot L + 2|\Delta L|)$": "c × (2·TO·L + 2|ΔL|)",
    r"$c$": "<i>c</i>",
    r"$TO$": "<i>TO</i>",
    r"$L$": "<i>L</i>",
    r"$= \mu - \tfrac{\gamma}{2}\sigma^2$": "= μ − (γ/2)σ²",
    r"$\mathbb{E}[(1+r)^{1-\gamma}]^{1/(1-\gamma)}-1$": "E[(1+r)<sup>1−γ</sup>]<sup>1/(1−γ)</sup> − 1",
    r"$(1+r)$": "(1+r)",
    r"$\mathrm{QLIKE}(f)=RV/f-\log(RV/f)-1$": "QLIKE(<i>f</i>) = RV/<i>f</i> − log(RV/<i>f</i>) − 1",
    r"$f$": "<i>f</i>",
    r"$RV$": "RV",
}
for k, v in MATH.items():
    md = md.replace(k, v)
leftover = re.findall(r"\$[^$\n]{1,80}\$", md)
if leftover:
    print("[경고] 미변환 수식:", leftover[:10])

# ---- 3) markdown → HTML ----
body = markdown.markdown(md, extensions=["tables", "smarty"])

# ---- 4) 그림 file:// 절대경로 → base64 임베드 (완전 자립 PDF) ----
def embed_img(m):
    src = m.group(1)
    p = (ROOT / "paper" / src).resolve()
    if not p.exists():
        print(f"[경고] 그림 없음: {src}"); return m.group(0)
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f'src="data:image/png;base64,{b64}"'
body, n_img = re.subn(r'src="([^"]+\.png)"', embed_img, body)
print(f"[그림] {n_img}장 임베드")

CSS = """
@page { size: A4; margin: 22mm 20mm; }
body { font-family: 'Noto Serif CJK KR','Noto Serif CJK SC',serif; font-size: 10.5pt;
       line-height: 1.75; color: #111; max-width: 100%; }
h1 { font-size: 16pt; text-align: center; line-height: 1.4; margin: 0 0 4pt; }
h2 { font-size: 12.5pt; margin: 22pt 0 8pt; border-bottom: 1px solid #999; padding-bottom: 3pt; }
h3 { font-size: 11pt; margin: 14pt 0 6pt; }
p { margin: 0 0 8pt; text-align: justify; }
table { border-collapse: collapse; margin: 10pt auto; font-size: 8.8pt; }
th, td { border-top: 1px solid #444; border-bottom: 1px solid #444;
         padding: 3pt 7pt; text-align: center; }
thead th { border-bottom: 1.5px solid #111; border-top: 1.5px solid #111; }
img { max-width: 100%; display: block; margin: 10pt auto; }
.eq { text-align: center; margin: 10pt 0; font-size: 10.5pt; }
blockquote { color: #444; border-left: 3px solid #bbb; margin: 8pt 0; padding: 2pt 10pt; font-size: 9.5pt; }
li { margin-bottom: 3pt; }
strong { font-weight: 700; }
"""
html = f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><style>{CSS}</style></head><body>{body}</body></html>'
OUT_HTML.write_text(html, encoding="utf-8")

# ---- 5) Chrome headless → PDF (오프라인) ----
r = subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox",
                    "--no-pdf-header-footer", f"--print-to-pdf={OUT_PDF}", f"file://{OUT_HTML}"],
                   capture_output=True, text=True, timeout=120)
assert OUT_PDF.exists() and OUT_PDF.stat().st_size > 100_000, f"PDF 생성 실패: {r.stderr[-300:]}"
try:
    import pypdf; pages = len(pypdf.PdfReader(str(OUT_PDF)).pages)
except Exception:
    pages = "?"
print(f"[GATES] PDF ✓ {OUT_PDF} ({OUT_PDF.stat().st_size//1024}KB, {pages}쪽) · 그림 {n_img}장 ✓")
