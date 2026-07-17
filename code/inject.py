"""Assemble deep.html from template + data.json + deep_sections fragment.
Also writes an offline screenshot variant (CDN stripped, reveal forced)."""
import re
from pathlib import Path

ROOT = Path("/mnt/20t/졸업논문")
tpl = (ROOT / "code/deep_template.html").read_text()
data = (ROOT / "output/web/data.json").read_text()
frag_path = ROOT / "output/web/deep_sections_fragment.html"
frag = frag_path.read_text() if frag_path.exists() else "<p style='color:var(--text-muted)'>정밀 독해 생성 중…</p>"
cpt_path = ROOT / "output/web/concepts_fragment.html"
cpt = cpt_path.read_text() if cpt_path.exists() else "<p style='color:var(--text-muted)'>개념 온보딩 생성 중…</p>"

out = (tpl.replace("__DATA__", data)
          .replace("__DEEP_SECTIONS__", frag)
          .replace("__CONCEPTS__", cpt))
(ROOT / "deep.html").write_text(out)
assert "__DATA__" not in out and "__DEEP_SECTIONS__" not in out, "placeholder left"
print(f"[deep.html] {len(out)//1024} KB")

# offline shot variant
h = re.sub(r'<link[^>]*fonts[^>]*>', '', out)
h = re.sub(r'<script[^>]*mathjax[^>]*></script>', '', h)
h = re.sub(r'<script>\s*MathJax = .*?</script>', '', h, flags=re.S)
h = h.replace('</style>', '.reveal{opacity:1!important;transform:none!important;}</style>')
SD = Path("/mnt/20t/tmp/claude-1001/-mnt-20t-----/da2b9872-77c2-46d2-a247-cd601338a0be/scratchpad")
(SD / "deep_shot.html").write_text(h)
print(f"[shot] {SD}/deep_shot.html")
