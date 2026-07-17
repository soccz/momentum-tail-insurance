"""
Render the deep-dissection workflow output (structured blocks) into a static HTML
fragment for deep.html. Reads output/web/deep_sections.json -> writes fragment.
Block types: p, h, eq, why, quote, list, key.
"""
import json
from pathlib import Path

SRC = Path("/mnt/20t/졸업논문/output/web/deep_sections.json")
OUT = Path("/mnt/20t/졸업논문/output/web/deep_sections_fragment.html")

ORDER = ["intro", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]


def safe(s):
    """Replace combining diacritics (σ̂, x̄) with spacing modifiers that render
    reliably across fonts. Applied to prose only (not LaTeX tex fields)."""
    return (s or "").replace("̂", "ˆ").replace("̄", "ˉ")


def render_block(b):
    t = b.get("t")
    if t == "p":
        return f'<p>{safe(b.get("html",""))}</p>'
    if t == "h":
        return f'<h4 class="subh">{safe(b.get("html",""))}</h4>'
    if t == "eq":
        cap = safe(b.get("cap", ""))
        tex = b.get("tex", "")
        return f'<div class="eq"><div class="cap">{cap}</div>\\[ {tex} \\]</div>'
    if t == "why":
        return f'<div class="why"><span class="k">왜</span>{safe(b.get("html",""))}</div>'
    if t == "quote":
        return f'<blockquote>{safe(b.get("html",""))}</blockquote>'
    if t == "key":
        return f'<div class="key">{safe(b.get("html",""))}</div>'
    if t == "list":
        lis = "".join(f"<li>{safe(x)}</li>" for x in b.get("items", []))
        return f"<ul>{lis}</ul>"
    if t == "table":
        head = "".join(f"<th>{safe(h)}</th>" for h in b.get("head", []))
        rows = "".join(
            "<tr" + (' class="hl"' if r and str(r[0]).startswith("★") else "") + ">"
            + "".join(f'<td{" style=\"text-align:left\"" if j == 0 else ""}>{safe(str(c)).lstrip("★")}</td>'
                      for j, c in enumerate(r)) + "</tr>"
            for r in b.get("rows", [])
        )
        cap = f'<p class="tbl-cap">{safe(b.get("cap",""))}</p>' if b.get("cap") else ""
        return (f'<div class="tbl-wrap"><table class="data"><thead><tr>{head}</tr></thead>'
                f"<tbody>{rows}</tbody></table></div>{cap}")
    return ""


BEG = Path("/mnt/20t/졸업논문/output/web/beginner.json")


def load_gateways():
    if not BEG.exists():
        return {}
    b = json.loads(BEG.read_text())
    gws = b.get("gateways", []) if isinstance(b, dict) else []
    return {g["id"]: safe(g.get("plain", "")) for g in gws}


def main():
    data = json.loads(SRC.read_text())
    secs = data["sections"] if isinstance(data, dict) else data
    by = {s["id"]: s for s in secs}
    gws = load_gateways()
    order = [i for i in ORDER if i in by] + [s["id"] for s in secs if s["id"] not in ORDER]
    parts = []
    for sid in order:
        s = by[sid]
        gate = (f'<div class="gateway"><div class="gk">💡 쉬운 말로 먼저</div>'
                f'<p>{gws[sid]}</p></div>') if sid in gws else ""
        body = gate + "\n".join(render_block(b) for b in s.get("blocks", []))
        parts.append(
            f'<details class="deepsec reveal" open id="deep-{s["id"]}">'
            f'<summary class="dhead"><span class="dn">{s["n"]}</span>'
            f'<span class="dt">{s["title"]}</span><span class="chev">▸</span></summary>'
            f'<div class="dbody">{body}</div></details>'
        )
    frag = "\n".join(parts)
    OUT.write_text(frag)
    nblocks = sum(len(s.get("blocks", [])) for s in secs)
    print(f"[out] {OUT}  {len(secs)} sections, {nblocks} blocks, {len(frag)//1024} KB")
    for s in secs:
        print(f"  {s['n']:>6} {s['id']:<6} blocks={len(s.get('blocks',[])):>2}  {s['title'][:40]}")


if __name__ == "__main__":
    main()
