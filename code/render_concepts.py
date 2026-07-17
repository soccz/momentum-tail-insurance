"""Render the beginner-layer concept clusters into an HTML fragment (__CONCEPTS__).
Reads output/web/beginner.json -> concept clusters of {term,emoji,analogy,plain,precise,example,matters}.
"""
import json
from pathlib import Path

SRC = Path("/mnt/20t/졸업논문/output/web/beginner.json")
OUT = Path("/mnt/20t/졸업논문/output/web/concepts_fragment.html")

CLUSTER_EMOJI = {"주식·모멘텀 기초": "🧱", "위험과 분포": "🎢",
                 "성과와 팩터": "🎯", "예측·계량 핵심": "🔮"}


def safe(s):
    return (s or "").replace("̂", "ˆ").replace("̄", "ˉ")


def concept_card(c):
    rows = [
        ("쉽게", c.get("plain", ""), ""),
        ("정확히", c.get("precise", ""), ""),
        ("예시", c.get("example", ""), " ex"),
        ("이 논문에선", c.get("matters", ""), ""),
    ]
    body = "".join(
        f'<div class="crow{cls}"><span class="cl">{lbl}</span>{safe(val)}</div>'
        for lbl, val, cls in rows if val
    )
    return (
        f'<details class="concept reveal">'
        f'<summary><span class="cemoji">{c.get("emoji","•")}</span>'
        f'<span class="cterm">{safe(c.get("term",""))}</span>'
        f'<span class="cana">{safe(c.get("analogy",""))}</span>'
        f'<span class="cchev">▸</span></summary>'
        f'<div class="cbody">{body}</div></details>'
    )


def main():
    b = json.loads(SRC.read_text())
    clusters = b.get("concepts", []) if isinstance(b, dict) else b
    parts = []
    for cl in clusters:
        name = cl.get("cluster", "")
        emoji = CLUSTER_EMOJI.get(name, "▪")
        cards = "".join(concept_card(c) for c in cl.get("concepts", []))
        parts.append(
            f'<div class="cluster">'
            f'<div class="cluster-title">{emoji} {name}</div>'
            f'<div class="cluster-grid">{cards}</div></div>'
        )
    frag = "\n".join(parts)
    OUT.write_text(frag)
    n = sum(len(cl.get("concepts", [])) for cl in clusters)
    print(f"[out] {OUT}  {len(clusters)} clusters, {n} concepts, {len(frag)//1024} KB")
    for cl in clusters:
        print(f"  {cl.get('cluster'):<16} concepts={len(cl.get('concepts',[]))}")


if __name__ == "__main__":
    main()
