"""Build the score tables of ALGORITHMS.md from logs/register/register.log (written by logs/register/run_register.sh).
python make_register.py  -> prints markdown; ALGORITHMS.md embeds it between the REGISTER markers."""
import re
from pathlib import Path

DEV = Path(__file__).resolve().parent
LOG = DEV / "logs" / "register" / "register.log"
DOC = DEV / "ALGORITHMS.md"
SPLITS = ["off_big", "dev", "off_dev", "hold", "off_hold", "off_low", "off_wide"]
HEAD = {"off_big": "official-style 400", "dev": "DEV", "off_dev": "off-DEV", "hold": "HOLD", "off_hold": "off-hold",
        "off_low": "low budget", "off_wide": "wide bounds"}
ROW = re.compile(r"^(\S+)\s+([\d.]+|nan)\s+([\d.]+|nan)\s+([\d.]+|nan)\s+([\d.]+|nan)\s+([\d.]+|nan)\s+([\d.]+|nan)\s+(\d+)")


def parse():
    tab, sec = {}, None
    for line in LOG.read_text().splitlines():
        if line.startswith("== "):
            sec = line[3:].strip()
            continue
        m = ROW.match(line)
        if m and sec:
            name = m.group(1)
            tab.setdefault(sec.replace("milestones ", ""), {})[name] = (float(m.group(2)), float(m.group(3)))
    return tab


def table(tab, names):
    out = ["| algorithm | " + " | ".join(HEAD[s] for s in SPLITS) + " | **mean (7 splits)** | **practice lab (20 seeds)** |",
           "|---|" + "---|" * (len(SPLITS) + 2)]
    for n in names:
        vals = [tab.get(s, {}).get(n) for s in SPLITS]
        cells = [f"{v[0]:.3f}" if v else "–" for v in vals]
        got = [v[0] for v in vals if v]
        mean = f"**{sum(got) / len(got):.3f}**" + ("" if len(got) == len(SPLITS) else f" ({len(got)})") if got else "–"
        pr = tab.get("practice20", {}).get(n)
        out.append(f"| `{n}` | " + " | ".join(cells) + f" | {mean} | {f'{pr[0]:.3f} ± {pr[1]:.3f}' if pr else '–'} |")
    return "\n".join(out)


if __name__ == "__main__":
    tab = parse()
    refs = ["reference_mfbo", "reference_screen_then_confirm", "mfgp_structured"]
    fins = ["generic_funnel", "generic_bayes", "generic_bayes2", "informed_funnel", "informed_library", "informed_bayes", "informed_bayes2"]
    mil = ["i3_funnel", "i6_funnel", "i10_funnel", "i14_funnel", "i17_funnel", "k5_bayes", "k7_voi_s2"]
    md = ("### Reference algorithms\n\n" + table(tab, refs) + "\n\n### Final algorithms (`dev/final/`)\n\n" +
          table(tab, fins) + "\n\n### Development milestones (`dev/algos/`)\n\n" + table(tab, mil) + "\n")
    doc = DOC.read_text()
    a, b = "<!-- REGISTER START -->", "<!-- REGISTER END -->"
    if a in doc:
        doc = doc[:doc.index(a) + len(a)] + "\n" + md + doc[doc.index(b):]
        DOC.write_text(doc)
    print(md)
