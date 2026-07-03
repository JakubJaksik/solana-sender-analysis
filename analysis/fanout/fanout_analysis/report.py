"""Assemble report.md + report.html from section result dicts."""
import html
from pathlib import Path

import pandas as pd

CAVEATS = [
    "All CIs are within-run sampling over the run's triggers, so this is not a persistent-ranking "
    "claim. Multi-run replication is the recommended next step.",
    "Client resident in Frankfurt (FRA); observation source is 100% ShredStream (SS), so "
    "UnknownPending may include landed-but-unobserved (right-censored).",
    "Throttle confound: jito-multi / syncro-fra / blockrazor were rate-limited "
    "(client min_send_interval_ms + provider HTTP-429), so judge them on the conditional estimand.",
    "wall_prepared_to_send is broken (pre-anchor artifact) and excluded; only send-RTT, "
    "send->observed, trigger->observed latencies are valid.",
    "Latency is Landed-only (selection bias) and unpaired (no two-lander trigger).",
    "Geo cells are thin (only 6 leaders have >=20 triggers) - per-country / per-validator are "
    "descriptive only, gated n>=20 inferential / 5-19 indicative / <5 suppressed.",
]


def _df_to_md(path, max_rows=30):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return f"_(could not read {path}: {e})_"
    note = "" if len(df) <= max_rows else f"\n\n_(showing {max_rows} of {len(df)} rows)_"
    return df.head(max_rows).to_markdown(index=False) + note


def build_report(section_results, outdir, manifest):
    outdir = Path(outdir)
    plots_rel = "plots"
    md = ["# Tick-Trigger Fan-Out - Sender Benchmark Report",
          "", f"**Run:** `{manifest.get('run_id')}` · epoch {manifest.get('epoch')} · "
          f"{manifest.get('n_rows')} rows · {manifest.get('n_triggers')} triggers · "
          f"{manifest.get('n_senders')} senders", "",
          "## ⚠️ Caveats (read first)", ""]
    md += [f"- {c}" for c in CAVEATS]
    md.append("")

    for res in section_results:
        if not res:
            continue
        md.append(f"## [{res.get('id','?')}] {res.get('title','')}")
        md.append("")
        kr = res.get("key_results") or {}
        if kr:
            md.append("**Key results:**")
            md.append("")
            for k, v in kr.items():
                md.append(f"- `{k}`: {v}")
            md.append("")
        captions = res.get("captions") or {}
        for name, fig in (res.get("figures") or {}).items():
            fp = Path(fig)
            if fp.suffix == ".html":
                md.append(f"- 🌍 interactive (download & open locally): [{name}]({plots_rel}/{fp.name})")
            else:
                md.append(f"![{name}]({plots_rel}/{fp.name})")
            cap = captions.get(name)
            if cap:
                md.append("")
                md.append(f"*{cap}*")
            md.append("")
        md.append("")
        for name, tbl in (res.get("tables") or {}).items():
            md.append(f"**{name}**")
            md.append("")
            md.append(_df_to_md(tbl))
            md.append("")
        for n in (res.get("notes") or []):
            md.append(f"> {n}")
        md.append("")

    md_text = "\n".join(md)
    (outdir / "report.md").write_text(md_text)

    # minimal HTML wrapper
    body = html.escape(md_text)
    html_doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Fan-Out Sender Report {manifest.get('run_id')}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;line-height:1.5}}
img{{max-width:100%;border:1px solid #ddd;margin:.5rem 0}} pre{{white-space:pre-wrap}}
table{{border-collapse:collapse}} td,th{{border:1px solid #ccc;padding:2px 6px;font-size:13px}}</style>
</head><body>
<p><em>Markdown source: report.md. Figures in plots/. This HTML is a plain wrapper; open report.md
in a markdown viewer for rendered tables/images.</em></p>
<pre>{body}</pre>
</body></html>"""
    (outdir / "report.html").write_text(html_doc)
    return outdir / "report.md", outdir / "report.html"
