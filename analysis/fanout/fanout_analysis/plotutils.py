"""Matplotlib/seaborn plot helpers (ported & extended from tick-trigger/analysis.py).

Common style: figsize (11,6), dpi 120, grid alpha 0.3, tight_layout, savefig, close.
Every function takes data + out_path and returns out_path.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402


def _save(fig, out_path):
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_ecdf_multi(series_by_group, title, xlabel, out_path, logx=False, xlim=None):
    """Overlaid ECDF, one step-line per group, legend annotated with p50 + n."""
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, s in sorted(series_by_group.items(),
                          key=lambda kv: (np.median(kv[1]) if len(kv[1]) else 1e18)):
        s = np.sort(np.asarray(s, float))
        s = s[~np.isnan(s)]
        if len(s) == 0:
            continue
        y = np.arange(1, len(s) + 1) / len(s)
        ax.step(s, y, where="post", label=f"{name} (p50={np.median(s):.0f}, n={len(s)})")
    if logx:
        ax.set_xscale("log")
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, out_path)


def plot_hist(series, title, xlabel, out_path, bins=80, vlines=None):
    fig, ax = plt.subplots(figsize=(11, 6))
    s = np.asarray(series, float)
    s = s[~np.isnan(s)]
    ax.hist(s, bins=bins, color="#1976D2", alpha=0.85)
    for q, c in (vlines or {}).items():
        ax.axvline(np.percentile(s, q), color=c, ls="--", alpha=0.7, label=f"p{q}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(f"{title} (n={len(s):,})")
    ax.grid(alpha=0.3)
    if vlines:
        ax.legend()
    return _save(fig, out_path)


def plot_violin(df, x, y, title, out_path, order=None, ylim=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(data=df, x=x, y=y, order=order, cut=0, ax=ax)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if ylim is not None:
        ax.set_ylim(*ylim)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    return _save(fig, out_path)


def plot_percentile_bar(summary_df, group_col, title, ylabel, out_path):
    """summary_df has group_col, p10, p50, p90, p99. Clustered bars per group."""
    melt = summary_df.melt(id_vars=[group_col], value_vars=["p10", "p50", "p90", "p99"],
                           var_name="pct", value_name="ms")
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=melt, x=group_col, y="ms", hue="pct", ax=ax)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    ax.grid(alpha=0.3)
    return _save(fig, out_path)


def forest_plot(rows, title, xlabel, out_path, ref=None, logx=False):
    """rows: list of (label, point, lo, hi). One row per item with horizontal CI."""
    fig, ax = plt.subplots(figsize=(11, max(4, 0.5 * len(rows))))
    ys = list(range(len(rows)))
    for y, (lab, pt, lo, hi) in zip(ys, rows):
        ax.plot([lo, hi], [y, y], color="#555")
        ax.plot(pt, y, "o", color="#1976D2")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    if ref is not None:
        ax.axvline(ref, color="red", ls="--", alpha=0.5)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return _save(fig, out_path)


def heatmap(matrix_df, title, out_path, fmt=".2f", cmap="viridis", center=None, annot=True):
    fig, ax = plt.subplots(figsize=(1.1 * matrix_df.shape[1] + 3, 0.5 * matrix_df.shape[0] + 3))
    sns.heatmap(matrix_df, annot=annot, fmt=fmt, cmap=cmap, center=center, ax=ax,
                cbar_kws={"shrink": .7})
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    return _save(fig, out_path)


def slopegraph(left_vals, right_vals, left_lab, right_lab, title, out_path, color_by=None):
    fig, ax = plt.subplots(figsize=(9, 7))
    for k in left_vals:
        ax.plot([0, 1], [left_vals[k], right_vals[k]], "-o", label=k,
                color=(color_by or {}).get(k))
        ax.text(1.02, right_vals[k], k, fontsize=8, va="center")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([left_lab, right_lab])
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return _save(fig, out_path)


def stacked_bar(df, cols, title, xlabel, out_path, horizontal=True, pct=False):
    data = df[cols] if not pct else df[cols].div(df[cols].sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(12, max(4, 0.5 * len(df))))
    data.plot(kind="barh" if horizontal else "bar", stacked=True, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return _save(fig, out_path)
