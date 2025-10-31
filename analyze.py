#!/usr/bin/env python3
"""
Per-tribe NFT trait rarity analysis

Features:
- Works with CSV schema: tribe,tier,minted,shiny,animated,special
- CLI per-tribe report: --tribe "Name"
- Cross-tribe simulation to estimate rarity rankings
- Exports CSVs of metrics and simulation results
- Generates two kinds of HTML:
  1) Rankings page: --html rankings.html
  2) Simple projections page (exact columns order) with --default --html default.html

Both HTML pages support client-side sorting via clickable header arrows on numeric columns.
"""

import argparse
import sys
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import binom

# -------------------------
# Constants and parameters
# -------------------------
DEFAULT_CAP = 1243
P = {"shiny": 0.05, "animated": 0.05, "special": 0.0025}
TRAITS = ["shiny", "animated", "special"]


# -------------------------
# Data loading and metrics
# -------------------------
def load_data(path, cap):
    df = pd.read_csv(path)
    expected_cols = {"tribe", "tier", "minted", "shiny", "animated", "special"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"data.csv is missing columns: {sorted(missing)}")
    df = df.copy()
    # Normalize types
    df["tribe"] = df["tribe"].astype(str)
    df["tier"] = df["tier"].astype(str)
    df["minted"] = df["minted"].astype(int)
    for t in TRAITS:
        df[t] = df[t].astype(int)
    # Remaining cannot be negative
    df["remaining"] = (cap - df["minted"]).clip(lower=0)
    return df


def per_trait_metrics(df, trait, p, cap, alpha=0.05, thresholds=None):
    out = []
    q5_full = int(binom.ppf(0.05, cap, p))
    for _, row in df.iterrows():
        obs = int(row[trait])
        rem = int(row["remaining"])
        exp_final = obs + p * rem  # projected (expected) final at full mint
        lo = int(obs + binom.ppf(alpha / 2, rem, p))
        hi = int(obs + binom.ppf(1 - alpha / 2, rem, p))
        item = {
            "tribe": row["tribe"],
            "tier": row["tier"],
            f"{trait}_obs": obs,
            f"{trait}_expected_final": float(exp_final),
            f"{trait}_pi95_lo": lo,
            f"{trait}_pi95_hi": hi,
            f"{trait}_q5_full": q5_full,
        }
        if thresholds:
            for name, T in thresholds.items():
                t_use = q5_full if (name == "t5pct_full" and T is None) else T
                k_needed = t_use - obs
                if k_needed < 0:
                    prob = 0.0
                else:
                    prob = float(binom.cdf(k_needed, rem, p))
                item[f"{trait}_Pr_final_≤{name}"] = prob
        out.append(item)
    return pd.DataFrame(out)


def build_metrics(df, cap, shiny_threshold=10, animated_threshold=10, special_threshold=1):
    shiny_df = per_trait_metrics(
        df, "shiny", P["shiny"], cap,
        thresholds={"t10": shiny_threshold, "t5pct_full": None}
    )
    anim_df = per_trait_metrics(
        df, "animated", P["animated"], cap,
        thresholds={"t10": animated_threshold, "t5pct_full": None}
    )
    spec_df = per_trait_metrics(
        df, "special", P["special"], cap,
        thresholds={"t1": special_threshold, "t0": 0, "t5pct_full": None}
    )
    metrics = (
        df[["tribe", "tier", "minted", "remaining"]]
        .merge(shiny_df, on=["tribe", "tier"])
        .merge(anim_df, on=["tribe", "tier"])
        .merge(spec_df, on=["tribe", "tier"])
    )
    return metrics


# -------------------------
# Cross-tribe simulation
# -------------------------
def simulate_rarity(df, trait, p, sims=20000, seed=42):
    rng = np.random.default_rng(seed)
    obs = df[trait].to_numpy(dtype=int)
    rem = df["remaining"].to_numpy(dtype=int)
    # Simulate remaining mints for each tribe
    draws = rng.binomial(rem, p, size=(sims, len(df)))  # (sims, n_tribes)
    final = draws + obs
    # Probability of finishing minimum (ties count for all tied)
    mins = final.min(axis=1, keepdims=True)
    is_min = (final == mins)
    prob_min = is_min.mean(axis=0)
    # Ranks: 1 = lowest (rarest)
    order = final.argsort(axis=1)  # ascending
    ranks = np.empty_like(order)
    # invert permutation to get rank positions
    for i in range(order.shape[0]):
        ranks[i, order[i]] = np.arange(order.shape[1])
    exp_rank = ranks.mean(axis=0) + 1
    prob_bottom3 = (ranks < 3).mean(axis=0)

    return pd.DataFrame({
        "tribe": df["tribe"],
        "tier": df["tier"],
        f"{trait}_Pr_rarest": prob_min,
        f"{trait}_Pr_bottom3": prob_bottom3,
        f"{trait}_Expected_rank": exp_rank
    })


def build_rarity(df, sims=20000, seed=42):
    shiny_sim = simulate_rarity(df, "shiny", P["shiny"], sims=sims, seed=seed)
    anim_sim = simulate_rarity(df, "animated", P["animated"], sims=sims, seed=seed)
    spec_sim = simulate_rarity(df, "special", P["special"], sims=sims, seed=seed)
    rarity = shiny_sim.merge(anim_sim, on=["tribe", "tier"]).merge(spec_sim, on=["tribe", "tier"])
    return rarity


def merge_all(metrics, rarity):
    return metrics.merge(rarity, on=["tribe", "tier"])


# -------------------------
# Formatting helpers
# -------------------------
def fmt_prob(x):
    return f"{x:.3f}"


def fmt_float(x):
    return f"{x:.2f}"


# -------------------------
# CLI printing
# -------------------------
def print_tribe_report(all_df, cap, tribe, shiny_threshold=10, animated_threshold=10, special_threshold=1):
    row = all_df.loc[all_df["tribe"].astype(str) == str(tribe)]
    if row.empty:
        print(f"[!] Tribe not found: {tribe}")
        return 1
    row = row.iloc[0]
    minted = int(row["minted"])
    remaining = int(row["remaining"])
    print(f"Tier: {row['tier']}")
    print(f"Tribe: {row['tribe']}")
    print(f"Minted: {minted}/{cap}  (Remaining: {remaining})")
    print("-" * 60)
    for trait in TRAITS:
        obs = int(row[f"{trait}_obs"])
        exp_final = row[f"{trait}_expected_final"]
        lo = int(row[f"{trait}_pi95_lo"])
        hi = int(row[f"{trait}_pi95_hi"])
        q5_full = int(row[f"{trait}_q5_full"])
        print(f"{trait.capitalize()}:")
        print(f"  Observed: {obs}")
        print(f"  Projected final: {fmt_float(exp_final)}")
        print(f"  95% predictive interval: [{lo}, {hi}]")
        print(f"  5th percentile at full mint baseline: {q5_full}")
        # Threshold tail probabilities
        keys = [k for k in row.index if k.startswith(f"{trait}_Pr_final_≤")]
        for k in sorted(keys):
            name = k.split("≤", 1)[1]
            print(f"  Pr(final ≤ {name}): {fmt_prob(row[k])}")
        # Cross-tribe
        pr_r = row.get(f"{trait}_Pr_rarest", np.nan)
        pr_b3 = row.get(f"{trait}_Pr_bottom3", np.nan)
        exp_rank = row.get(f"{trait}_Expected_rank", np.nan)
        if not np.isnan(pr_r):
            print(f"  Cross-tribe Pr(rarest): {fmt_prob(pr_r)}")
            print(f"  Cross-tribe Pr(bottom-3): {fmt_prob(pr_b3)}")
            print(f"  Cross-tribe expected rank (1 = rarest): {fmt_float(exp_rank)}")
        print("")
    return 0


def print_ranking(all_df, trait, metric, top=None, ascending=None):
    if ascending is None:
        # For Expected_rank, smaller is rarer; for probabilities, larger means rarer
        ascending = (metric in ["Expected_rank", f"{trait}_Expected_rank"])
    col = metric if metric.startswith(trait) or metric in ["Expected_rank"] else f"{trait}_{metric}"
    if col not in all_df.columns:
        raise ValueError(f"Unknown metric column: {col}")
    df = all_df[["tribe", "tier", "minted", "remaining"] + [
        f"{trait}_obs", f"{trait}_expected_final", f"{trait}_pi95_lo", f"{trait}_pi95_hi",
        f"{trait}_Pr_rarest", f"{trait}_Pr_bottom3", f"{trait}_Expected_rank"
    ]].copy()
    df = df.sort_values(col, ascending=ascending)
    if top:
        df = df.head(top)
    print(f"Ranking for {trait} by {col} ({'asc' if ascending else 'desc'}):")
    print("-" * 60)
    for _, r in df.iterrows():
        print(f"[{r['tier']}] {r['tribe']:<22}  minted {int(r['minted']):4d}  "
              f"{trait}_obs={int(r[f'{trait}_obs']):4d}  "
              f"proj={fmt_float(r[f'{trait}_expected_final'])}  "
              f"PI95=[{int(r[f'{trait}_pi95_lo'])},{int(r[f'{trait}_pi95_hi'])}]  "
              f"Pr_rare={fmt_prob(r[f'{trait}_Pr_rarest'])}  "
              f"Pr_btm3={fmt_prob(r[f'{trait}_Pr_bottom3'])}  "
              f"rank~{fmt_float(r[f'{trait}_Expected_rank'])}")
    print("")


# -------------------------
# HTML generation (sortable)
# -------------------------
def _sortable_js_css():
    # Shared CSS/JS for both HTML pages
    return f"""
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }}
  h1, h2 {{ margin: 8px 0; }}
  .meta {{ color: #666; margin-bottom: 16px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0 32px; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 10px; font-size: 14px; }}
  th {{ background: #f5f5f5; text-align: left; user-select: none; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  code {{ background: #f1f1f1; padding: 2px 4px; border-radius: 3px; }}
  th[data-sort-type="num"] {{ cursor: pointer; }}
  .sort-arrow {{ margin-left: 6px; color: #888; }}
</style>

<script>
(function(){{
  function getVal(td) {{
    const v = td.dataset.value ?? td.textContent.trim();
    return v;
  }}
  function cmp(a, b, type, dir) {{
    if (type === 'num') {{
      const na = parseFloat(a), nb = parseFloat(b);
      if (isNaN(na) && isNaN(nb)) return 0;
      if (isNaN(na)) return 1;
      if (isNaN(nb)) return -1;
      return dir * (na - nb);
    }} else {{
      return dir * String(a).localeCompare(String(b));
    }}
  }}
  function sortTable(table, colIndex, type, dir) {{
    const tbody = table.tBodies[0];
    const rows = Array.from(tbody.rows);
    rows.sort((r1, r2) => {{
      const a = getVal(r1.cells[colIndex]);
      const b = getVal(r2.cells[colIndex]);
      return cmp(a, b, type, dir);
    }});
    rows.forEach(r => tbody.appendChild(r));
  }}
  function setup(table) {{
    const headers = table.tHead ? Array.from(table.tHead.rows[0].cells) : [];
    headers.forEach((th, i) => {{
      const type = th.dataset.sortType;
      const arrow = th.querySelector('.sort-arrow');
      if (!type || !arrow) return;
      let dir = 0; // 0 neutral, 1 asc, -1 desc
      th.addEventListener('click', () => {{
        dir = dir === 1 ? -1 : 1; // toggle asc/desc
        // reset other arrows in this header row
        headers.forEach(h => {{
          if (h !== th) {{
            const a2 = h.querySelector('.sort-arrow');
            if (a2) a2.textContent = '↕';
            h.dataset.sortDir = '';
          }}
        }});
        arrow.textContent = dir === 1 ? '▲' : '▼';
        th.dataset.sortDir = dir;
        sortTable(table, i, type, dir);
      }});
    }});
  }}
  document.addEventListener('DOMContentLoaded', () => {{
    document.querySelectorAll('table.sortable').forEach(setup);
  }});
}})();
</script>
    """


def generate_html(all_df, cap, out_path):
    gen_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # Build three tables (initially sorted by Pr_rarest desc)
    sections = []
    for trait in TRAITS:
        df = all_df.sort_values(f"{trait}_Pr_rarest", ascending=False).copy()
        rows_html = []
        for _, r in df.iterrows():
            rows_html.append(f"""
            <tr>
              <td>{r['tier']}</td>
              <td>{r['tribe']}</td>
              <td data-value="{int(r['minted'])}">{int(r['minted'])}/{cap}</td>
              <td data-value="{int(r[f'{trait}_obs'])}">{int(r[f'{trait}_obs'])}</td>
              <td data-value="{float(r[f'{trait}_expected_final']):.6f}">{fmt_float(r[f'{trait}_expected_final'])}</td>
              <td>[{int(r[f'{trait}_pi95_lo'])}, {int(r[f'{trait}_pi95_hi'])}]</td>
              <td data-value="{float(r[f'{trait}_Pr_rarest']):.6f}">{fmt_prob(r[f'{trait}_Pr_rarest'])}</td>
              <td data-value="{float(r[f'{trait}_Pr_bottom3']):.6f}">{fmt_prob(r[f'{trait}_Pr_bottom3'])}</td>
              <td data-value="{float(r[f'{trait}_Expected_rank']):.6f}">{fmt_float(r[f'{trait}_Expected_rank'])}</td>
            </tr>
            """)
        table = f"""
        <h2>{trait.capitalize()} ranking (sorted by Pr_rarest)</h2>
        <table class="sortable">
          <thead>
            <tr>
              <th>Tier</th>
              <th>Tribe</th>
              <th data-sort-type="num">Minted <span class="sort-arrow">↕</span></th>
              <th data-sort-type="num">Observed <span class="sort-arrow">↕</span></th>
              <th data-sort-type="num">Projected final <span class="sort-arrow">↕</span></th>
              <th>95% PI</th>
              <th data-sort-type="num">Pr(rarest) <span class="sort-arrow">↕</span></th>
              <th data-sort-type="num">Pr(bottom-3) <span class="sort-arrow">↕</span></th>
              <th data-sort-type="num">Expected rank <span class="sort-arrow">↕</span></th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows_html)}
          </tbody>
        </table>
        """
        sections.append(table)

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Tribe rarity rankings</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
{_sortable_js_css()}
</head>
<body>
  <h1>Tribe rarity rankings</h1>
  <div class="meta">
    Generated: {gen_time}<br/>
    Cap per tribe: {cap}<br/>
    Trait odds: shiny={P['shiny']}, animated={P['animated']}, special={P['special']}
  </div>
  {' '.join(sections)}
  <p>Tip: Re-generate after updating data and refresh your browser.
     To serve locally, run <code>python -m http.server</code> in this folder and open
     <code>http://localhost:8000/{os.path.basename(out_path)}</code>.</p>
</body>
</html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


def generate_default_html(metrics_df, out_path):
    """
    Generates a simple table with columns (in this exact order):
    tier, tribe, minted, shiny, projected shiny, animated, projected animated, special, projected special
    Numeric columns are sortable with header arrows.
    """
    gen_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    view = metrics_df[[
        "tier", "tribe", "minted",
        "shiny_obs", "shiny_expected_final",
        "animated_obs", "animated_expected_final",
        "special_obs", "special_expected_final",
    ]].copy().sort_values(["tier", "tribe"])

    def row_to_html(r):
        return f"""
        <tr>
          <td>{r['tier']}</td>
          <td>{r['tribe']}</td>
          <td data-value="{int(r['minted'])}">{int(r['minted'])}</td>
          <td data-value="{int(r['shiny_obs'])}">{int(r['shiny_obs'])}</td>
          <td data-value="{float(r['shiny_expected_final']):.6f}">{fmt_float(r['shiny_expected_final'])}</td>
          <td data-value="{int(r['animated_obs'])}">{int(r['animated_obs'])}</td>
          <td data-value="{float(r['animated_expected_final']):.6f}">{fmt_float(r['animated_expected_final'])}</td>
          <td data-value="{int(r['special_obs'])}">{int(r['special_obs'])}</td>
          <td data-value="{float(r['special_expected_final']):.6f}">{fmt_float(r['special_expected_final'])}</td>
        </tr>
        """

    rows = "\n".join(row_to_html(r) for _, r in view.iterrows())

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Default projections</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
{_sortable_js_css()}
</head>
<body>
  <h1>Projected counts at full mint</h1>
  <div class="meta">Generated: {gen_time}</div>
  <table class="sortable">
    <thead>
      <tr>
        <th>tier</th>
        <th>tribe</th>
        <th data-sort-type="num">minted <span class="sort-arrow">↕</span></th>
        <th data-sort-type="num">shiny <span class="sort-arrow">↕</span></th>
        <th data-sort-type="num">projected shiny <span class="sort-arrow">↕</span></th>
        <th data-sort-type="num">animated <span class="sort-arrow">↕</span></th>
        <th data-sort-type="num">projected animated <span class="sort-arrow">↕</span></th>
        <th data-sort-type="num">shiny animated <span class="sort-arrow">↕</span></th>
        <th data-sort-type="num">projected shiny animated <span class="sort-arrow">↕</span></th>
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
</body>
</html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


# -------------------------
# Main (CLI)
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Per-tribe NFT trait rarity analysis")
    parser.add_argument("--data", required=True, help="Path to data CSV (columns: tribe,tier,minted,shiny,animated,special)")
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP, help=f"Max per tribe (default {DEFAULT_CAP})")
    parser.add_argument("--sims", type=int, default=20000, help="Simulation runs for cross-tribe rarity")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for simulation")

    # Simple default page toggle
    parser.add_argument("--default", action="store_true",
                        help="Generate simple projections table (requires --html PATH).")

    # Per-tribe quick query
    parser.add_argument("--tribe", help="Show detailed stats for a specific tribe name")

    # Rankings
    parser.add_argument("--rank", choices=["shiny", "animated", "special", "all"], help="Show ranking(s)")
    parser.add_argument("--metric", default="Pr_rarest",
                        help="Metric for ranking: Pr_rarest, Pr_bottom3, Expected_rank, expected_final, etc.")
    parser.add_argument("--top", type=int, help="Show only the top N rows (according to sort).")

    # Outputs
    parser.add_argument("--export_csv", action="store_true", help="Export per_tribe_metrics.csv and cross_tribe_rarity.csv")
    parser.add_argument("--html", help="Generate an HTML file at the given path")

    # Thresholds for tail probabilities
    parser.add_argument("--shiny-threshold", type=int, default=10, help="Absolute low threshold for shiny tail prob")
    parser.add_argument("--animated-threshold", type=int, default=10, help="Absolute low threshold for animated tail prob")
    parser.add_argument("--special-threshold", type=int, default=1, help="Absolute low threshold for special tail prob")

    args = parser.parse_args()

    if args.default and not args.html:
        print("[!] --default requires --html <output.html>")
        sys.exit(2)

    try:
        df = load_data(args.data, args.cap)
    except Exception as e:
        print(f"[!] Failed to load data: {e}")
        sys.exit(1)

    # Always build metrics (needed for projections and reports)
    metrics = build_metrics(
        df, args.cap,
        shiny_threshold=args.shiny_threshold,
        animated_threshold=args.animated_threshold,
        special_threshold=args.special_threshold
    )

    # Only build rarity if needed (saves time for --default-only)
    need_ranking = bool(args.tribe or args.rank or (args.html and not args.default) or args.export_csv)
    rarity = build_rarity(df, sims=args.sims, seed=args.seed) if need_ranking else pd.DataFrame(columns=["tribe", "tier"])

    all_df = merge_all(metrics, rarity) if not rarity.empty else metrics

    # Export CSVs if requested (requires rarity to be computed)
    if args.export_csv:
        if rarity.empty:
            rarity = build_rarity(df, sims=args.sims, seed=args.seed)
            all_df = merge_all(metrics, rarity)
        metrics.to_csv("per_tribe_metrics.csv", index=False)
        rarity.to_csv("cross_tribe_rarity.csv", index=False)
        print("Wrote per_tribe_metrics.csv and cross_tribe_rarity.csv")

    exit_code = 0
    if args.tribe:
        # Ensure we have rarity for cross-tribe lines in the report
        if rarity.empty:
            rarity = build_rarity(df, sims=args.sims, seed=args.seed)
            all_df = merge_all(metrics, rarity)
        exit_code = print_tribe_report(
            all_df, args.cap, args.tribe,
            shiny_threshold=args.shiny_threshold,
            animated_threshold=args.animated_threshold,
            special_threshold=args.special_threshold
        )

    if args.rank:
        if rarity.empty:
            rarity = build_rarity(df, sims=args.sims, seed=args.seed)
            all_df = merge_all(metrics, rarity)
        traits = TRAITS if args.rank == "all" else [args.rank]
        for trait in traits:
            metric = args.metric
            if metric == "expected_final":
                metric = f"{trait}_expected_final"
            elif metric in ["Pr_rarest", "Pr_bottom3", "Expected_rank"]:
                metric = metric if metric == "Expected_rank" else f"{trait}_{metric}"
            print_ranking(all_df, trait, metric, top=args.top)

    if args.html:
        if args.default:
            out = generate_default_html(metrics, args.html)
            print(f"Wrote default projections HTML: {out}")
        else:
            if rarity.empty:
                rarity = build_rarity(df, sims=args.sims, seed=args.seed)
                all_df = merge_all(metrics, rarity)
            out = generate_html(all_df, args.cap, args.html)
            print(f"Wrote rankings HTML: {out}")

    if not (args.tribe or args.rank or args.html or args.export_csv or args.default):
        print("Analysis complete. Use flags like --tribe \"Name\" or --rank shiny --metric Pr_rarest.")
        print("Run with -h for full options.")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()