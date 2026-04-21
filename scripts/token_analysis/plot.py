#!/usr/bin/env python3
"""
Lê o JSON gerado por analyze.py e produz figuras em reports/token_analysis/.

  - token_analysis_context_fit.png/.pdf  — barras horizontais (entrada + geração + folga vs janela)
  - by_k.png/.pdf — uma linha por modelo: tokens totais vs k (0 → 32)

Uso:

  python3 scripts/token_analysis/plot.py
  python3 scripts/token_analysis/plot.py --input reports/token_analysis/report_....json
  python3 scripts/token_analysis/plot.py --out-dir reports/token_analysis/figures
  python3 scripts/token_analysis/plot.py --response-quartiles
  python3 scripts/token_analysis/plot.py --response-quartiles --quartiles-input reports/token_analysis/response_size_quartiles_by_model.json
  python3 scripts/token_analysis/plot.py --prompt-quartiles
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_TA = ROOT / "reports" / "token_analysis"

C_INPUT = "#4C72B0"
C_OUTPUT = "#DD8452"
C_MARGIN = "#55A868"
C_DEFICIT = "#C44E52"
C_LIMIT = "#2d2d2d"


def load_report(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_quartiles_report(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _stats_ok(block: Any) -> bool:
    return isinstance(block, dict) and "q1" in block and "median" in block


def plot_response_size_quartiles(
    report: Dict[str, Any],
    out_prefix_base: Path,
    field: str,
    xlabel: str,
    title_line2: str,
    title_line1: Optional[str] = None,
) -> None:
    """
    Uma figura por `field` (`utf8_bytes` ou `tokens`): por modelo, traço min–máx,
    barra horizontal Q1–Q3, segmento na mediana, losango na média.
    """
    rows = report.get("by_model") or []
    rows = [r for r in rows if _stats_ok(r.get(field))]
    if not rows:
        print(f"Sem dados para plotar ({field}).")
        return

    n = len(rows)
    fig_h = max(6.0, n * 0.46)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    y = np.arange(n)
    bar_h = 0.52
    xmax = 0.0

    for i, row in enumerate(rows):
        s = row[field]
        q1, med, q3 = float(s["q1"]), float(s["median"]), float(s["q3"])
        lo, hi = float(s["min"]), float(s["max"])
        mean = float(s["mean"])
        xmax = max(xmax, hi, mean, q3)

        ax.plot(
            [lo, hi],
            [i, i],
            color="#b8b8b8",
            linewidth=1.35,
            solid_capstyle="round",
            zorder=1,
        )
        ax.barh(
            i,
            max(0.0, q3 - q1),
            left=q1,
            height=bar_h,
            color=C_INPUT,
            alpha=0.88,
            zorder=2,
        )
        ax.plot(
            [med, med],
            [i - bar_h / 2, i + bar_h / 2],
            color="#1a1a1a",
            linewidth=2.4,
            solid_capstyle="projecting",
            zorder=4,
        )
        ax.scatter(
            [mean],
            [i],
            s=38,
            c=C_OUTPUT,
            marker="D",
            zorder=5,
            edgecolors="#333333",
            linewidths=0.6,
        )

    labels = [r.get("label_short", "?") for r in rows]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.6)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_xlim(0, xmax * 1.06 if xmax > 0 else 1.0)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(round(v)):,}"))
    ax.grid(axis="x", linestyle="--", alpha=0.32, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ntot = report.get("files_total_valid", "?")
    line1 = title_line1 or "Tamanho da resposta por modelo"
    ax.set_title(f"{line1}\n{title_line2}", fontsize=10, loc="left")
    patches = [
        mpatches.Patch(color=C_INPUT, alpha=0.88, label="IQR (Q1–Q3)"),
        Line2D([0], [0], color="#1a1a1a", linewidth=2.4, label="Mediana"),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor=C_OUTPUT,
            markeredgecolor="#333",
            markersize=7,
            linewidth=0,
            label="Média",
        ),
        Line2D([0], [0], color="#b8b8b8", linewidth=1.35, label="Min–máx"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=8, framealpha=0.94)
    plt.tight_layout()
    base = Path(out_prefix_base)
    fig.savefig(base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  {base}.png/pdf")


def plot_response_size_quartiles_figures(
    report: Dict[str, Any], out_dir: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_response_size_quartiles(
        report,
        out_dir / "response_size_quartiles_bytes",
        "utf8_bytes",
        "Bytes UTF-8",
        "Cinzento: min–máx · Azul: IQR · Traço escuro: mediana · Losango: média",
    )
    if report.get("no_tokens_flag"):
        print("  (tokens omitidos: relatório gerado com --no-tokens)")
        return
    rows = report.get("by_model") or []
    if any(isinstance(r.get("tokens"), dict) and "error" in r["tokens"] for r in rows):
        print("  Aviso: alguns modelos sem bloco tokens válido; plot de tokens só com ok.")
    # Só plotar tokens se todos os rows têm stats ok em tokens
    tok_rows = [r for r in rows if _stats_ok(r.get("tokens"))]
    if tok_rows:
        rep_tok = dict(report)
        rep_tok["by_model"] = tok_rows
        plot_response_size_quartiles(
            rep_tok,
            out_dir / "response_size_quartiles_tokens",
            "tokens",
            "Tokens",
            "Cinzento: min–máx · Azul: IQR · Traço escuro: mediana · Losango: média",
        )


def plot_prompt_input_quartiles_figures(
    report: Dict[str, Any], out_dir: Path
) -> None:
    """Figura única: tokens de entrada (campo `tokens` no JSON do script prompt_input_*)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = report.get("by_model") or []
    tok_rows = [r for r in rows if _stats_ok(r.get("tokens"))]
    if not tok_rows:
        print("Sem dados tokens no relatório de prompts; não gera figura.")
        return
    rep_tok = dict(report)
    rep_tok["by_model"] = tok_rows
    ntot = report.get("files_total_valid", "?")
    plot_response_size_quartiles(
        rep_tok,
        out_dir / "prompt_input_quartiles_tokens",
        "tokens",
        "Tokens de entrada (PromptEngine type2 + apply_chat_template; tokenizer de cada modelo)",
        "Cinzento: min–máx · Azul: IQR · Traço escuro: mediana · Losango: média",
        title_line1=(
            f"Tamanho do prompt de entrada type2 por modelo — n≈{ntot} extrações"
        ),
    )


def _k_worst_keys(meta: Dict[str, Any]) -> Tuple[str, str]:
    k = int(meta.get("k_reference_worst_case", 32))
    return f"k{k}_input", f"k{k}_total_with_generation"


def models_ok(report: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    meta = report.get("meta", {})
    _, k_tot_key = _k_worst_keys(meta)
    by_id: Dict[str, Dict[str, Any]] = {}
    for hf_id, m in report.get("models", {}).items():
        if m.get("status") == "ok" and m.get("tokens"):
            by_id[hf_id] = m

    order = meta.get("dissertation_model_order")
    if order:
        rows = [(hid, by_id[hid]) for hid in order if hid in by_id]
        # Tabela: 1.ª linha no topo; em barh y=0 é em baixo → inverter
        return list(reversed(rows))

    out = list(by_id.items())
    out.sort(key=lambda x: x[1]["tokens"].get(k_tot_key, 0), reverse=True)
    return out


def models_ok_dissertation_order(report: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """Mesma ordem da tabela da dissertação (sem inverter eixo Y)."""
    meta = report.get("meta", {})
    by_id: Dict[str, Dict[str, Any]] = {}
    for hf_id, m in report.get("models", {}).items():
        if m.get("status") == "ok" and m.get("tokens"):
            by_id[hf_id] = m

    order = meta.get("dissertation_model_order")
    if order:
        return [(hid, by_id[hid]) for hid in order if hid in by_id]

    rows = list(by_id.items())
    rows.sort(key=lambda x: x[0])
    return rows


def plot_context_fit(report: Dict[str, Any], out_prefix: Path) -> None:
    meta = report.get("meta", {})
    max_new = int(meta.get("max_new_tokens", 600))
    k_in_key, k_tot_key = _k_worst_keys(meta)
    k_ref = int(meta.get("k_reference_worst_case", 32))
    rows = models_ok(report)
    if not rows:
        print("Nenhum modelo com status=ok; não gera context_fit.")
        return

    n = len(rows)
    fig_h = max(6.0, n * 0.45)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    y = np.arange(n)
    bar_h = 0.62

    for i, (_, m) in enumerate(rows):
        tok = m["tokens"]
        w_in = int(tok[k_in_key])
        lim = int(m["effective_window"])
        total = int(tok[k_tot_key])
        margin = lim - total

        ax.barh(y[i], w_in, height=bar_h, left=0, color=C_INPUT, zorder=3)
        ax.barh(y[i], max_new, height=bar_h, left=w_in, color=C_OUTPUT, zorder=3)
        if margin >= 0:
            ax.barh(y[i], margin, height=bar_h, left=total, color=C_MARGIN, alpha=0.55, zorder=3)
        else:
            ax.barh(y[i], -margin, height=bar_h, left=lim, color=C_DEFICIT, alpha=0.75, hatch="////", zorder=3)

        ax.vlines(lim, y[i] - bar_h / 2, y[i] + bar_h / 2, color=C_LIMIT, linewidth=2.2, zorder=5)
        ax.text(total + max(lim, total) * 0.02, y[i], f"{total:,}", va="center", ha="left", fontsize=7.5)

    labels = [m["label_short"] for _, m in rows]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Tokens", fontsize=10)
    xmax = max(int(m["effective_window"]) for _, m in rows)
    xmax = max(xmax, max(int(m["tokens"][k_tot_key]) for _, m in rows)) * 1.08
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.grid(axis="x", linestyle="--", alpha=0.35, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    pmid = meta.get("longest_pmid", "?")
    ax.set_title(
        f"Ajuste à janela — maior artigo BC5CDR (pmid={pmid}), k={k_ref}\n"
        f"Azul: entrada · Laranja: +{max_new} geração · Verde: folga · Vermelho: déficit · | : janela efetiva",
        fontsize=10,
        loc="left",
    )
    patches = [
        mpatches.Patch(color=C_INPUT, label="Tokens de entrada"),
        mpatches.Patch(color=C_OUTPUT, label=f"Reserva geração ({max_new})"),
        mpatches.Patch(color=C_MARGIN, alpha=0.55, label="Folga até janela efetiva"),
        mpatches.Patch(color=C_DEFICIT, alpha=0.75, hatch="////", label="Déficit"),
        mpatches.Patch(color=C_LIMIT, label="Janela efetiva (min 16k, vLLM max)"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=7.5, framealpha=0.92)
    plt.tight_layout()
    base = Path(out_prefix)
    fig.savefig(base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  {base}.png/pdf")


def plot_by_k(report: Dict[str, Any], out_prefix: Path) -> None:
    meta = report.get("meta", {})
    ks = list(meta.get("ks", [0, 1, 2, 4, 8, 16, 32]))
    rows = models_ok_dissertation_order(report)
    if not rows:
        print("Nenhum modelo ok; não gera by_k.")
        return

    fig, ax = plt.subplots(figsize=(11, 6.2))
    cmap = plt.get_cmap("tab20")
    n = len(rows)
    ymax = 0.0

    for i, (_, m) in enumerate(rows):
        by_k = m["tokens"].get("by_k_total_with_generation") or {}
        ys: List[float] = []
        xs: List[int] = []
        for k in ks:
            v = by_k.get(str(k))
            if v is not None:
                xs.append(int(k))
                ys.append(float(v))
                ymax = max(ymax, float(v))
        if not xs:
            continue
        color = cmap((i % 20) / 20.0)
        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=4,
            linewidth=1.35,
            label=m.get("label_short", "?"),
            color=color,
            alpha=0.9,
        )

    ax.axhline(8192, color="#E88F4A", linestyle="-.", linewidth=1.5, label="8 192", zorder=0)
    ax.axhline(16384, color="#3DA15A", linestyle=":", linewidth=1.6, label="16 384", zorder=0)

    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks], fontsize=9)
    ax.set_xlabel("k (nº de exemplos few-shot; 0 = zero-shot)", fontsize=10)
    ax.set_ylabel("Tokens totais (entrada + geração)", fontsize=10)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    y_top = min(17500, max(ymax * 1.08, 1000))
    ax.set_ylim(0, y_top)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.grid(axis="x", linestyle=":", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    pmid = meta.get("longest_pmid", "?")
    ax.set_title(
        f"Tokens por tokenizer e por k — maior artigo BC5CDR (pmid={pmid})\n"
        f"Uma linha por modelo: k=0 … k=32 · {n} modelos",
        fontsize=10,
        loc="left",
    )

    ax.legend(
        ncol=3,
        fontsize=6.8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        framealpha=0.95,
        columnspacing=0.9,
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    base = Path(out_prefix)
    fig.savefig(base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  {base}.png/pdf")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=REPORTS_TA / "latest.json",
        help="JSON do analyze.py",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Pasta de saída (default: reports/token_analysis/figures)",
    )
    parser.add_argument(
        "--response-quartiles",
        action="store_true",
        help="Gera só figuras a partir de response_size_quartiles_by_model.json",
    )
    parser.add_argument(
        "--quartiles-input",
        type=Path,
        default=REPORTS_TA / "response_size_quartiles_by_model.json",
        help="JSON do response_size_quartiles_by_model.py",
    )
    parser.add_argument(
        "--prompt-quartiles",
        action="store_true",
        help="Figuras a partir de prompt_input_quartiles_by_model.json (entrada)",
    )
    parser.add_argument(
        "--prompt-quartiles-input",
        type=Path,
        default=REPORTS_TA / "prompt_input_quartiles_by_model.json",
        help="JSON do prompt_input_quartiles_by_model.py",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or (REPORTS_TA / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.prompt_quartiles:
        if not args.prompt_quartiles_input.exists():
            print(
                f"Ficheiro não encontrado: {args.prompt_quartiles_input}\n"
                "Corre primeiro: python3 scripts/token_analysis/prompt_input_quartiles_by_model.py"
            )
            return 1
        prep = load_quartiles_report(args.prompt_quartiles_input)
        print(f"A plotar quartis (prompt entrada) a partir de {args.prompt_quartiles_input}")
        plot_prompt_input_quartiles_figures(prep, out_dir)
        return 0

    if args.response_quartiles:
        if not args.quartiles_input.exists():
            print(
                f"Ficheiro não encontrado: {args.quartiles_input}\n"
                "Corre primeiro: python3 scripts/token_analysis/response_size_quartiles_by_model.py"
            )
            return 1
        qrep = load_quartiles_report(args.quartiles_input)
        print(f"A plotar quartis a partir de {args.quartiles_input}")
        plot_response_size_quartiles_figures(qrep, out_dir)
        return 0

    if not args.input.exists():
        print(f"Ficheiro não encontrado: {args.input}\nCorre primeiro: python3 scripts/token_analysis/analyze.py")
        return 1

    report = load_report(args.input)
    if report.get("schema_version") is None:
        print("Aviso: JSON sem schema_version (pode ser legado).")

    print(f"A plotar a partir de {args.input}")
    plot_context_fit(report, out_dir / "context_fit")
    plot_by_k(report, out_dir / "by_k")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
