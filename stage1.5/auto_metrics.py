"""
SwitchLingua 2.0 — Automatic Descriptive Metrics for CS Dialogues

计算描述性统计指标（非质量评判），用于论文的 Supplementary Analysis。
所有指标量化的是"分布特征"而非"好坏"——好坏由人工评估决定。

指标：
  1. CMI 分布统计 (mean, std, percentiles, histogram)
  2. 切换类型分布 (intra / inter / tag 比例)
  3. 切换位置密度 (switches per sentence)
  4. L2 span 长度分布
  5. 内容多样性 (Distinct-1/2/3, Self-BLEU)
  6. Archetype 可辨识性 (特征聚类 + 分布熵)
  7. 与参考语料库的 KL 散度 (CMI 分布)

Usage:
    python stage1.5/auto_metrics.py \
        --input output/zh_en_dialogues.jsonl \
        --report output/metrics/zh_en_metrics.md \
        --reference-cmi-mean 0.27 --reference-cmi-std 0.18

    # 对比多个系统
    python stage1.5/auto_metrics.py \
        --input ours.jsonl baseline1.jsonl baseline2.jsonl \
        --labels "SwitchLingua 2.0" "Naive Prompting" "EZSwitch" \
        --report output/metrics/comparison.md
"""

import json
import math
import argparse
import logging
import sys
from pathlib import Path
from collections import Counter, defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("auto_metrics")

# Add infrastructure to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage1_infrastructure"))

try:
    from evaluator_agents import TextAnalyzer, TextAnalysis
except ImportError:
    logger.warning("Could not import TextAnalyzer. CMI/switch metrics will be unavailable.")
    TextAnalyzer = None


# ============================================================
# Metric Computations
# ============================================================

def compute_cmi_stats(dialogues: list[dict], analyzer=None) -> dict:
    """Compute CMI distribution statistics from dialogues."""
    cmis = []

    for dlg in dialogues:
        for turn in dlg.get("turns", []):
            # Use pre-computed CMI if available
            if "cmi" in turn and turn["cmi"] is not None:
                cmis.append(float(turn["cmi"]))
            elif analyzer and "text" in turn:
                analysis = analyzer.analyze(turn["text"])
                cmis.append(analysis.cmi)

    if not cmis:
        return {"count": 0}

    cmis_sorted = sorted(cmis)
    n = len(cmis)
    mean = sum(cmis) / n
    variance = sum((x - mean) ** 2 for x in cmis) / n
    std = variance ** 0.5

    return {
        "count": n,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "median": round(cmis_sorted[n // 2], 4),
        "p25": round(cmis_sorted[n // 4], 4),
        "p75": round(cmis_sorted[3 * n // 4], 4),
        "min": round(cmis_sorted[0], 4),
        "max": round(cmis_sorted[-1], 4),
        "raw": cmis,
    }


def compute_switch_type_distribution(dialogues: list[dict], analyzer=None) -> dict:
    """Compute intra/inter/tag switching proportions."""
    intra = inter = tag = 0

    for dlg in dialogues:
        for turn in dlg.get("turns", []):
            text = turn.get("text", "")
            if not text or not analyzer:
                continue
            analysis = analyzer.analyze(text)
            for sp in analysis.switch_points:
                if sp.switch_type == "intra":
                    intra += 1
                elif sp.switch_type == "inter":
                    inter += 1
                else:
                    tag += 1

    total = intra + inter + tag
    if total == 0:
        return {"total": 0}

    return {
        "total": total,
        "intra": round(intra / total, 3),
        "inter": round(inter / total, 3),
        "tag": round(tag / total, 3),
        "intra_count": intra,
        "inter_count": inter,
        "tag_count": tag,
    }


def compute_l2_span_stats(dialogues: list[dict], analyzer=None) -> dict:
    """Compute L2 span length distribution."""
    spans = []

    for dlg in dialogues:
        for turn in dlg.get("turns", []):
            text = turn.get("text", "")
            if not text or not analyzer:
                continue
            analysis = analyzer.analyze(text)
            for seg in analysis.segments:
                if seg["lang"] == analysis.l2_code:
                    l2_tokens = sum(1 for t in seg["tokens"] if t.lang == analysis.l2_code)
                    if l2_tokens > 0:
                        spans.append(l2_tokens)

    if not spans:
        return {"count": 0}

    spans_sorted = sorted(spans)
    n = len(spans)
    mean = sum(spans) / n

    # Distribution
    dist = Counter(min(s, 10) for s in spans)  # cap at 10+

    return {
        "count": n,
        "mean": round(mean, 2),
        "median": spans_sorted[n // 2],
        "max": spans_sorted[-1],
        "distribution": {k: dist[k] for k in sorted(dist.keys())},
    }


def compute_distinct_n(dialogues: list[dict], max_n: int = 3) -> dict:
    """Compute Distinct-1/2/3 (content diversity metric)."""
    all_texts = []
    for dlg in dialogues:
        for turn in dlg.get("turns", []):
            text = turn.get("text", "")
            if text:
                all_texts.append(text)

    results = {}
    for n in range(1, max_n + 1):
        all_ngrams = []
        for text in all_texts:
            words = text.split()
            ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
            all_ngrams.extend(ngrams)

        if all_ngrams:
            distinct = len(set(all_ngrams)) / len(all_ngrams)
            results[f"distinct_{n}"] = round(distinct, 4)
        else:
            results[f"distinct_{n}"] = 0.0

    return results


def compute_self_bleu(dialogues: list[dict], sample_size: int = 500) -> float:
    """
    Compute Self-BLEU (average BLEU of each sample against all others).
    Lower = more diverse. Uses simple unigram overlap as approximation.
    """
    import random

    texts = []
    for dlg in dialogues:
        for turn in dlg.get("turns", []):
            text = turn.get("text", "")
            if text:
                texts.append(set(text.split()))

    if len(texts) < 2:
        return 0.0

    if len(texts) > sample_size:
        random.seed(42)
        texts = random.sample(texts, sample_size)

    total_bleu = 0.0
    count = 0
    for i, hyp in enumerate(texts):
        if not hyp:
            continue
        others = [texts[j] for j in range(len(texts)) if j != i]
        # Unigram precision against pooled references
        ref_pool = set()
        for ref in others[:50]:  # limit for speed
            ref_pool.update(ref)
        overlap = len(hyp & ref_pool)
        precision = overlap / len(hyp) if hyp else 0
        total_bleu += precision
        count += 1

    return round(total_bleu / max(count, 1), 4)


def compute_archetype_distribution(dialogues: list[dict]) -> dict:
    """Compute archetype distribution and entropy."""
    arc_counts = Counter()
    for dlg in dialogues:
        for key in ["speaker_a", "speaker_b"]:
            arc_id = dlg.get(key, {}).get("archetype_id", "unknown")
            arc_counts[arc_id] += 1

    total = sum(arc_counts.values())
    if total == 0:
        return {"entropy": 0, "distribution": {}}

    dist = {k: round(v / total, 3) for k, v in sorted(arc_counts.items())}

    # Shannon entropy
    entropy = 0.0
    for count in arc_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    return {
        "entropy": round(entropy, 3),
        "max_entropy": round(math.log2(len(arc_counts)), 3),
        "distribution": dist,
        "counts": dict(sorted(arc_counts.items())),
    }


def compute_kl_divergence(p_values: list[float], q_mean: float, q_std: float,
                          num_bins: int = 20) -> float:
    """
    Compute KL divergence D(P || Q) where P is empirical and Q is Gaussian.
    Used to compare generated CMI distribution against reference corpus.
    """
    if not p_values or q_std <= 0:
        return float('inf')

    # Build histogram for P
    min_val, max_val = 0.0, max(max(p_values), q_mean + 3 * q_std)
    bin_width = (max_val - min_val) / num_bins
    if bin_width <= 0:
        return float('inf')

    p_hist = [0] * num_bins
    for v in p_values:
        idx = min(int((v - min_val) / bin_width), num_bins - 1)
        p_hist[idx] += 1

    # Normalize P
    n = len(p_values)
    p_probs = [(c / n) if c > 0 else 1e-10 for c in p_hist]

    # Build Q (Gaussian)
    q_probs = []
    for i in range(num_bins):
        center = min_val + (i + 0.5) * bin_width
        # Gaussian PDF
        q = (1 / (q_std * math.sqrt(2 * math.pi))) * \
            math.exp(-0.5 * ((center - q_mean) / q_std) ** 2)
        q_probs.append(max(q * bin_width, 1e-10))

    # Normalize Q
    q_sum = sum(q_probs)
    q_probs = [q / q_sum for q in q_probs]

    # KL divergence
    kl = 0.0
    for p, q in zip(p_probs, q_probs):
        if p > 1e-10:
            kl += p * math.log(p / q)

    return round(kl, 4)


def compute_topic_distribution(dialogues: list[dict]) -> dict:
    """Compute topic distribution."""
    topics = Counter(dlg.get("topic", "unknown") for dlg in dialogues)
    total = sum(topics.values())
    return {k: round(v / total, 3) for k, v in sorted(topics.items())}


# ============================================================
# Report Generation
# ============================================================

def generate_report(results: dict, labels: list[str] = None) -> str:
    """Generate markdown report from computed metrics."""
    lines = ["# SwitchLingua 2.0 — Automatic Descriptive Metrics Report\n"]

    if isinstance(results, list):
        # Multi-system comparison
        lines.append("## System Comparison\n")

        # CMI table
        lines.append("### CMI Distribution\n")
        lines.append("| System | N | Mean | Std | Median | P25 | P75 | KL(vs SEAME) |")
        lines.append("|--------|---|------|-----|--------|-----|-----|-------------|")
        for i, r in enumerate(results):
            label = labels[i] if labels else f"System {i+1}"
            cmi = r.get("cmi", {})
            kl = r.get("kl_divergence", "—")
            lines.append(
                f"| {label} | {cmi.get('count', 0)} | {cmi.get('mean', 0):.3f} | "
                f"{cmi.get('std', 0):.3f} | {cmi.get('median', 0):.3f} | "
                f"{cmi.get('p25', 0):.3f} | {cmi.get('p75', 0):.3f} | {kl} |"
            )
        lines.append("")

        # Switch type table
        lines.append("### Switch Type Distribution\n")
        lines.append("| System | Intra | Inter | Tag | Total Switches |")
        lines.append("|--------|-------|-------|-----|---------------|")
        for i, r in enumerate(results):
            label = labels[i] if labels else f"System {i+1}"
            sw = r.get("switch_types", {})
            lines.append(
                f"| {label} | {sw.get('intra', 0):.1%} | {sw.get('inter', 0):.1%} | "
                f"{sw.get('tag', 0):.1%} | {sw.get('total', 0)} |"
            )
        lines.append("")
        lines.append("*SEAME reference: intra=63%, inter=27%, tag=10%*\n")

        # Diversity table
        lines.append("### Content Diversity\n")
        lines.append("| System | Distinct-1 | Distinct-2 | Distinct-3 | Self-BLEU |")
        lines.append("|--------|-----------|-----------|-----------|-----------|")
        for i, r in enumerate(results):
            label = labels[i] if labels else f"System {i+1}"
            div = r.get("diversity", {})
            sb = r.get("self_bleu", "—")
            lines.append(
                f"| {label} | {div.get('distinct_1', 0):.3f} | "
                f"{div.get('distinct_2', 0):.3f} | {div.get('distinct_3', 0):.3f} | {sb} |"
            )
        lines.append("")

        # Archetype distribution
        lines.append("### Archetype Distribution\n")
        arcs = set()
        for r in results:
            arcs.update(r.get("archetype", {}).get("distribution", {}).keys())
        arcs = sorted(arcs)

        header = "| System | " + " | ".join(arcs) + " | Entropy |"
        sep = "|--------|" + "|".join(["------"] * len(arcs)) + "|---------|"
        lines.append(header)
        lines.append(sep)
        for i, r in enumerate(results):
            label = labels[i] if labels else f"System {i+1}"
            ad = r.get("archetype", {}).get("distribution", {})
            ent = r.get("archetype", {}).get("entropy", 0)
            vals = " | ".join(f"{ad.get(a, 0):.1%}" for a in arcs)
            lines.append(f"| {label} | {vals} | {ent:.2f} |")
        lines.append("")

    else:
        # Single system
        r = results
        lines.append(f"## CMI Distribution (N={r.get('cmi', {}).get('count', 0)})\n")
        cmi = r.get("cmi", {})
        for k in ["mean", "std", "median", "p25", "p75", "min", "max"]:
            lines.append(f"- {k}: {cmi.get(k, 0):.4f}")
        lines.append("")

        lines.append("## Switch Type Distribution\n")
        sw = r.get("switch_types", {})
        lines.append(f"- Intra-sentential: {sw.get('intra', 0):.1%} ({sw.get('intra_count', 0)})")
        lines.append(f"- Inter-sentential: {sw.get('inter', 0):.1%} ({sw.get('inter_count', 0)})")
        lines.append(f"- Tag-switching: {sw.get('tag', 0):.1%} ({sw.get('tag_count', 0)})")
        lines.append("")

        lines.append("## L2 Span Length\n")
        sp = r.get("l2_spans", {})
        lines.append(f"- Mean: {sp.get('mean', 0):.2f} words")
        lines.append(f"- Median: {sp.get('median', 0)}")
        lines.append(f"- Max: {sp.get('max', 0)}")
        lines.append("")

        lines.append("## Content Diversity\n")
        div = r.get("diversity", {})
        for k, v in div.items():
            lines.append(f"- {k}: {v:.4f}")
        lines.append(f"- Self-BLEU: {r.get('self_bleu', 0):.4f}")
        lines.append("")

        lines.append("## Archetype Distribution\n")
        ad = r.get("archetype", {})
        lines.append(f"- Entropy: {ad.get('entropy', 0):.3f} / {ad.get('max_entropy', 0):.3f}")
        for arc, prop in ad.get("distribution", {}).items():
            count = ad.get("counts", {}).get(arc, 0)
            lines.append(f"- {arc}: {prop:.1%} ({count})")
        lines.append("")

        lines.append("## Topic Distribution\n")
        for topic, prop in r.get("topics", {}).items():
            lines.append(f"- {topic}: {prop:.1%}")
        lines.append("")

        if "kl_divergence" in r:
            lines.append(f"## KL Divergence vs Reference\n")
            lines.append(f"- KL(P_generated || Q_reference): {r['kl_divergence']:.4f}")
            lines.append("")

    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def analyze_single(input_path: str, analyzer=None,
                   ref_cmi_mean: float = None, ref_cmi_std: float = None) -> dict:
    """Analyze a single JSONL file."""
    dialogues = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                dialogues.append(json.loads(line))

    logger.info(f"Loaded {len(dialogues)} dialogues from {input_path}")

    results = {}

    # 1. CMI
    logger.info("Computing CMI distribution...")
    results["cmi"] = compute_cmi_stats(dialogues, analyzer)

    # 2. Switch types
    if analyzer:
        logger.info("Computing switch type distribution...")
        results["switch_types"] = compute_switch_type_distribution(dialogues, analyzer)

        logger.info("Computing L2 span lengths...")
        results["l2_spans"] = compute_l2_span_stats(dialogues, analyzer)
    else:
        results["switch_types"] = {}
        results["l2_spans"] = {}

    # 3. Diversity
    logger.info("Computing content diversity (Distinct-n)...")
    results["diversity"] = compute_distinct_n(dialogues)

    logger.info("Computing Self-BLEU...")
    results["self_bleu"] = compute_self_bleu(dialogues)

    # 4. Archetype distribution
    results["archetype"] = compute_archetype_distribution(dialogues)

    # 5. Topic distribution
    results["topics"] = compute_topic_distribution(dialogues)

    # 6. KL divergence
    if ref_cmi_mean is not None and ref_cmi_std is not None:
        raw_cmis = results["cmi"].get("raw", [])
        if raw_cmis:
            results["kl_divergence"] = compute_kl_divergence(
                raw_cmis, ref_cmi_mean, ref_cmi_std
            )

    # Clean up raw data
    if "raw" in results.get("cmi", {}):
        del results["cmi"]["raw"]

    return results


def main():
    parser = argparse.ArgumentParser(
        description="SwitchLingua 2.0 — Automatic Descriptive Metrics"
    )
    parser.add_argument("--input", nargs="+", required=True,
                        help="One or more JSONL files to analyze")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Labels for each input file (for comparison mode)")
    parser.add_argument("--report", default=None,
                        help="Output markdown report path")
    parser.add_argument("--reference-cmi-mean", type=float, default=0.27,
                        help="Reference corpus CMI mean (SEAME default: 0.27)")
    parser.add_argument("--reference-cmi-std", type=float, default=0.18,
                        help="Reference corpus CMI std (SEAME default: 0.18)")
    parser.add_argument("--lang-pair", default=None,
                        help="Language pair for TextAnalyzer (e.g., zh_en)")
    args = parser.parse_args()

    # Setup analyzer
    analyzer = None
    if TextAnalyzer:
        if args.lang_pair:
            try:
                from language_config import LanguagePairConfig
                cfg = LanguagePairConfig.load(args.lang_pair)
                analyzer = TextAnalyzer(lang_config=cfg)
                logger.info(f"TextAnalyzer initialized for {args.lang_pair}")
            except Exception as e:
                logger.warning(f"Could not load lang config: {e}")
                analyzer = TextAnalyzer()
        else:
            analyzer = TextAnalyzer()

    # Analyze
    if len(args.input) == 1:
        results = analyze_single(
            args.input[0], analyzer,
            args.reference_cmi_mean, args.reference_cmi_std,
        )
        report = generate_report(results)
    else:
        all_results = []
        for path in args.input:
            r = analyze_single(
                path, analyzer,
                args.reference_cmi_mean, args.reference_cmi_std,
            )
            all_results.append(r)
        labels = args.labels or [Path(p).stem for p in args.input]
        report = generate_report(all_results, labels)

    # Output
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Report saved to {args.report}")
    else:
        print(report)


if __name__ == "__main__":
    main()
