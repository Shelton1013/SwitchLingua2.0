"""
SwitchLingua 2.0 — Stage 1.5: Evaluation Results Analyzer

Reads Stage 1.5 evaluation JSONL and generates quality reports.

Usage:
    python stage1.5/analyze_results.py \
        --input output/stage1.5/zh_en_eval.jsonl \
        --report output/stage1.5/zh_en_report.md
"""

import json
import argparse
import logging
from pathlib import Path
from collections import Counter, defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stage1.5_analyzer")


def load_results(path: str) -> list[dict]:
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def generate_report(results: list[dict], original_jsonl: str = None) -> str:
    """Generate a markdown quality report."""

    lines = ["# Stage 1.5 LLM Evaluation Report\n"]

    # --- Overview ---
    total = len(results)
    valid = [r for r in results if r.get("final_decision") != "ERROR"]
    decisions = Counter(r.get("final_decision", "UNKNOWN").upper() for r in valid)

    lines.append(f"## Overview\n")
    lines.append(f"- Total dialogues evaluated: **{total}**")
    lines.append(f"- Valid evaluations: **{len(valid)}**")
    lines.append(f"- PASS: **{decisions.get('PASS', 0)}** ({decisions.get('PASS', 0)/max(len(valid),1)*100:.1f}%)")
    lines.append(f"- REVIEW: **{decisions.get('REVIEW', 0)}** ({decisions.get('REVIEW', 0)/max(len(valid),1)*100:.1f}%)")
    lines.append(f"- FAIL: **{decisions.get('FAIL', 0)}** ({decisions.get('FAIL', 0)/max(len(valid),1)*100:.1f}%)")
    lines.append("")

    # --- Agent Score Distribution ---
    lines.append(f"## Agent Score Distribution\n")
    lines.append(f"| Agent | Mean | Median | Min | Max | Std |")
    lines.append(f"|-------|------|--------|-----|-----|-----|")

    agent_names = ["fluency", "naturalness", "socioculture", "cs_pattern"]
    agent_all_scores = {}

    for agent in agent_names:
        scores = []
        for r in valid:
            s = r.get("agent_scores", {}).get(agent)
            if isinstance(s, (int, float)) and s > 0:
                scores.append(float(s))
        agent_all_scores[agent] = scores

        if scores:
            scores_sorted = sorted(scores)
            mean = sum(scores) / len(scores)
            median = scores_sorted[len(scores) // 2]
            mn, mx = min(scores), max(scores)
            variance = sum((x - mean) ** 2 for x in scores) / len(scores)
            std = variance ** 0.5
            lines.append(f"| {agent:20s} | {mean:.2f} | {median:.1f} | {mn:.0f} | {mx:.0f} | {std:.2f} |")
        else:
            lines.append(f"| {agent:20s} | — | — | — | — | — |")

    # Overall
    final_scores = [r["final_score"] for r in valid
                    if isinstance(r.get("final_score"), (int, float)) and r["final_score"] > 0]
    if final_scores:
        mean = sum(final_scores) / len(final_scores)
        final_sorted = sorted(final_scores)
        median = final_sorted[len(final_scores) // 2]
        lines.append(f"| **overall** | **{mean:.2f}** | **{median:.1f}** | **{min(final_scores):.0f}** | **{max(final_scores):.0f}** | — |")
    lines.append("")

    # --- Score Distribution Histogram (text-based) ---
    lines.append(f"## Score Distribution\n")
    for agent in agent_names:
        scores = agent_all_scores.get(agent, [])
        if not scores:
            continue
        buckets = Counter()
        for s in scores:
            buckets[int(s)] += 1
        lines.append(f"**{agent}:**")
        for i in range(1, 6):
            count = buckets.get(i, 0)
            bar = "█" * count
            lines.append(f"  {i}: {bar} ({count})")
        lines.append("")

    # --- Failure Analysis ---
    failures = [r for r in valid if r.get("final_decision", "").upper() == "FAIL"]
    if failures:
        lines.append(f"## Failure Analysis ({len(failures)} dialogues)\n")

        # Collect critical issues
        all_issues = []
        for r in failures:
            issues = r.get("summary", {}).get("critical_issues", [])
            all_issues.extend(issues)

        issue_counts = Counter(all_issues)
        if issue_counts:
            lines.append("**Most common critical issues:**\n")
            for issue, count in issue_counts.most_common(10):
                lines.append(f"- {issue} ({count}x)")
            lines.append("")

        # Which agent scores lowest in failures?
        lines.append("**Weakest agent in failed dialogues:**\n")
        agent_fail_scores = defaultdict(list)
        for r in failures:
            for agent in agent_names:
                s = r.get("agent_scores", {}).get(agent)
                if isinstance(s, (int, float)):
                    agent_fail_scores[agent].append(s)

        for agent in agent_names:
            scores = agent_fail_scores[agent]
            if scores:
                avg = sum(scores) / len(scores)
                lines.append(f"- {agent}: avg={avg:.2f}")
        lines.append("")

    # --- Per-Archetype Analysis ---
    lines.append(f"## Per-Archetype Analysis\n")
    lines.append(f"| Archetype | Count | Avg Score | Pass Rate |")
    lines.append(f"|-----------|-------|-----------|-----------|")

    # Load original JSONL to get archetype info
    if original_jsonl and Path(original_jsonl).exists():
        dlg_archetypes = {}
        with open(original_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                dlg = json.loads(line.strip())
                dlg_id = dlg.get("dialogue_id")
                arc = dlg.get("speaker_a", {}).get("archetype_id", "?")
                dlg_archetypes[dlg_id] = arc

        arc_results = defaultdict(list)
        for r in valid:
            dlg_id = r.get("dialogue_id")
            arc = dlg_archetypes.get(dlg_id, "unknown")
            arc_results[arc].append(r)

        for arc_id in sorted(arc_results.keys()):
            rs = arc_results[arc_id]
            scores = [r["final_score"] for r in rs
                      if isinstance(r.get("final_score"), (int, float)) and r["final_score"] > 0]
            passes = sum(1 for r in rs if r.get("final_decision", "").upper() == "PASS")
            avg = sum(scores) / len(scores) if scores else 0
            rate = passes / len(rs) * 100 if rs else 0
            lines.append(f"| {arc_id:10s} | {len(rs):5d} | {avg:.2f} | {rate:.1f}% |")
    else:
        lines.append("*(Original JSONL not provided — archetype breakdown unavailable)*")

    lines.append("")

    # --- Recommendations ---
    lines.append(f"## Recommendations\n")
    if final_scores:
        avg_overall = sum(final_scores) / len(final_scores)
        pass_rate = decisions.get("PASS", 0) / max(len(valid), 1) * 100

        if avg_overall >= 4.0 and pass_rate >= 80:
            lines.append("Dataset quality is **GOOD**. Ready for Stage 2 synthesis.")
        elif avg_overall >= 3.5 and pass_rate >= 60:
            lines.append("Dataset quality is **ACCEPTABLE** with some issues. Consider:")
            # Find weakest agent
            weakest = min(agent_all_scores, key=lambda k: sum(agent_all_scores[k]) / max(len(agent_all_scores[k]), 1))
            lines.append(f"- Weakest dimension: **{weakest}** — review prompt templates or evaluation rules")
            lines.append(f"- Filter out FAIL dialogues before Stage 2")
        else:
            lines.append("Dataset quality is **BELOW THRESHOLD**. Recommended actions:")
            lines.append("- Review and improve prompt templates")
            lines.append("- Check if evaluation rules are too strict/lenient")
            lines.append("- Consider regenerating with adjusted parameters")

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1.5 Evaluation Results Analyzer"
    )
    parser.add_argument("--input", required=True, help="Evaluation JSONL from llm_evaluator.py")
    parser.add_argument("--original", default=None, help="Original Stage 1 JSONL (for archetype breakdown)")
    parser.add_argument("--report", default=None, help="Output markdown report path")
    args = parser.parse_args()

    results = load_results(args.input)
    logger.info(f"Loaded {len(results)} evaluation results")

    report = generate_report(results, original_jsonl=args.original)

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Report saved to {args.report}")
    else:
        print(report)


if __name__ == "__main__":
    main()
