"""
SwitchLingua 2.0 — Stage 1.5: LLM-Based Post-Generation Quality Evaluation

基于 SwitchLingua 1.0 的 4-Agent 评估框架，改造为生成后的批量评估系统。
与 Stage 1 的 rule-based evaluator 互补：rule-based 捕捉结构性问题，
LLM-based 评估整体语感、文化适配和对话连贯性等难以规则化的维度。

4 个评估 Agent + 1 个汇总 Agent：
  1. FluencyAgent:      双语语法与流畅度（LLM 视角）
  2. NaturalnessAgent:   CS 切换自然度（是否像真人说的）
  3. SocioCultureAgent:  社会文化适当性（语域、地域、身份）
  4. CSPatternAgent:     CS 模式与 archetype 一致性
  5. SummaryAgent:       汇总 4 个 Agent 的评估，给出最终判定

Usage:
    python stage1.5/llm_evaluator.py \
        --input output/zh_en_dialogues.jsonl \
        --output output/stage1.5/zh_en_eval.jsonl \
        --api-base http://localhost:8001/v1 \
        --model Qwen3.5-122B-A10B-FP8 \
        --sample-rate 0.1 \
        --lang-pair zh_en
"""

import json
import time
import random
import logging
import argparse
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("stage1.5")


# ============================================================
# LLM Client (reuse from Stage 1)
# ============================================================

class LLMClient:
    """Simple vLLM-compatible API client."""

    def __init__(self, api_base: str, api_key: str = "empty",
                 model: str = "", timeout: int = 120):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()

    def chat(self, system_prompt: str, user_prompt: str,
             temperature: float = 0.3, max_tokens: int = 1024) -> str:
        url = f"{self.api_base}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if "qwen" in self.model.lower() or "Qwen" in self.model:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        resp = self.session.post(
            url, json=payload, timeout=self.timeout,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


# ============================================================
# Evaluation Agents
# ============================================================

FLUENCY_SYSTEM = """\
You are a bilingual language quality assessor. Your task is to evaluate the \
grammatical correctness and fluency of code-switched text.

Evaluate the following dimensions:
1. L1 Grammar: Is the L1 (primary language) portion grammatically correct?
2. L2 Grammar: Is the L2 (secondary language) portion grammatically correct?
3. Coherence: Does the text read smoothly as a whole? Are there awkward breaks or unnatural phrasing?
4. Completeness: Is the text a complete, natural utterance (not truncated or garbled)?

Rate each dimension 1-5 (1=very poor, 5=excellent).
Provide a brief explanation for each rating.

Output in this EXACT JSON format:
{
  "l1_grammar": {"score": <1-5>, "reason": "<brief explanation>"},
  "l2_grammar": {"score": <1-5>, "reason": "<brief explanation>"},
  "coherence": {"score": <1-5>, "reason": "<brief explanation>"},
  "completeness": {"score": <1-5>, "reason": "<brief explanation>"},
  "overall": <1-5>
}"""

NATURALNESS_SYSTEM = """\
You are a sociolinguistics expert specializing in code-switching. Your task is \
to evaluate whether the code-switching in the given text sounds natural — like \
something a real bilingual speaker would actually say in daily conversation.

Evaluate the following dimensions:
1. Switch Points: Do the language switches occur at natural positions? (Not mid-word, not breaking fixed phrases)
2. Switch Motivation: Does each switch have a plausible reason? (terminology, emphasis, quotation, habit, discourse marker)
3. Ping-Pong Effect: Does the text avoid excessive back-and-forth switching that feels artificial?
4. Authenticity: Overall, would you believe a real bilingual person said this?

Rate each dimension 1-5 (1=very unnatural, 5=perfectly natural).

Output in this EXACT JSON format:
{
  "switch_points": {"score": <1-5>, "reason": "<brief explanation>"},
  "switch_motivation": {"score": <1-5>, "reason": "<brief explanation>"},
  "ping_pong": {"score": <1-5>, "reason": "<brief explanation>"},
  "authenticity": {"score": <1-5>, "reason": "<brief explanation>"},
  "overall": <1-5>
}"""

SOCIOCULTURE_SYSTEM = """\
You are a cultural anthropologist and sociolinguist. Your task is to evaluate \
the social and cultural appropriateness of code-switched dialogue.

Given the speaker's profile (region, age, profession, formality level) and the \
dialogue context, evaluate:
1. Register Match: Does the language register match the formality level? (formal/casual/semi-formal)
2. Regional Authenticity: Does the CS pattern match the claimed region's bilingual norms? \
   (e.g., Singapore English particles, Hong Kong Cantonese-English mixing patterns)
3. Professional Plausibility: Does the vocabulary and CS style fit the speaker's claimed profession?
4. Relationship Dynamics: Is the language appropriate for the stated relationship between speakers?

Rate each dimension 1-5 (1=completely inappropriate, 5=perfectly appropriate).

Output in this EXACT JSON format:
{
  "register_match": {"score": <1-5>, "reason": "<brief explanation>"},
  "regional_authenticity": {"score": <1-5>, "reason": "<brief explanation>"},
  "professional_plausibility": {"score": <1-5>, "reason": "<brief explanation>"},
  "relationship_dynamics": {"score": <1-5>, "reason": "<brief explanation>"},
  "overall": <1-5>
}"""

CS_PATTERN_SYSTEM = """\
You are a code-switching researcher. Your task is to evaluate whether the \
code-switching pattern in the text matches the expected behavioral archetype.

The archetypes are:
- ARC_01 (Insertional): L1 dominant, single L2 words/phrases embedded (1-3 words), intra-sentential
- ARC_02 (Alternational): Switches at sentence/clause boundaries, each language internally complete
- ARC_03 (Dense Mixer): High-frequency intra-sentential mixing, languages deeply fused
- ARC_04 (Pragmatic): Switches serve specific functions (emphasis, humor, quotation)
- ARC_05 (Reluctant): Nearly monolingual, only brand names/proper nouns in L2
- ARC_06 (Accommodation): Adapts CS level to conversation partner
- ARC_07 (Backflag): L2 dominant, only L1 discourse markers/interjections retained

Evaluate:
1. Archetype Match: Does the actual CS pattern match the expected archetype?
2. CMI Consistency: Is the code-mixing density consistent with the expected range?
3. Switch Type: Are the switch types (intra/inter-sentential) consistent with the archetype?
4. Consistency Across Turns: Does the speaker maintain a consistent CS pattern across all their turns?

Rate each dimension 1-5 (1=completely mismatched, 5=perfect match).

Output in this EXACT JSON format:
{
  "archetype_match": {"score": <1-5>, "reason": "<brief explanation>"},
  "cmi_consistency": {"score": <1-5>, "reason": "<brief explanation>"},
  "switch_type": {"score": <1-5>, "reason": "<brief explanation>"},
  "cross_turn_consistency": {"score": <1-5>, "reason": "<brief explanation>"},
  "overall": <1-5>
}"""

SUMMARY_SYSTEM = """\
You are the chief evaluator for a code-switching dataset quality assessment. \
You receive evaluation results from 4 specialized agents (Fluency, Naturalness, \
SocioCulture, CS Pattern). Your job is to:

1. Synthesize the 4 evaluations into an overall quality assessment
2. Identify the most critical issues (if any)
3. Make a final decision: PASS / REVIEW / FAIL
4. Provide a brief overall comment

Decision criteria:
- PASS: All agents scored >= 3.5 overall, no critical issues
- REVIEW: Any agent scored 2.5-3.5, or mixed signals between agents
- FAIL: Any agent scored < 2.5, or multiple agents below 3.0

Output in this EXACT JSON format:
{
  "decision": "<PASS/REVIEW/FAIL>",
  "overall_score": <1.0-5.0>,
  "critical_issues": ["<issue1>", "<issue2>", ...],
  "strengths": ["<strength1>", ...],
  "comment": "<2-3 sentence overall assessment>"
}"""


# ============================================================
# Agent Runner
# ============================================================

def _format_dialogue_text(dialogue: dict) -> str:
    """Format dialogue turns into readable text."""
    lines = []
    for turn in dialogue.get("turns", []):
        speaker = turn["speaker"]
        text = turn["text"]
        lines.append(f"[Speaker {speaker}]: {text}")
    return "\n".join(lines)


def _format_speaker_info(dialogue: dict) -> str:
    """Format speaker metadata for context."""
    parts = []
    for key in ["speaker_a", "speaker_b"]:
        s = dialogue.get(key, {})
        parts.append(
            f"Speaker {key[-1].upper()}: "
            f"archetype={s.get('archetype_id', '?')}, "
            f"region={s.get('region', '?')}, "
            f"profession={s.get('profession', '?')}, "
            f"L2_proficiency={s.get('L2_proficiency', '?')}, "
            f"language_mode={s.get('language_mode', '?')}, "
            f"effective_cmi={s.get('effective_cmi', '?')}"
        )
    return "\n".join(parts)


def _parse_json_response(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (```json and ```)
        json_lines = []
        started = False
        for line in lines:
            if line.strip().startswith("```") and not started:
                started = True
                continue
            if line.strip() == "```" and started:
                break
            if started:
                json_lines.append(line)
        text = "\n".join(json_lines)

    # Try to find JSON object in text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse JSON: {text[:200]}...")
        return {"overall": 3, "parse_error": True}


def run_agent(llm: LLMClient, system_prompt: str, user_prompt: str,
              agent_name: str) -> dict:
    """Run a single evaluation agent and parse its response."""
    try:
        raw = llm.chat(system_prompt, user_prompt, temperature=0.3)
        result = _parse_json_response(raw)
        result["_agent"] = agent_name
        return result
    except Exception as e:
        logger.error(f"{agent_name} failed: {e}")
        return {"overall": 0, "_agent": agent_name, "error": str(e)}


# ============================================================
# Main Evaluation Pipeline
# ============================================================

def evaluate_dialogue(llm: LLMClient, dialogue: dict) -> dict:
    """Run all 4 agents + summary on a single dialogue."""

    dlg_id = dialogue.get("dialogue_id", "unknown")
    dlg_text = _format_dialogue_text(dialogue)
    speaker_info = _format_speaker_info(dialogue)
    topic = dialogue.get("topic", "unknown")
    formality = dialogue.get("formality", "unknown")
    relationship = dialogue.get("relationship", "unknown")

    context_block = (
        f"Topic: {topic}\n"
        f"Formality: {formality}\n"
        f"Relationship: {relationship}\n"
        f"\n{speaker_info}\n"
        f"\n--- Dialogue ---\n{dlg_text}"
    )

    # 1. Fluency Agent
    fluency_prompt = (
        f"Evaluate the fluency of this code-switched dialogue:\n\n"
        f"{context_block}"
    )
    fluency = run_agent(llm, FLUENCY_SYSTEM, fluency_prompt, "fluency")

    # 2. Naturalness Agent
    naturalness_prompt = (
        f"Evaluate the naturalness of code-switching in this dialogue:\n\n"
        f"{context_block}"
    )
    naturalness = run_agent(llm, NATURALNESS_SYSTEM, naturalness_prompt, "naturalness")

    # 3. SocioCulture Agent
    socioculture_prompt = (
        f"Evaluate the social and cultural appropriateness of this dialogue:\n\n"
        f"{context_block}"
    )
    socioculture = run_agent(llm, SOCIOCULTURE_SYSTEM, socioculture_prompt, "socioculture")

    # 4. CS Pattern Agent
    cs_pattern_prompt = (
        f"Evaluate whether the code-switching pattern matches the expected archetypes:\n\n"
        f"{context_block}"
    )
    cs_pattern = run_agent(llm, CS_PATTERN_SYSTEM, cs_pattern_prompt, "cs_pattern")

    # 5. Summary Agent
    summary_input = (
        f"Here are the evaluations from 4 specialized agents for dialogue {dlg_id}:\n\n"
        f"=== Fluency Agent ===\n{json.dumps(fluency, ensure_ascii=False, indent=2)}\n\n"
        f"=== Naturalness Agent ===\n{json.dumps(naturalness, ensure_ascii=False, indent=2)}\n\n"
        f"=== SocioCulture Agent ===\n{json.dumps(socioculture, ensure_ascii=False, indent=2)}\n\n"
        f"=== CS Pattern Agent ===\n{json.dumps(cs_pattern, ensure_ascii=False, indent=2)}\n\n"
        f"Original dialogue context:\n{context_block}\n\n"
        f"Please synthesize these evaluations and make your final decision."
    )
    summary = run_agent(llm, SUMMARY_SYSTEM, summary_input, "summary")

    return {
        "dialogue_id": dlg_id,
        "agents": {
            "fluency": fluency,
            "naturalness": naturalness,
            "socioculture": socioculture,
            "cs_pattern": cs_pattern,
        },
        "summary": summary,
        "agent_scores": {
            "fluency": fluency.get("overall", 0),
            "naturalness": naturalness.get("overall", 0),
            "socioculture": socioculture.get("overall", 0),
            "cs_pattern": cs_pattern.get("overall", 0),
        },
        "final_decision": summary.get("decision", "UNKNOWN"),
        "final_score": summary.get("overall_score", 0),
    }


# ============================================================
# Batch Processing
# ============================================================

def load_dialogues(input_path: str, sample_rate: float = 1.0,
                   sample_seed: int = 42) -> list[dict]:
    """Load dialogues from JSONL, optionally sampling a subset."""
    dialogues = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                dialogues.append(json.loads(line))

    if sample_rate < 1.0:
        random.seed(sample_seed)
        n = max(1, int(len(dialogues) * sample_rate))
        dialogues = random.sample(dialogues, n)
        logger.info(f"Sampled {n}/{len(dialogues)} dialogues (rate={sample_rate})")

    return dialogues


def run_batch_evaluation(llm: LLMClient, dialogues: list[dict],
                         output_path: str, delay: float = 1.0):
    """Evaluate a batch of dialogues and write results."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    stats = {"pass": 0, "review": 0, "fail": 0, "error": 0}

    with open(output_path, "w", encoding="utf-8") as f:
        for i, dlg in enumerate(dialogues):
            dlg_id = dlg.get("dialogue_id", f"dlg_{i}")
            logger.info(f"[{i+1}/{len(dialogues)}] Evaluating {dlg_id}...")

            try:
                result = evaluate_dialogue(llm, dlg)
                results.append(result)

                decision = result["final_decision"].upper()
                if decision in stats:
                    stats[decision] += 1
                else:
                    stats["error"] += 1

                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()

                scores = result["agent_scores"]
                logger.info(
                    f"  → {decision} (score={result['final_score']:.1f}) | "
                    f"Flu={scores['fluency']} Nat={scores['naturalness']} "
                    f"Soc={scores['socioculture']} CSP={scores['cs_pattern']}"
                )

            except Exception as e:
                logger.error(f"  → ERROR: {e}")
                stats["error"] += 1
                f.write(json.dumps({
                    "dialogue_id": dlg_id, "error": str(e),
                    "final_decision": "ERROR"
                }, ensure_ascii=False) + "\n")

            if delay > 0:
                time.sleep(delay)

    # Print summary
    total = len(dialogues)
    print(f"\n{'='*60}")
    print(f"Stage 1.5 Evaluation Complete: {total} dialogues")
    print(f"{'='*60}")
    print(f"  PASS:   {stats['pass']:4d} ({stats['pass']/max(total,1)*100:.1f}%)")
    print(f"  REVIEW: {stats['review']:4d} ({stats['review']/max(total,1)*100:.1f}%)")
    print(f"  FAIL:   {stats['fail']:4d} ({stats['fail']/max(total,1)*100:.1f}%)")
    print(f"  ERROR:  {stats['error']:4d}")
    print(f"{'='*60}")

    if results:
        avg_scores = {}
        for key in ["fluency", "naturalness", "socioculture", "cs_pattern"]:
            scores = [r["agent_scores"][key] for r in results
                      if isinstance(r["agent_scores"].get(key), (int, float))]
            if scores:
                avg_scores[key] = sum(scores) / len(scores)
        print(f"\nAverage Agent Scores:")
        for key, avg in avg_scores.items():
            print(f"  {key:20s}: {avg:.2f} / 5.0")

        final_scores = [r["final_score"] for r in results
                        if isinstance(r.get("final_score"), (int, float)) and r["final_score"] > 0]
        if final_scores:
            print(f"  {'overall':20s}: {sum(final_scores)/len(final_scores):.2f} / 5.0")

    return results


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="SwitchLingua 2.0 — Stage 1.5: LLM-Based Quality Evaluation"
    )
    parser.add_argument("--input", required=True, help="Stage 1 JSONL file")
    parser.add_argument("--output", required=True, help="Output evaluation JSONL")
    parser.add_argument("--api-base", required=True, help="vLLM API base URL")
    parser.add_argument("--api-key", default="empty", help="API key")
    parser.add_argument("--model", required=True, help="Model name/path")
    parser.add_argument("--sample-rate", type=float, default=1.0,
                        help="Fraction of dialogues to evaluate (0.0-1.0)")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Delay between dialogues (seconds)")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--lang-pair", default="zh_en",
                        help="Language pair ID (for logging)")
    args = parser.parse_args()

    logger.info(f"Stage 1.5 LLM Evaluation — lang_pair={args.lang_pair}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Sample rate: {args.sample_rate}")

    llm = LLMClient(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        timeout=args.timeout,
    )

    dialogues = load_dialogues(args.input, args.sample_rate, args.sample_seed)
    logger.info(f"Loaded {len(dialogues)} dialogues for evaluation")

    run_batch_evaluation(llm, dialogues, args.output, delay=args.delay)


if __name__ == "__main__":
    main()
