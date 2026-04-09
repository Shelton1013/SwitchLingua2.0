"""
SwitchLingua 2.0 — Stage 4 Layer 2: 规则化多维度 CS 质量评估器
Rule-Based Multi-Dimensional CS Quality Evaluator

与 SwitchLingua 1.0 的核心区别：
  1.0: 用 4 个 LLM Agent 独立打分 → 存在 LLM 评 LLM 的 self-enhancement bias
  2.0: 用 6 个规则评分模块 → 全规则化，无 LLM 偏见，可复现，零 API 成本

评估流程：
  1. TextAnalyzer 对输入文本做 token 级语言标注和切换点检测
  2. 6 个 Checker 各自基于规则计算 0-10 分
  3. WeightedAggregator 加权汇总 + 一票否决 → 最终 pass/fail/review

6 个评估维度：
  - FluencyChecker:          双语语法正确性 (ECT/MLF)
  - NaturalnessChecker:      CS 模式自然度 (Grosjean + Poplack)
  - SwitchMotivationChecker: 切换动机可归因性 (Gumperz 1982)
  - ProfileConsistencyChecker: 与说话人画像的一致性
  - CulturalCoherenceChecker:  文化语域一致性 (Myers-Scotton)
  - DiversityChecker:         与已有语料的差异度 (DEITA)
"""

import re
import math
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# ============================================================
# 数据结构
# ============================================================

@dataclass
class TokenInfo:
    """单个 token 的信息"""
    text: str          # token 原文
    lang: str          # 语言标签: "zh", "en", "punct", "num", "other"
    position: int      # 在 token 列表中的位置

@dataclass
class SwitchPoint:
    """一个语言切换点"""
    position: int           # 切换发生的 token 位置（第一个新语言 token）
    from_lang: str          # 切换前的语言
    to_lang: str            # 切换后的语言
    switch_type: str        # "intra" (句内) 或 "inter" (句间)
    context_before: str     # 切换前 3 个 token
    context_after: str      # 切换后 3 个 token
    motivation: str = ""    # 归因的动机类型（由 SwitchMotivationChecker 填充）

@dataclass
class TextAnalysis:
    """TextAnalyzer 的完整分析结果"""
    raw_text: str                   # 原始文本
    tokens: list[TokenInfo]         # token 列表
    zh_tokens: list[TokenInfo]      # 中文 token
    en_tokens: list[TokenInfo]      # 英文 token
    segments: list[dict]            # 连续同语言片段: [{lang, text, tokens, start, end}]
    switch_points: list[SwitchPoint] # 所有切换点
    sentences: list[str]            # 分句结果
    cmi: float                      # Code-Mixing Index
    matrix_language: str            # 基质语言 (zh/en)
    total_content_tokens: int       # 有效内容 token 数（排除标点数字）

@dataclass
class CheckResult:
    """单个 Checker 的评估结果"""
    checker_name: str       # 评估器名称
    score: float            # 0-10 分
    max_score: float = 10.0
    details: dict = field(default_factory=dict)   # 各子项的详细得分
    violations: list[str] = field(default_factory=list)  # 违规项列表
    is_veto: bool = False   # 是否触发一票否决

@dataclass
class EvaluationResult:
    """最终汇总评估结果"""
    text: str                       # 原始文本
    final_score: float              # 加权最终分
    decision: str                   # "pass" / "fail" / "review"
    checker_results: dict[str, CheckResult]  # 各 Checker 的结果
    analysis: TextAnalysis          # 文本分析结果
    veto_reason: str = ""           # 一票否决原因（若有）


# ============================================================
# TextAnalyzer: 文本预处理与语言标注
# ============================================================

class TextAnalyzer:
    """
    对输入的中英混合文本进行：
    1. 分词 + token 级语言标注（基于 Unicode 范围，无需外部 NLP 模型）
    2. 连续同语言片段（segment）识别
    3. 切换点检测与分类（句内 intra / 句间 inter）
    4. CMI (Code-Mixing Index) 计算
    5. 基质语言判定

    语言识别原理：
    - 中文: Unicode CJK 统一汉字区 (\\u4e00-\\u9fff) + 扩展区
    - 英文: ASCII 字母 (a-z, A-Z)
    - 标点: 中英文标点符号
    - 数字: 0-9
    """

    # 中文 Unicode 范围
    _ZH_RANGES = [
        (0x4E00, 0x9FFF),    # CJK Unified Ideographs
        (0x3400, 0x4DBF),    # CJK Extension A
        (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
    ]

    # 句子结束标记
    _SENTENCE_ENDINGS = set("。！？.!?")

    def __init__(self):
        pass

    def _classify_char(self, ch: str) -> str:
        """判断单个字符的语言类型"""
        cp = ord(ch)
        # 中文字符
        for start, end in self._ZH_RANGES:
            if start <= cp <= end:
                return "zh"
        # 英文字母
        if ch.isascii() and ch.isalpha():
            return "en"
        # 数字
        if ch.isdigit():
            return "num"
        # 标点或空白
        return "punct"

    def _tokenize(self, text: str) -> list[TokenInfo]:
        """
        简单分词：
        - 中文：每个汉字作为一个 token
        - 英文：连续字母作为一个 token（即英文单词）
        - 数字：连续数字作为一个 token
        - 标点：每个标点作为一个 token
        - 空白：跳过
        """
        tokens = []
        i = 0
        pos = 0
        while i < len(text):
            ch = text[i]

            if ch.isspace():
                i += 1
                continue

            char_type = self._classify_char(ch)

            if char_type == "zh":
                # 中文：每个字一个 token
                tokens.append(TokenInfo(text=ch, lang="zh", position=pos))
                pos += 1
                i += 1

            elif char_type == "en":
                # 英文：连续字母组成一个单词
                j = i
                while j < len(text) and text[j].isascii() and text[j].isalpha():
                    j += 1
                word = text[i:j]
                tokens.append(TokenInfo(text=word, lang="en", position=pos))
                pos += 1
                i = j

            elif char_type == "num":
                # 数字：连续数字组成一个 token
                j = i
                while j < len(text) and text[j].isdigit():
                    j += 1
                tokens.append(TokenInfo(text=text[i:j], lang="num", position=pos))
                pos += 1
                i = j

            else:
                # 标点
                tokens.append(TokenInfo(text=ch, lang="punct", position=pos))
                pos += 1
                i += 1

        return tokens

    def _build_segments(self, tokens: list[TokenInfo]) -> list[dict]:
        """
        将连续同语言的 token 合并为 segment。
        只考虑 zh 和 en，标点和数字归入相邻的语言段。
        """
        if not tokens:
            return []

        segments = []
        current_lang = None
        current_tokens = []
        start_pos = 0

        for token in tokens:
            # 标点和数字不影响语言判定，归入当前段
            effective_lang = token.lang if token.lang in ("zh", "en") else None

            if effective_lang is None:
                # 标点/数字：跟随当前段
                current_tokens.append(token)
            elif current_lang is None or current_lang == effective_lang:
                # 同语言或首个内容 token
                current_lang = effective_lang
                current_tokens.append(token)
            else:
                # 语言切换：保存当前段，开始新段
                if current_tokens:
                    segments.append({
                        "lang": current_lang,
                        "tokens": current_tokens,
                        "text": "".join(t.text for t in current_tokens),
                        "start": start_pos,
                        "end": current_tokens[-1].position,
                    })
                current_lang = effective_lang
                current_tokens = [token]
                start_pos = token.position

        # 最后一个段
        if current_tokens:
            segments.append({
                "lang": current_lang or "other",
                "tokens": current_tokens,
                "text": "".join(t.text for t in current_tokens),
                "start": start_pos,
                "end": current_tokens[-1].position,
            })

        return segments

    def _detect_switch_points(
        self, segments: list[dict], sentences: list[str]
    ) -> list[SwitchPoint]:
        """
        检测所有语言切换点，并判定切换类型（句内 intra / 句间 inter）。

        判定逻辑：
        - 如果切换点恰好在句子边界 → inter-sentential
        - 否则 → intra-sentential
        """
        switch_points = []

        # 构建句子边界位置集合（用字符级偏移近似）
        sentence_boundaries = set()
        char_pos = 0
        for sent in sentences:
            char_pos += len(sent)
            sentence_boundaries.add(char_pos)

        for i in range(1, len(segments)):
            prev_seg = segments[i - 1]
            curr_seg = segments[i]

            # 只关注 zh ↔ en 的切换
            if prev_seg["lang"] not in ("zh", "en"):
                continue
            if curr_seg["lang"] not in ("zh", "en"):
                continue
            if prev_seg["lang"] == curr_seg["lang"]:
                continue

            # 判断是否在句子边界
            # 用 segment 结束位置近似判断
            prev_text_end = prev_seg["text"]
            is_at_sentence_boundary = (
                prev_text_end and prev_text_end[-1] in self._SENTENCE_ENDINGS
            )

            switch_type = "inter" if is_at_sentence_boundary else "intra"

            # 提取上下文
            prev_tokens = [t.text for t in prev_seg["tokens"] if t.lang in ("zh", "en")]
            curr_tokens = [t.text for t in curr_seg["tokens"] if t.lang in ("zh", "en")]
            context_before = " ".join(prev_tokens[-3:]) if prev_tokens else ""
            context_after = " ".join(curr_tokens[:3]) if curr_tokens else ""

            switch_points.append(SwitchPoint(
                position=curr_seg["start"],
                from_lang=prev_seg["lang"],
                to_lang=curr_seg["lang"],
                switch_type=switch_type,
                context_before=context_before,
                context_after=context_after,
            ))

        return switch_points

    def _split_sentences(self, text: str) -> list[str]:
        """按中英文句末标点分句"""
        sentences = re.split(r'(?<=[。！？.!?])\s*', text)
        return [s.strip() for s in sentences if s.strip()]

    def _compute_cmi(self, zh_count: int, en_count: int) -> float:
        """
        计算 Code-Mixing Index (Gamback & Das, 2014)。
        CMI = (N - max(w_i)) / N
        其中 N = 总内容 token 数, max(w_i) = 最多语言的 token 数。
        CMI ∈ [0, 0.5]，0 = 单语，0.5 = 两种语言完全均分。
        """
        total = zh_count + en_count
        if total == 0:
            return 0.0
        return (total - max(zh_count, en_count)) / total

    def analyze(self, text: str) -> TextAnalysis:
        """对输入文本执行完整分析"""
        tokens = self._tokenize(text)
        zh_tokens = [t for t in tokens if t.lang == "zh"]
        en_tokens = [t for t in tokens if t.lang == "en"]
        segments = self._build_segments(tokens)
        sentences = self._split_sentences(text)
        switch_points = self._detect_switch_points(segments, sentences)
        cmi = self._compute_cmi(len(zh_tokens), len(en_tokens))
        total_content = len(zh_tokens) + len(en_tokens)
        matrix_lang = "zh" if len(zh_tokens) >= len(en_tokens) else "en"

        return TextAnalysis(
            raw_text=text,
            tokens=tokens,
            zh_tokens=zh_tokens,
            en_tokens=en_tokens,
            segments=segments,
            switch_points=switch_points,
            sentences=sentences,
            cmi=cmi,
            matrix_language=matrix_lang,
            total_content_tokens=total_content,
        )


# ============================================================
# Checker 1: FluencyChecker — 双语语法正确性
# ============================================================

class FluencyChecker:
    """
    检查双语文本的基本语法正确性和完整性。

    硬性规则（不满足 → 直接 0 分）：
    - 文本必须包含两种语言
    - token 数在 [5, 500] 范围内
    - 无乱码 / HTML 残留 / 明显截断

    评分项：
    - L1 (中文) 片段质量：字符合法性、片段最小长度
    - L2 (英文) 片段质量：单词拼写合法性
    - 整体完整性：括号配对、句末完整
    """

    # 常见英文词列表（高频 3000 词的子集 + CS 常见词）
    # 实际使用时应从外部文件加载完整词表
    _COMMON_EN_WORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "shall",
        "should", "may", "might", "can", "could", "must", "need", "dare",
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
        "us", "them", "my", "your", "his", "its", "our", "their",
        "this", "that", "these", "those", "what", "which", "who", "whom",
        "and", "or", "but", "not", "no", "so", "if", "then", "than",
        "to", "of", "in", "on", "at", "for", "with", "from", "by",
        "about", "into", "through", "during", "before", "after",
        "up", "down", "out", "off", "over", "under", "again",
        "very", "really", "also", "just", "like", "well", "too",
        # CS 高频词
        "meeting", "deadline", "project", "email", "schedule", "report",
        "algorithm", "debug", "deploy", "database", "api", "code",
        "design", "review", "update", "feedback", "presentation",
        "actually", "basically", "honestly", "anyway", "okay", "sure",
        "make", "sense", "problem", "way", "time", "thing", "point",
        "good", "bad", "nice", "great", "cool", "fine", "okay",
        "oh", "wow", "yes", "no", "yeah", "nah", "hmm",
        "get", "go", "come", "take", "give", "put", "set", "run",
        "say", "tell", "ask", "know", "think", "feel", "want", "try",
        "look", "see", "find", "work", "call", "use", "help",
        "new", "old", "big", "small", "long", "short", "high", "low",
        "first", "last", "next", "other", "same", "different",
        "because", "since", "although", "though", "while", "when",
        "where", "how", "why", "all", "some", "any", "each", "every",
        "much", "many", "more", "most", "few", "little", "enough",
        "here", "there", "now", "then", "today", "tomorrow", "yesterday",
        "always", "never", "often", "sometimes", "usually",
        "right", "wrong", "true", "false", "real", "fake",
        "start", "stop", "end", "begin", "finish", "continue",
        "open", "close", "turn", "move", "change", "keep",
        "let", "leave", "live", "die", "play", "pay", "buy", "sell",
        "read", "write", "learn", "teach", "show", "send", "receive",
        "hold", "bring", "carry", "pull", "push", "pick", "drop",
        "eat", "drink", "sleep", "sit", "stand", "walk", "talk",
        "example", "fact", "course", "sure", "wonder", "mind",
        "already", "still", "yet", "ever", "even", "else",
        "between", "among", "against", "along", "across", "around",
        "above", "below", "behind", "beside", "near", "far",
        "however", "therefore", "moreover", "furthermore",
        "perhaps", "maybe", "probably", "definitely", "certainly",
        "paper", "research", "study", "data", "model", "system",
        "plan", "team", "company", "office", "school", "class",
        # 补充常见 CS 固定表达中的词
        "by", "the", "way", "no", "problem", "long", "never",
    }

    # 乱码/HTML 检测模式
    _GARBAGE_PATTERNS = [
        r'<[a-zA-Z/][^>]*>',       # HTML 标签
        r'&[a-z]+;',                # HTML 实体
        r'\\x[0-9a-fA-F]{2}',      # 转义字节
        r'[\x00-\x08\x0b\x0c\x0e-\x1f]',  # 控制字符
    ]

    def check(self, analysis: TextAnalysis) -> CheckResult:
        """执行流畅度检查"""
        violations = []
        score = 10.0

        # ===== 硬性规则检查 =====
        # 1. 必须包含两种语言（重扣分但不直接否决，给重试机会选更好的）
        if not analysis.zh_tokens or not analysis.en_tokens:
            return CheckResult(
                checker_name="fluency", score=1.0,
                violations=["文本不包含两种语言"],
                is_veto=False,
            )

        # 2. token 数范围
        total = analysis.total_content_tokens
        if total < 5:
            return CheckResult(
                checker_name="fluency", score=0.0,
                violations=[f"token 数过少 ({total} < 5)"],
                is_veto=True,
            )
        if total > 500:
            return CheckResult(
                checker_name="fluency", score=0.0,
                violations=[f"token 数过多 ({total} > 500)"],
                is_veto=True,
            )

        # 3. 乱码/HTML 检查
        for pattern in self._GARBAGE_PATTERNS:
            if re.search(pattern, analysis.raw_text):
                return CheckResult(
                    checker_name="fluency", score=0.0,
                    violations=["检测到乱码/HTML残留"],
                    is_veto=True,
                )

        # 4. LLM 思考过程残留检查（Qwen3.5 thinking mode 泄漏）
        thinking_markers = ["**Analyze", "**Role:**", "**Task:**",
                           "**Constraint", "**Drafting", "Thinking Process"]
        if any(m in analysis.raw_text for m in thinking_markers):
            return CheckResult(
                checker_name="fluency", score=0.0,
                violations=["检测到 LLM 思考过程残留（非对话内容）"],
                is_veto=True,
            )

        details = {}

        # ===== L1 (中文) 片段质量 (权重 0.4) =====
        l1_score = 10.0
        # 检查中文片段是否过短（单字碎片）
        zh_segments = [s for s in analysis.segments if s["lang"] == "zh"]
        single_char_segments = sum(
            1 for s in zh_segments
            if sum(1 for t in s["tokens"] if t.lang == "zh") == 1
        )
        if zh_segments and single_char_segments / len(zh_segments) > 0.5:
            l1_score -= 2.0
            violations.append(f"过多单字中文碎片 ({single_char_segments}/{len(zh_segments)})")
        details["l1_quality"] = max(0, l1_score)

        # ===== L2 (英文) 片段质量 (权重 0.4) =====
        l2_score = 10.0
        # 检查英文单词是否在常见词表中（宽松检查：容忍未知词）
        unknown_words = []
        for token in analysis.en_tokens:
            if token.text.lower() not in self._COMMON_EN_WORDS and len(token.text) > 1:
                unknown_words.append(token.text)
        # 未知词比例过高才扣分（>50% 的英文词都不认识 → 可能是乱码）
        if analysis.en_tokens:
            unknown_ratio = len(unknown_words) / len(analysis.en_tokens)
            if unknown_ratio > 0.7:
                l2_score -= 3.0
                violations.append(f"英文词汇异常率过高 ({unknown_ratio:.0%})")
            elif unknown_ratio > 0.5:
                l2_score -= 1.5
        details["l2_quality"] = max(0, l2_score)

        # ===== 整体完整性 (权重 0.2) =====
        completeness_score = 10.0
        text = analysis.raw_text

        # 括号配对检查（口语中括号不配对很常见，轻微扣分）
        for open_ch, close_ch in [("（", "）"), ("「", "」"), ("【", "】")]:
            if text.count(open_ch) != text.count(close_ch):
                completeness_score -= 0.5
                violations.append(f"括号未配对: {open_ch}...{close_ch}")
                break

        # 句末完整性：最后一个字符应为标点或自然结尾
        text_stripped = text.rstrip()
        if text_stripped:
            last_char = text_stripped[-1]
            natural_endings = set("。！？.!?~…了的呢吧啊吗")
            # 也允许英文字母或中文字符结尾（口语化表达）
            if (last_char not in natural_endings
                    and not last_char.isalpha()
                    and self._classify_char_simple(last_char) != "zh"):
                completeness_score -= 2.0
                violations.append("文本可能截断（非自然结尾）")
        details["completeness"] = max(0, completeness_score)

        # 加权计算总分
        score = (
            details["l1_quality"] * 0.4
            + details["l2_quality"] * 0.4
            + details["completeness"] * 0.2
        )
        score = max(0.0, min(10.0, score))

        return CheckResult(
            checker_name="fluency",
            score=round(score, 2),
            details=details,
            violations=violations,
        )

    @staticmethod
    def _classify_char_simple(ch: str) -> str:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF:
            return "zh"
        if ch.isascii() and ch.isalpha():
            return "en"
        return "other"


# ============================================================
# Checker 2: NaturalnessChecker — CS 自然度
# ============================================================

class NaturalnessChecker:
    """
    基于语言学理论检查 CS 模式的自然度。

    理论基础：
    - Poplack (1980): Equivalence Constraint — 切换点前后应兼容两种语言的句法
    - Poplack (1980): Free Morpheme Constraint — 不在绑定语素边界切换
    - Myers-Scotton (1993): MLF — 基质语言提供句法框架
    - Grosjean (2001): Language Mode — 基质语言应可辨识

    评分项：
    - 切换频率合理性：不过度乒乓，也不过少
    - L2 片段完整性：嵌入的 L2 应是完整短语单位
    - 语言模式连续性：应有可辨识的基质语言
    """

    def check(self, analysis: TextAnalysis) -> CheckResult:
        """执行自然度检查"""
        violations = []
        sub_scores = {}

        # ===== 1. 切换频率合理性 (权重 0.30) =====
        freq_score = 10.0

        if analysis.total_content_tokens > 0:
            # 检查是否过度切换（乒乓效应）
            # 计算：平均每个 segment 包含多少 content token
            content_segments = [
                s for s in analysis.segments if s["lang"] in ("zh", "en")
            ]
            if content_segments:
                avg_segment_len = analysis.total_content_tokens / len(content_segments)
                if avg_segment_len < 1.5:
                    # 平均每段不到 1.5 个词 → 严重乒乓
                    freq_score -= 4.0
                    violations.append(
                        f"过度切换：平均每段仅 {avg_segment_len:.1f} 个词（乒乓效应）"
                    )
                elif avg_segment_len < 2.5:
                    freq_score -= 2.0
                    violations.append(
                        f"切换频率偏高：平均每段 {avg_segment_len:.1f} 个词"
                    )

            # 检查是否过少切换（声称是 CS 但几乎无切换）
            if analysis.cmi < 0.02:
                freq_score -= 4.0
                violations.append(f"CMI 过低 ({analysis.cmi:.3f})，几乎无切换")
            elif analysis.cmi < 0.05 and len(analysis.switch_points) <= 1:
                freq_score -= 2.0
                violations.append("切换次数过少")

        sub_scores["switch_frequency"] = max(0, freq_score)

        # ===== 2. L2 片段完整性 (权重 0.25) =====
        l2_integrity_score = 10.0

        en_segments = [s for s in analysis.segments if s["lang"] == "en"]
        if en_segments:
            # 检查是否有过多单词碎片（非完整短语）
            # 单个英文词的段不一定有问题（术语嵌入是正常的）
            # 但如果大量段都是单个词，且不是已知术语/固定表达 → 可能不自然
            very_short = sum(
                1 for s in en_segments
                if sum(1 for t in s["tokens"] if t.lang == "en") == 1
            )
            if len(en_segments) > 2 and very_short / len(en_segments) > 0.8:
                l2_integrity_score -= 2.0
                violations.append("英文片段过于碎片化，缺少完整短语")

        sub_scores["l2_integrity"] = max(0, l2_integrity_score)

        # ===== 3. 语言模式连续性 — Grosjean (权重 0.25) =====
        mode_score = 10.0

        # 基质语言应占主导（至少 55% 的 content token）
        if analysis.total_content_tokens > 0:
            dominant_ratio = max(
                len(analysis.zh_tokens), len(analysis.en_tokens)
            ) / analysis.total_content_tokens
            if dominant_ratio < 0.55:
                # 两种语言几乎完全对等 → 没有可辨识的基质语言
                # 这种情况在真实 CS 中存在但较少见，轻微扣分
                mode_score -= 1.5
                violations.append(
                    f"基质语言不明确：主导语言仅占 {dominant_ratio:.0%}"
                )

        sub_scores["language_mode"] = max(0, mode_score)

        # ===== 4. 切换点句法合理性 (权重 0.20) =====
        syntax_score = 10.0

        # 检查词素内部切换（如"跑running"、"美beautiful"）
        # 简化检测：中文字符紧邻英文字符且无空格/标点分隔
        raw = analysis.raw_text
        # 匹配 "汉字英文" 或 "英文汉字" 且中间无空格
        morpheme_violations = len(re.findall(
            r'[\u4e00-\u9fff][a-zA-Z]|[a-zA-Z][\u4e00-\u9fff]', raw
        ))
        if morpheme_violations > 3:
            syntax_score -= 3.0
            violations.append(f"疑似词素内部切换 ({morpheme_violations} 处)")
        elif morpheme_violations > 0:
            # 少量可能是排版问题，轻微扣分
            syntax_score -= 0.5 * morpheme_violations

        sub_scores["switch_syntax"] = max(0, syntax_score)

        # 加权汇总
        score = (
            sub_scores["switch_frequency"] * 0.30
            + sub_scores["l2_integrity"] * 0.25
            + sub_scores["language_mode"] * 0.25
            + sub_scores["switch_syntax"] * 0.20
        )
        score = max(0.0, min(10.0, score))

        return CheckResult(
            checker_name="naturalness",
            score=round(score, 2),
            details=sub_scores,
            violations=violations,
        )


# ============================================================
# Checker 3: SwitchMotivationChecker — 切换动机
# ============================================================

class SwitchMotivationChecker:
    """
    检查每次语言切换是否有可识别的语用/词汇动机。

    理论基础：
    - Gumperz (1982) 的 6 种 CS 会话功能：
      quotation, addressee_specification, interjection,
      reiteration, message_qualification, personalization
    - 词汇层面动机：领域术语、固定表达、话语标记

    评分逻辑：
    - 对每个切换点，尝试归因到已知动机类型
    - 可归因切换占比 ≥80% → 9-10 分
    - ≥60% → 7-8 分
    - ≥40% → 5-6 分
    - <40% → 低于 5 分
    """

    # 英文固定表达（在 CS 中作为不可拆分单元出现）
    FIXED_EXPRESSIONS = {
        "by the way", "make sense", "no problem", "you know", "i mean",
        "come on", "take care", "oh my god", "for example", "in fact",
        "of course", "kind of", "sort of", "as well", "to be honest",
        "no wonder", "never mind", "on the other hand", "at the end",
        "long time no see", "no choice", "last time", "next time",
        "so far", "by the way", "in case", "what the",
    }

    # 英文话语标记（常作为 interjection 功能出现）
    DISCOURSE_MARKERS = {
        "like", "well", "so", "actually", "basically", "literally",
        "honestly", "anyway", "right", "okay", "ok", "sure", "exactly",
        "obviously", "seriously", "apparently",
    }

    # 常见 CS 领域术语模式（正则）
    DOMAIN_TERM_PATTERNS = [
        # 技术类
        r'\b(?:debug|deploy|commit|push|pull|merge|code|api|database|'
        r'server|frontend|backend|stack|framework|algorithm|function|'
        r'variable|interface|module|package|library|docker|cloud)\b',
        # 商务类
        r'\b(?:meeting|deadline|project|email|schedule|report|budget|'
        r'client|feedback|presentation|proposal|strategy|kpi|roi)\b',
        # 学术类
        r'\b(?:paper|research|study|thesis|professor|lecture|seminar|'
        r'assignment|course|exam|grade|campus|lab|data|model)\b',
    ]

    # 感叹词/语气词
    INTERJECTIONS_EN = {"oh", "wow", "oops", "hmm", "huh", "ah", "ugh", "yay", "omg"}
    INTERJECTIONS_ZH = {"哎", "唉", "哇", "嗯", "啊", "呀", "哦", "嘿", "诶"}

    def __init__(self):
        # 编译领域术语正则
        self._domain_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.DOMAIN_TERM_PATTERNS
        ]

    def _is_domain_term(self, word: str) -> bool:
        """检查一个英文词是否为领域术语"""
        for pattern in self._domain_patterns:
            if pattern.search(word):
                return True
        return False

    def _is_fixed_expression(self, text_lower: str, position: int) -> bool:
        """检查切换点附近是否包含固定表达"""
        for expr in self.FIXED_EXPRESSIONS:
            if expr in text_lower:
                return True
        return False

    def _is_discourse_marker(self, tokens: list[TokenInfo], switch_pos: int) -> bool:
        """检查切换点的 token 是否为话语标记"""
        for token in tokens:
            if (token.lang == "en"
                    and abs(token.position - switch_pos) <= 1
                    and token.text.lower() in self.DISCOURSE_MARKERS):
                return True
        return False

    def _is_interjection(self, tokens: list[TokenInfo], switch_pos: int) -> bool:
        """检查切换点附近是否有感叹词"""
        for token in tokens:
            if abs(token.position - switch_pos) <= 1:
                if token.lang == "en" and token.text.lower() in self.INTERJECTIONS_EN:
                    return True
                if token.lang == "zh" and token.text in self.INTERJECTIONS_ZH:
                    return True
        return False

    def check(self, analysis: TextAnalysis) -> CheckResult:
        """执行切换动机检查"""
        if not analysis.switch_points:
            # 无切换点 → 不适用（交给 NaturalnessChecker 处理）
            return CheckResult(
                checker_name="switch_motivation",
                score=8.0,
                details={"note": "无切换点，跳过动机检查"},
            )

        text_lower = analysis.raw_text.lower()
        attributed = 0
        total = len(analysis.switch_points)
        motivation_counts = Counter()

        for sp in analysis.switch_points:
            motivation = self._attribute_motivation(
                sp, analysis.tokens, text_lower
            )
            if motivation:
                attributed += 1
                sp.motivation = motivation
                motivation_counts[motivation] += 1

        # 计算可归因比例
        ratio = attributed / total if total > 0 else 0

        # 映射到分数（放宽阈值：LLM 生成的 CS 很多切换是自然的但难以规则归因）
        if ratio >= 0.60:
            score = 9.0 + ratio  # 9.0 - 9.6
        elif ratio >= 0.40:
            score = 7.0 + (ratio - 0.40) * 10  # 7.0 - 9.0
        elif ratio >= 0.20:
            score = 5.0 + (ratio - 0.20) * 10  # 5.0 - 7.0
        else:
            score = 4.0 + ratio * 5  # 4.0 - 5.0

        score = max(0.0, min(10.0, score))

        violations = []
        if ratio < 0.20:
            violations.append(
                f"仅 {ratio:.0%} 的切换有可识别动机（阈值 20%）"
            )

        return CheckResult(
            checker_name="switch_motivation",
            score=round(score, 2),
            details={
                "attribution_ratio": round(ratio, 3),
                "total_switches": total,
                "attributed_switches": attributed,
                "motivation_distribution": dict(motivation_counts),
            },
            violations=violations,
        )

    def _attribute_motivation(
        self, sp: SwitchPoint, tokens: list[TokenInfo], text_lower: str
    ) -> str:
        """
        尝试为一个切换点归因动机类型。
        按优先级依次检查，返回第一个匹配的动机。
        """
        # 1. 感叹词/语气词 → interjection (Gumperz)
        if self._is_interjection(tokens, sp.position):
            return "interjection"

        # 2. 话语标记 → discourse_marker
        if self._is_discourse_marker(tokens, sp.position):
            return "discourse_marker"

        # 3. 固定表达 → fixed_expression
        if self._is_fixed_expression(text_lower, sp.position):
            return "fixed_expression"

        # 4. 领域术语 → domain_term
        # 检查切换后的英文词是否为领域术语
        for token in tokens:
            if (token.position >= sp.position
                    and token.position <= sp.position + 2
                    and token.lang == "en"
                    and self._is_domain_term(token.text)):
                return "domain_term"

        # 5. 句间切换 → 通常有话题转换的隐含动机，宽松归因
        if sp.switch_type == "inter":
            return "topic_shift"

        # 6. 未能归因
        return ""


# ============================================================
# Checker 4: ProfileConsistencyChecker — 画像一致性
# ============================================================

class ProfileConsistencyChecker:
    """
    检查生成的 CS 文本是否符合输入的 Archetype + Persona 画像参数。

    评分项：
    - CMI 是否在 archetype 指定的 CMI_range 内
    - 切换类型比例是否符合 archetype 特征
    - L2 片段长度是否符合 archetype 预期
    - 是否使用了与话题相关的领域词
    """

    # 各原型对切换类型比例的期望
    _SWITCH_TYPE_EXPECTATIONS = {
        "ARC_01": {"intra_min": 0.70, "desc": "Insertional: 应以句内嵌入为主"},
        "ARC_02": {"inter_min": 0.60, "desc": "Alternational: 应以句间交替为主"},
        "ARC_03": {"intra_min": 0.50, "min_switch_density": 0.15,
                    "desc": "Dense: 高频句内切换"},
        "ARC_04": {"desc": "Pragmatic: 灵活，无固定约束"},
        "ARC_05": {"max_cmi": 0.15, "desc": "Reluctant: CMI 应很低"},
        "ARC_06": {"desc": "Accommodation: 适应型，无固定约束"},
    }

    # 各原型对 L2 片段长度的期望
    _L2_SPAN_EXPECTATIONS = {
        "ARC_01": {"typical_max": 3, "desc": "通常 1-3 词"},
        "ARC_02": {"typical_min": 4, "desc": "通常整句/整从句"},
        "ARC_03": {"typical_max": 5, "desc": "1-5 词频繁"},
    }

    def check(
        self,
        analysis: TextAnalysis,
        archetype_id: str = "",
        archetype_cmi_range: list[float] = None,
        expected_topic: str = "",
        domain_words: list[str] = None,
    ) -> CheckResult:
        """
        执行画像一致性检查。

        参数：
        - analysis: 文本分析结果
        - archetype_id: 原型 ID（如 "ARC_01"）
        - archetype_cmi_range: 原型的 CMI 范围（如 [0.05, 0.25]）
        - expected_topic: 期望的话题
        - domain_words: 期望出现的领域词列表
        """
        violations = []
        sub_scores = {}

        # ===== 1. CMI 范围一致性 (权重 0.35) =====
        cmi_score = 10.0
        if archetype_cmi_range and len(archetype_cmi_range) == 2:
            cmi_low, cmi_high = archetype_cmi_range
            actual_cmi = analysis.cmi

            if cmi_low <= actual_cmi <= cmi_high:
                cmi_score = 10.0  # 完美匹配
            else:
                # 计算偏差
                if actual_cmi < cmi_low:
                    deviation = cmi_low - actual_cmi
                else:
                    deviation = actual_cmi - cmi_high

                if deviation <= 0.05:
                    cmi_score = 8.0  # 轻微偏差
                elif deviation <= 0.10:
                    cmi_score = 6.0
                    violations.append(
                        f"CMI 偏差: 实际 {actual_cmi:.3f}, "
                        f"期望 [{cmi_low:.2f}, {cmi_high:.2f}]"
                    )
                else:
                    cmi_score = 3.0
                    violations.append(
                        f"CMI 严重偏差: 实际 {actual_cmi:.3f}, "
                        f"期望 [{cmi_low:.2f}, {cmi_high:.2f}]"
                    )
        sub_scores["cmi_range"] = cmi_score

        # ===== 2. 切换类型一致性 (权重 0.30) =====
        type_score = 10.0
        if archetype_id and archetype_id in self._SWITCH_TYPE_EXPECTATIONS:
            exp = self._SWITCH_TYPE_EXPECTATIONS[archetype_id]

            if analysis.switch_points:
                intra_count = sum(
                    1 for sp in analysis.switch_points if sp.switch_type == "intra"
                )
                inter_count = sum(
                    1 for sp in analysis.switch_points if sp.switch_type == "inter"
                )
                total_sp = len(analysis.switch_points)
                intra_ratio = intra_count / total_sp
                inter_ratio = inter_count / total_sp

                # 检查 intra_min 约束
                if "intra_min" in exp and intra_ratio < exp["intra_min"]:
                    gap = exp["intra_min"] - intra_ratio
                    type_score -= min(5.0, gap * 15)
                    violations.append(
                        f"句内切换比例不足: {intra_ratio:.0%} < {exp['intra_min']:.0%} "
                        f"({exp['desc']})"
                    )

                # 检查 inter_min 约束
                if "inter_min" in exp and inter_ratio < exp["inter_min"]:
                    gap = exp["inter_min"] - inter_ratio
                    type_score -= min(5.0, gap * 15)
                    violations.append(
                        f"句间切换比例不足: {inter_ratio:.0%} < {exp['inter_min']:.0%} "
                        f"({exp['desc']})"
                    )

                # 检查 max_cmi 约束（ARC_05 不情愿型）
                if "max_cmi" in exp and analysis.cmi > exp["max_cmi"]:
                    type_score -= 4.0
                    violations.append(
                        f"不情愿型但 CMI 过高: {analysis.cmi:.3f} > {exp['max_cmi']}"
                    )

        sub_scores["switch_type"] = max(0, type_score)

        # ===== 3. L2 片段长度一致性 (权重 0.20) =====
        span_score = 10.0
        if archetype_id and archetype_id in self._L2_SPAN_EXPECTATIONS:
            exp = self._L2_SPAN_EXPECTATIONS[archetype_id]
            en_segments = [s for s in analysis.segments if s["lang"] == "en"]

            if en_segments:
                en_spans = [
                    sum(1 for t in s["tokens"] if t.lang == "en")
                    for s in en_segments
                ]
                avg_span = sum(en_spans) / len(en_spans)

                if "typical_max" in exp and avg_span > exp["typical_max"] * 1.5:
                    span_score -= 3.0
                    violations.append(
                        f"L2 片段过长: 平均 {avg_span:.1f} 词, "
                        f"期望 ≤{exp['typical_max']} ({exp['desc']})"
                    )
                if "typical_min" in exp and avg_span < exp["typical_min"] * 0.5:
                    span_score -= 3.0
                    violations.append(
                        f"L2 片段过短: 平均 {avg_span:.1f} 词, "
                        f"期望 ≥{exp['typical_min']} ({exp['desc']})"
                    )

        sub_scores["l2_span_length"] = max(0, span_score)

        # ===== 4. 领域词使用 (权重 0.15) =====
        domain_score = 10.0
        if domain_words:
            text_lower = analysis.raw_text.lower()
            found = sum(1 for w in domain_words if w.lower() in text_lower)
            if found == 0:
                domain_score = 4.0
                violations.append("未使用任何期望的领域词")
            elif found < 2:
                domain_score = 7.0
        sub_scores["domain_words"] = domain_score

        # 加权汇总
        score = (
            sub_scores["cmi_range"] * 0.35
            + sub_scores["switch_type"] * 0.30
            + sub_scores["l2_span_length"] * 0.20
            + sub_scores["domain_words"] * 0.15
        )
        score = max(0.0, min(10.0, score))

        return CheckResult(
            checker_name="profile_consistency",
            score=round(score, 2),
            details=sub_scores,
            violations=violations,
        )


# ============================================================
# Checker 5: CulturalCoherenceChecker — 文化语域一致性
# ============================================================

class CulturalCoherenceChecker:
    """
    检查 CS 文本的文化适当性和语域一致性。

    理论基础：
    - Myers-Scotton 标记性模型：CS 选择反映社会关系
    - 正式度与语言选择的关联

    评分项：
    - 正式度一致性：正式场合不应有过多俚语
    - 地域表达合理性：语气词与地区匹配
    - 敏感内容过滤
    """

    # 各地区的特征语气词
    REGION_MARKERS = {
        "singapore": {"lah", "lor", "leh", "sia", "hor", "meh", "ar"},
        "malaysia": {"lah", "lor", "ma", "bah"},
        "hongkong": {"啦", "嘅", "嗰", "喺", "咩", "嘢"},
    }

    # 非正式/俚语标记（仅 formal 场合扣分，casual/semi_formal 不扣）
    INFORMAL_MARKERS = {
        "666", "yyds", "绝绝子", "awsl", "xswl", "srds",
        "tmd", "草", "卧槽", "靠",
    }
    # 仅在 formal 场合扣分的英文俚语（casual 中完全正常）
    FORMAL_ONLY_MARKERS = {
        "wtf", "lmao", "lol", "bruh", "ngl", "omg",
    }

    def check(
        self,
        analysis: TextAnalysis,
        formality: str = "casual",
        region: str = "",
    ) -> CheckResult:
        """
        执行文化语域一致性检查。

        参数：
        - analysis: 文本分析结果
        - formality: 正式度 ("casual" / "semi_formal" / "formal")
        - region: 地区标识 ("singapore" / "malaysia" / "hongkong" / "mainland" 等)
        """
        violations = []
        sub_scores = {}
        text_lower = analysis.raw_text.lower()

        # ===== 1. 正式度一致性 (权重 0.50) =====
        formality_score = 10.0

        if formality == "formal":
            # formal 场合：检查所有非正式标记
            found_informal = [
                m for m in (self.INFORMAL_MARKERS | self.FORMAL_ONLY_MARKERS)
                if m in text_lower
            ]
            if found_informal:
                penalty = min(4.0, len(found_informal) * 1.5)
                formality_score -= penalty
        elif formality == "semi_formal":
            # semi_formal 场合：仅检查粗话类
            found_informal = [
                m for m in self.INFORMAL_MARKERS if m in text_lower
            ]
            if found_informal:
                penalty = min(3.0, len(found_informal) * 1.0)
                formality_score -= penalty
                violations.append(
                    f"正式场合使用了非正式表达: {', '.join(found_informal[:3])}"
                )

        sub_scores["formality"] = max(0, formality_score)

        # ===== 2. 地域表达合理性 (权重 0.30) =====
        region_score = 10.0

        if region:
            # 检查是否使用了其他地区的特征词
            all_words = {t.text.lower() for t in analysis.tokens}

            for other_region, markers in self.REGION_MARKERS.items():
                if other_region == region:
                    continue  # 跳过自己的地区
                found_foreign = markers & all_words
                if found_foreign:
                    region_score -= min(3.0, len(found_foreign) * 1.5)
                    violations.append(
                        f"{region} 说话人使用了 {other_region} 特征词: "
                        f"{', '.join(found_foreign)}"
                    )

        sub_scores["regional"] = max(0, region_score)

        # ===== 3. 敏感内容过滤 (权重 0.20) =====
        sensitivity_score = 10.0
        # 基础的敏感词检测（生产中应使用更完整的敏感词库）
        # 此处仅做框架示意
        sub_scores["sensitivity"] = sensitivity_score

        # 加权汇总
        score = (
            sub_scores["formality"] * 0.50
            + sub_scores["regional"] * 0.30
            + sub_scores["sensitivity"] * 0.20
        )
        score = max(0.0, min(10.0, score))

        return CheckResult(
            checker_name="cultural_coherence",
            score=round(score, 2),
            details=sub_scores,
            violations=violations,
        )


# ============================================================
# Checker 6: DiversityChecker — 多样性
# ============================================================

class DiversityChecker:
    """
    检查新样本与已接受语料的差异度，避免数据同质化。

    基于 DEITA (2024) 的多样性原则：
    - N-gram 新颖度：与已有数据的 n-gram 重叠越少越好
    - 切换模式多样性：切换点位置模式不应千篇一律

    使用方式：
    - 维护一个已接受样本的 n-gram 集合（增量更新）
    - 每条新样本计算与已有数据的重叠率
    """

    def __init__(self, ngram_n: int = 4):
        self.ngram_n = ngram_n
        # 已接受样本的 n-gram 集合（持续积累）
        self._accepted_ngrams: set[str] = set()
        # 已接受样本的切换模式集合
        self._accepted_switch_patterns: Counter = Counter()
        # 已接受样本的 MinHash 签名集合（用于去重）
        self._accepted_hashes: set[str] = set()
        # 已接受样本数
        self._accepted_count: int = 0

    def _extract_ngrams(self, tokens: list[TokenInfo]) -> set[str]:
        """提取 content token 的 n-gram 集合"""
        content = [
            t.text.lower() for t in tokens if t.lang in ("zh", "en")
        ]
        if len(content) < self.ngram_n:
            return set()
        ngrams = set()
        for i in range(len(content) - self.ngram_n + 1):
            ngram = "|".join(content[i:i + self.ngram_n])
            ngrams.add(ngram)
        return ngrams

    def _extract_switch_pattern(self, switch_points: list[SwitchPoint]) -> str:
        """将切换点序列编码为模式字符串"""
        if not switch_points:
            return "no_switch"
        pattern_parts = []
        for sp in switch_points:
            pattern_parts.append(f"{sp.from_lang}>{sp.to_lang}:{sp.switch_type}")
        return "|".join(pattern_parts)

    def _compute_text_hash(self, text: str) -> str:
        """计算文本的 hash（用于精确去重）"""
        # 归一化：去空白、转小写
        normalized = re.sub(r'\s+', '', text.lower())
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()

    def check(self, analysis: TextAnalysis) -> CheckResult:
        """
        执行多样性检查。

        注意：此 Checker 是有状态的——它维护已接受样本的信息。
        首批样本（_accepted_count == 0 时）自动获得满分。
        """
        violations = []
        sub_scores = {}

        # ===== 0. 精确去重 =====
        text_hash = self._compute_text_hash(analysis.raw_text)
        if text_hash in self._accepted_hashes:
            return CheckResult(
                checker_name="diversity",
                score=0.0,
                violations=["与已有数据完全重复"],
                is_veto=True,
            )

        # 首批样本：无参照基准，给满分
        if self._accepted_count == 0:
            return CheckResult(
                checker_name="diversity",
                score=10.0,
                details={"note": "首批样本，无参照基准"},
            )

        # ===== 1. N-gram 新颖度 (权重 0.50) =====
        ngram_score = 10.0
        sample_ngrams = self._extract_ngrams(analysis.tokens)
        if sample_ngrams:
            overlap = sample_ngrams & self._accepted_ngrams
            overlap_ratio = len(overlap) / len(sample_ngrams)

            if overlap_ratio > 0.70:
                ngram_score = 2.0
                violations.append(f"N-gram 重叠率过高: {overlap_ratio:.0%}")
            elif overlap_ratio > 0.50:
                ngram_score = 5.0
                violations.append(f"N-gram 重叠率偏高: {overlap_ratio:.0%}")
            elif overlap_ratio > 0.30:
                ngram_score = 7.5
            else:
                ngram_score = 10.0
        sub_scores["ngram_novelty"] = ngram_score

        # ===== 2. 切换模式多样性 (权重 0.50) =====
        pattern_score = 10.0
        pattern = self._extract_switch_pattern(analysis.switch_points)
        pattern_count = self._accepted_switch_patterns.get(pattern, 0)

        if self._accepted_count > 100:
            # 只有积累足够样本后才开始检查模式重复
            pattern_freq = pattern_count / self._accepted_count
            if pattern_freq > 0.10:
                pattern_score = 5.0
                violations.append(
                    f"切换模式过于集中: '{pattern}' 已占 {pattern_freq:.0%}"
                )
            elif pattern_freq > 0.05:
                pattern_score = 7.5
        sub_scores["switch_pattern_novelty"] = pattern_score

        # 加权汇总
        score = (
            sub_scores["ngram_novelty"] * 0.50
            + sub_scores["switch_pattern_novelty"] * 0.50
        )
        score = max(0.0, min(10.0, score))

        return CheckResult(
            checker_name="diversity",
            score=round(score, 2),
            details=sub_scores,
            violations=violations,
        )

    def accept_sample(self, analysis: TextAnalysis):
        """
        将一条通过评估的样本加入已接受集合。
        后续样本将与此样本比较多样性。
        """
        # 更新 n-gram 集合
        ngrams = self._extract_ngrams(analysis.tokens)
        self._accepted_ngrams.update(ngrams)

        # 更新切换模式计数
        pattern = self._extract_switch_pattern(analysis.switch_points)
        self._accepted_switch_patterns[pattern] += 1

        # 更新 hash 集合
        text_hash = self._compute_text_hash(analysis.raw_text)
        self._accepted_hashes.add(text_hash)

        self._accepted_count += 1


# ============================================================
# 主 Pipeline: RuleBasedEvaluatorPipeline
# ============================================================

class RuleBasedEvaluatorPipeline:
    """
    规则化多维度 CS 质量评估管道。

    使用流程：
    1. 初始化 pipeline（加载宪法配置）
    2. 对每条生成的 CS 文本调用 evaluate()
    3. 根据返回的 EvaluationResult.decision 做 pass/fail/review 决策
    4. 通过评估的样本调用 accept() 更新多样性基准

    示例：
        pipeline = RuleBasedEvaluatorPipeline()
        result = pipeline.evaluate(
            text="今天的 meeting 我觉得还 okay，就是那个 deadline 有点赶。",
            archetype_id="ARC_01",
            archetype_cmi_range=[0.05, 0.25],
            formality="casual",
            region="singapore",
            domain_words=["meeting", "deadline", "project"],
        )
        if result.decision == "pass":
            pipeline.accept(result)
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化评估管道。

        参数：
        - config_path: evaluation_constitutions.yaml 的路径。
          若为 None，使用与本文件同目录下的默认配置。
        """
        if config_path is None:
            config_path = Path(__file__).parent / "evaluation_constitutions.yaml"
        else:
            config_path = Path(config_path)

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # 初始化各组件
        self.analyzer = TextAnalyzer()
        self.fluency_checker = FluencyChecker()
        self.naturalness_checker = NaturalnessChecker()
        self.motivation_checker = SwitchMotivationChecker()
        self.profile_checker = ProfileConsistencyChecker()
        self.cultural_checker = CulturalCoherenceChecker()
        self.diversity_checker = DiversityChecker(
            ngram_n=self.config.get("diversity", {})
            .get("scoring", {})
            .get("ngram_novelty", {})
            .get("ngram_n", 4)
        )

        # 全局配置
        global_cfg = self.config.get("global", {})
        self.pass_threshold = global_cfg.get("pass_threshold", 6.0)
        self.review_range = global_cfg.get("human_review_range", [5.0, 6.0])
        self.veto_threshold = global_cfg.get("veto_threshold", 2.0)
        self.weights = global_cfg.get("weights", {
            "fluency": 0.20,
            "naturalness": 0.25,
            "switch_motivation": 0.20,
            "profile_consistency": 0.15,
            "cultural_coherence": 0.10,
            "diversity": 0.10,
        })

    def evaluate(
        self,
        text: str,
        archetype_id: str = "",
        archetype_cmi_range: list[float] = None,
        formality: str = "casual",
        region: str = "",
        expected_topic: str = "",
        domain_words: list[str] = None,
    ) -> EvaluationResult:
        """
        对一条 CS 文本执行全维度规则评估。

        参数：
        - text: 待评估的 CS 文本
        - archetype_id: 生成时使用的原型 ID（如 "ARC_01"）
        - archetype_cmi_range: 原型的 CMI 范围
        - formality: 场景正式度
        - region: 说话人地区
        - expected_topic: 期望话题
        - domain_words: 期望的领域词列表

        返回：
        - EvaluationResult，包含最终分数、决策和各维度详情
        """
        # Step 1: 文本分析
        analysis = self.analyzer.analyze(text)

        # Step 2: 运行 6 个 Checker
        results = {}

        results["fluency"] = self.fluency_checker.check(analysis)

        results["naturalness"] = self.naturalness_checker.check(analysis)

        results["switch_motivation"] = self.motivation_checker.check(analysis)

        results["profile_consistency"] = self.profile_checker.check(
            analysis,
            archetype_id=archetype_id,
            archetype_cmi_range=archetype_cmi_range,
            expected_topic=expected_topic,
            domain_words=domain_words,
        )

        results["cultural_coherence"] = self.cultural_checker.check(
            analysis,
            formality=formality,
            region=region,
        )

        results["diversity"] = self.diversity_checker.check(analysis)

        # Step 3: 一票否决检查
        veto_reason = ""
        for name, cr in results.items():
            if cr.is_veto:
                veto_reason = f"{name}: {'; '.join(cr.violations)}"
                break
            if cr.score < self.veto_threshold:
                veto_reason = (
                    f"{name} 分数过低 ({cr.score:.1f} < {self.veto_threshold})"
                )
                break

        if veto_reason:
            return EvaluationResult(
                text=text,
                final_score=0.0,
                decision="fail",
                checker_results=results,
                analysis=analysis,
                veto_reason=veto_reason,
            )

        # Step 4: 加权汇总
        final_score = sum(
            results[name].score * self.weights.get(name, 0)
            for name in results
        )
        final_score = round(final_score, 2)

        # Step 5: 决策
        if final_score >= self.pass_threshold:
            decision = "pass"
        elif final_score >= self.review_range[0]:
            decision = "review"
        else:
            decision = "fail"

        return EvaluationResult(
            text=text,
            final_score=final_score,
            decision=decision,
            checker_results=results,
            analysis=analysis,
        )

    def accept(self, result: EvaluationResult):
        """将通过评估的样本加入多样性基准"""
        self.diversity_checker.accept_sample(result.analysis)

    def get_summary(self, result: EvaluationResult) -> str:
        """生成人类可读的评估摘要"""
        lines = [
            f"=== 评估结果: {result.decision.upper()} "
            f"(总分: {result.final_score:.1f}/10.0) ===",
            f"文本: {result.text[:80]}{'...' if len(result.text) > 80 else ''}",
            f"CMI: {result.analysis.cmi:.3f} | "
            f"基质语言: {result.analysis.matrix_language} | "
            f"切换点: {len(result.analysis.switch_points)} 个",
            "",
        ]

        if result.veto_reason:
            lines.append(f"[VETO] 一票否决: {result.veto_reason}")
            lines.append("")

        for name, cr in result.checker_results.items():
            weight = self.weights.get(name, 0)
            weighted = cr.score * weight
            lines.append(
                f"  [{name:25s}] "
                f"{cr.score:5.1f}/10 × {weight:.2f} = {weighted:.2f}"
            )
            if cr.violations:
                for v in cr.violations:
                    lines.append(f"    -> {v}")

        lines.append("")
        return "\n".join(lines)


# ============================================================
# CLI 入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SwitchLingua 2.0 Rule-Based CS Quality Evaluator"
    )
    parser.add_argument(
        "--text", type=str,
        default="今天的meeting我觉得还okay，就是那个deadline有点赶，"
                "我们要不要reschedule一下？By the way，你那个report写完了吗？",
        help="待评估的 CS 文本",
    )
    parser.add_argument("--archetype", type=str, default="ARC_01")
    parser.add_argument("--cmi-range", nargs=2, type=float, default=[0.05, 0.25])
    parser.add_argument("--formality", type=str, default="casual")
    parser.add_argument("--region", type=str, default="singapore")
    parser.add_argument(
        "--domain-words", nargs="*",
        default=["meeting", "deadline", "report"],
    )
    args = parser.parse_args()

    pipeline = RuleBasedEvaluatorPipeline()

    result = pipeline.evaluate(
        text=args.text,
        archetype_id=args.archetype,
        archetype_cmi_range=args.cmi_range,
        formality=args.formality,
        region=args.region,
        domain_words=args.domain_words,
    )

    print(pipeline.get_summary(result))

    # 打印详细分析
    print("--- 详细分析 ---")
    print(f"Tokens: {len(result.analysis.tokens)}")
    print(f"中文 tokens: {len(result.analysis.zh_tokens)}")
    print(f"英文 tokens: {len(result.analysis.en_tokens)}")
    print(f"Segments: {len(result.analysis.segments)}")
    for seg in result.analysis.segments:
        print(f"  [{seg['lang']}] {seg['text'][:50]}")
    print(f"Switch points: {len(result.analysis.switch_points)}")
    for sp in result.analysis.switch_points:
        print(
            f"  pos={sp.position}: {sp.from_lang}→{sp.to_lang} "
            f"({sp.switch_type}) motivation={sp.motivation or 'N/A'}"
        )
