# Stage 1 理论基础：CS 行为原型的语言学文献支撑

## 1. 核心理论框架

### 1.1 Muysken (2000) — 三类 CS 类型学

**文献：** Muysken, P. (2000). *Bilingual Speech: A Typology of Code-Mixing*. Cambridge University Press.

**核心观点：** CS 可分为三种结构类型：
- **Insertion（嵌入）：** 一种语言的词汇项嵌入另一种语言的句法框架中。基质语言提供语法结构，嵌入语言提供词汇内容。特征：单个词或固定短语的嵌入，通常在名词短语位置。
- **Alternation（交替）：** 两种语言在句间或从句边界进行切换，每种语言各自保持完整的句法结构。特征：切换点在句法边界，切换前后各语言语法自洽。
- **Congruent Lexicalization（同构词汇化）：** 两种语言共享句法框架，词汇可自由来自任一语言。特征：密集的语内混合，难以区分基质/嵌入语言，常见于结构相似的语言对。

**对原型的映射：**
| Muysken 类型 | 对应原型 |
|---|---|
| Insertion | ARC_01 Insertional Switcher |
| Alternation | ARC_02 Alternational Switcher |
| Congruent Lexicalization | ARC_03 Dense Mixer |

---

### 1.2 Myers-Scotton (1993, 2002) — Matrix Language Frame (MLF) 模型

**文献：**
- Myers-Scotton, C. (1993). *Duelling Languages: Grammatical Structure in Codeswitching*. Oxford University Press.
- Myers-Scotton, C. (2002). *Contact Linguistics: Bilingual Encounters and Grammatical Outcomes*. Oxford University Press.

**核心观点：**
- 在 intra-sentential CS 中，存在一个**基质语言（Matrix Language, ML）** 提供句法框架（尤其是功能语素和词序），另一个**嵌入语言（Embedded Language, EL）** 提供内容语素。
- **ML Hypothesis：** ML 决定了混合句的形态句法框架。
- **System Morpheme Principle：** 系统语素（功能词、屈折标记）必须来自 ML。
- **Morpheme Order Principle：** 混合成分中的语素顺序由 ML 决定。

**对原型设计的指导：**
- ARC_01（Insertional Switcher）严格遵循 MLF 模型：L1 作为 ML，L2 词汇作为 EL 嵌入
- ARC_03（Dense Mixer）挑战 MLF 模型：两种语言共享框架时，ML/EL 界限模糊
- Post-Generation Validation 中的 ECT/MLF 硬约束检查基于此理论

---

### 1.3 Poplack (1980) — 切换约束与类型

**文献：** Poplack, S. (1980). Sometimes I'll start a sentence in Spanish Y TERMINO EN ESPAÑOL: toward a typology of code-switching. *Linguistics*, 18(7-8), 581-618.

**核心观点：**
- **Free Morpheme Constraint：** CS 不能发生在一个词的词干和它的约束语素之间（除非该词已被音位整合）。
- **Equivalence Constraint：** CS 倾向于发生在两种语言的表层词序一致的位置。
- **切换类型分类：**
  - Tag-switching：插入感叹词、话语标记等不受句法约束的元素（如 "you know"、"对吧"）
  - Inter-sentential switching：在句子/从句边界切换
  - Intra-sentential switching：在句内切换（需满足两种语言的句法约束）

**对原型的指导：**
- ARC_05（Reluctant Switcher）主要进行 tag-switching
- ARC_02（Alternational Switcher）主要进行 inter-sentential switching
- ARC_01 和 ARC_03 主要涉及 intra-sentential switching
- Equivalence Constraint 用于 Post-Generation Validation 的切换点合法性检查

---

### 1.4 Grosjean (1998, 2001) — 语言模式连续体

**文献：**
- Grosjean, F. (1998). Studying bilinguals: Methodological and conceptual issues. *Bilingualism: Language and Cognition*, 1(2), 131-149.
- Grosjean, F. (2001). The bilingual's language modes. In J. Nicol (Ed.), *One Mind, Two Languages: Bilingual Language Processing* (pp. 1-22). Blackwell.

**核心观点：**
- 双语者在任何时刻处于一个**语言模式连续体**上：从纯单语模式（仅激活一种语言）到完全双语模式（两种语言同时高度激活）。
- 影响语言模式的因素：
  - **对话者**：对方的语言能力和偏好
  - **情境**：正式度、话题、场所
  - **话语功能**：引用、强调、澄清
  - **社会规范**：社区对 CS 的态度

**对原型和 Language Mode Controller 的指导：**
- ARC_05（Reluctant Switcher）对应连续体的单语端
- ARC_03（Dense Mixer）对应连续体的双语端
- ARC_06（Accommodation Switcher）的行为本质上是在连续体上动态移动
- Language Mode Controller 的 formality/topic/interlocutor 调节因子直接来自此理论

---

### 1.5 Gumperz (1982) — 会话 CS 与语用功能

**文献：** Gumperz, J. J. (1982). *Discourse Strategies*. Cambridge University Press.

**核心观点：**
- CS 是一种**语境化线索（contextualization cue）**，用于标记话语中的社会和语用意义。
- **We-code / They-code 区分：** 少数族群语言（in-group, 亲密）vs. 主流语言（正式, 外部）。
- CS 服务的语用功能：
  - 引用（quotation）
  - 受众指定（addressee specification）
  - 感叹/插入语（interjection）
  - 重复/强调（reiteration）
  - 信息修饰（message qualification）
  - 角色/身份标记（personalization vs. objectivization）

**对原型的映射：**
- ARC_04（Pragmatic Switcher）直接基于此理论设计
- We-code/They-code 概念影响 ARC_04 的 example_description 中的语言选择策略

---

### 1.6 Giles — Communication Accommodation Theory (CAT)

**文献：**
- Giles, H., Coupland, N., & Coupland, J. (1991). Accommodation theory: Communication, context, and consequence. In H. Giles, J. Coupland, & N. Coupland (Eds.), *Contexts of Accommodation* (pp. 1-68). Cambridge University Press.
- Giles, H., & Ogay, T. (2007). Communication Accommodation Theory. In B. B. Whaley & W. Samter (Eds.), *Explaining Communication: Contemporary Theories and Exemplars* (pp. 293-310). Lawrence Erlbaum.

**核心观点：**
- **趋同（Convergence）：** 说话者调整自己的语言行为以更接近对话者，目的是获得认同、提高沟通效率或表达亲近。
- **趋异（Divergence）：** 说话者有意偏离对话者的语言模式，目的是强调群体身份差异或保持社交距离。
- 在 CS 语境中：双语者会根据对话者的 CS 程度调整自己的切换频率。

**对原型的映射：**
- ARC_06（Accommodation Switcher）直接基于 CAT 设计
- 多轮对话生成（模式 B）中的 Communication Accommodation 模拟基于此理论

---

## 2. 补充文献：CS 行为的量化研究

### 2.1 CMI (Code-Mixing Index)

**文献：** Gamback, B., & Das, A. (2014). On measuring the complexity of code-mixing. In *Proceedings of the 11th International Conference on Natural Language Processing*, 1-7.

**定义：** CMI = (N - max(Li)) / N × 100，其中 N 为总 token 数，max(Li) 为最多语言的 token 数。CMI ∈ [0, 100]，0 为纯单语，值越高混合越密集。

**在本项目中的使用：** 归一化到 [0, 1] 作为原型的 CMI_range 参数。

### 2.2 SEAME 语料库统计

**文献：**
- Lyu, D. C., Tan, T. P., Chng, E. S., & Li, H. (2015). An analysis of a Mandarin-English code-switching speech corpus: SEAME. *Age*, 21, 25-8.
- Li, H., Ma, B., & Lee, K. A. (2013). Spoken language recognition: from fundamentals to practice. *Proceedings of the IEEE*, 101(5), 1136-1159.

**SEAME 报告的关键统计：**
- 平均 CMI ≈ 0.20-0.35（因说话人差异大）
- Intra-sentential switching 占主导（~60-70%）
- Inter-sentential switching 约 20-30%
- Tag-switching 约 5-10%
- 切换点最常出现在 NP 和 VP 边界
- 说话人间 CS 行为差异显著（部分说话人 CMI < 0.05，部分 > 0.50）

### 2.3 ASCEND 语料库统计

**文献：** Lovenia, H., Cahyawijaya, S., Winata, G. I., et al. (2022). ASCEND: A spontaneous Chinese-English dataset for code-switching in multi-turn conversation. In *Proceedings of LREC 2022*.

**ASCEND 报告的关键统计：**
- 平均 CMI ≈ 0.15-0.25
- 以 Mandarin 为主导语言的说话人居多
- 英文嵌入以名词和固定短语为主
- 自发对话，包含丰富的犹豫、修正和填充词

### 2.4 其他相关量化研究

- **Bullock & Toribio (2009):** *The Cambridge Handbook of Linguistic Code-switching*. 提供了跨语言对的 CS 类型学综述。
- **Auer (1999):** From codeswitching via language mixing to fused lects. *International Journal of Bilingualism*, 3(4), 309-332. 提出了从 CS 到语言融合的连续体模型。
- **Li Wei (2005):** "How can you tell?" Towards a common sense explanation of conversational code-switching. *Journal of Pragmatics*, 37(3), 375-389. 强调 CS 的交际策略功能。

---

## 3. 2000 年后的理论更新

### 3.1 Muysken (2013)：第四种 CS 类型 — Backflagging

**文献：** Muysken, P. (2013). Language contact outcomes as the result of bilingual optimization strategies. *Bilingualism: Language and Cognition*, 16(4), 709-730.

**核心扩展：** Muysken 将自己 2000 年的三分类型学扩展为**四分类型学**，新增 **Backflagging**：
- 在 L2 主导的话语中，插入 L1 的话语标记、感叹词或边缘性语法成分
- 与 Insertion 形成镜像：Insertion 是 L1 框架+L2 嵌入；Backflagging 是 L2 框架+L1 标记
- 典型场景：移民二代在社会主导语言话语中保留母语感叹词和话语标记

**对原型的映射：** → ARC_07 Backflag Switcher

---

### 3.2 Myers-Scotton (2002)：4-M 模型

**文献：** Myers-Scotton, C. (2002). *Contact Linguistics: Bilingual Encounters and Grammatical Outcomes*. Oxford University Press.

**核心精细化：** 将语素分为四类，精确预测 CS 中哪些元素可以来自嵌入语言：
1. Content morphemes（内容语素）— 概念激活，可来自 EL ✓
2. Early system morphemes（早期系统语素）— 概念激活，可来自 EL ✓
3. Late bridge system morphemes（晚期桥接系统语素）— 结构分配，必须来自 ML ✗
4. Late outsider system morphemes（晚期外部系统语素）— 结构分配，必须来自 ML ✗

**对原型的影响：** 精细化 ARC_01 的 `syntactic_positions.morpheme_constraint`

---

### 3.3 Green & Abutalebi (2013)：适应性控制假说（ACH）

**文献：** Green, D.W. & Abutalebi, J. (2013). Language control in bilinguals: The adaptive control hypothesis. *Journal of Cognitive Psychology*, 25(5), 515-530.

**核心框架：** 2000 年后最有影响力的 CS 认知理论。区分三种交互语境：

| 语境类型 | 控制需求 | 对应原型 |
|---------|---------|---------|
| Single-language（单语） | 强抑制性控制 | ARC_05 |
| Dual-language（双语） | 竞争性控制 | ARC_02, ARC_06 |
| Dense code-switching（密集CS） | 合作性/开放控制 | ARC_03 |

**关键贡献：** 为行为层面的原型分类提供了**认知机制层面的解释**——不同 CS 行为对应不同的认知控制策略。

---

### 3.4 Green & Wei (2014)：控制过程模型（CPM）

**文献：** Green, D.W. & Wei, L. (2014). A control process model of code-switching. *Language, Cognition and Neuroscience*, 29(4), 499-511.

**核心创新：** 将 Muysken 类型学与认知控制机制直接对接：

| 控制模式 | Muysken 类型 | 对应原型 |
|---------|-------------|---------|
| Competitive（竞争） | — | ARC_05, ARC_07 |
| Coupled（耦合） | Insertion, Alternation | ARC_01, ARC_02, ARC_04 |
| Open（开放） | Congruent Lexicalization | ARC_03 |
| Adaptive（适应） | — | ARC_06 |

---

### 3.5 Gullifer & Titone (2021)：语言熵方法

**文献：** Gullifer, J.W. & Titone, D. (2021). Bilingualism: A neurocognitive exercise in managing uncertainty. *Neurobiology of Language*, 2(4), 464-486.

**核心创新：** 引入信息论的"熵"概念量化双语使用多样性：
- Language Entropy: 最小值 = 完全单语，最大值 = 两种语言完全均等使用
- 发现双语者发展出不同的**吸引态（attractor states）**，与原型分类天然对应
- 揭示 CS 行为不仅受交际意图驱动，还受**长期神经认知适应**塑造

---

### 3.6 Bhatt & Bolonyai (2011)：OT 框架

**文献：** Bhatt, R.M. & Bolonyai, A. (2011). Code-switching and the optimal grammar of bilingual language use. *Bilingualism: Language and Cognition*, 14(4), 522-546.

**核心贡献：** 使用最优论建模 CS，提出五个普遍性约束：
- **Faith**（忠实性）、**Power**（权力）、**Solidarity**（团结）、**Face**（面子）、**Perspective**（视角）
- 不同双语社区中这五个约束的层级排序不同，解释了跨社区的 CS 差异

**对原型的影响：** 补充 ARC_04 的语用驱动机制

---

### 3.7 Auer & Hakimov (2021)：CS → LM → FL 连续体更新

**文献：** Auer, P. & Hakimov, N. (2021). From language mixing to fused lects: The process and its outcomes. *International Journal of Bilingualism*, 25(2), 361-386.

**核心更新：** CS 向融合语码的演变是**结构沉淀（structural sedimentation）** 过程：
- Code-Switching（CS）→ Language Mixing（LM）→ Fused Lect（FL）
- 每一阶段的特征：CS 阶段切换有局部意义（marked）；LM 阶段单次切换失去显著性；FL 阶段两种语言深度语法化融合

---

### 3.8 Li Wei (2018)：Translanguaging

**文献：** Li Wei (2018). Translanguaging as a practical theory of language. *Applied Linguistics*, 39(1), 9-30.

**认识论挑战：** 多语者拥有一个统一语言系统（unitary linguistic repertoire），"命名语言"是社会建构。Translanguaging 不是"在语言间切换"，而是"部署完整语言资源库"。

**实用评估：** 对计算建模操作性有限，但提醒 ARC_03 更应被视为"融合语码"而非"高频切换"。已纳入 ARC_03 理论基础。

---

### 3.9 计算 CS 研究综述

**文献：** Winata, G., Aji, A.F., Yong, Z.X., & Solorio, T. (2023). The decades progress on code-switching research in NLP: A systematic survey on trends and challenges. *Findings of ACL 2023*, 2936-2978.

**补充度量指标（可扩展）：**
- **I-index (Integration Index)**：切换点分布的均匀性
- **Burstiness**：切换的聚集 vs. 分散程度
- 计算社区仍主要使用 Muysken 三分类型学 + CMI

---

## 4. 理论到原型的映射总结（更新版）

| 原型 | 主要理论来源 | 认知控制模式 (CPM) | 理论关键概念 |
|------|-------------|-------------------|-------------|
| ARC_01 Insertional | Muysken (Insertion); Myers-Scotton (MLF + 4-M) | Coupled | ML 框架 + EL 内容语素嵌入 |
| ARC_02 Alternational | Muysken (Alternation); Poplack (inter-sentential) | Coupled/Competitive | 句法边界切换，各语言独立完整 |
| ARC_03 Dense Mixer | Muysken (CL); Auer (fused lect); ACH (dense CS); Li Wei (translanguaging) | Open | 共享框架/融合语码，词汇自由取用 |
| ARC_04 Pragmatic | Gumperz (cues); Bhatt & Bolonyai (OT) | Coupled (variable) | CS 作为语用标记工具 |
| ARC_05 Reluctant | Grosjean (monolingual end); ACH (single-language) | Competitive | 强抑制控制，仅 tag-switching |
| ARC_06 Accommodation | Giles (CAT); ACH (context-adaptive) | Adaptive | 根据对话者动态调整控制策略 |
| ARC_07 Backflag | Muysken (2013) Backflagging | Competitive | L2 框架 + L1 话语标记/感叹词（ARC_01 镜像） |
