# Superpod Offline Synthesis

## 目录结构

```
superpod/
├── synthesize_offline.py   # 离线合成脚本（直接加载模型，无需API服务）
├── setup_env.sh            # 一次性环境配置（conda + fish-speech + 模型下载）
├── job_synthesize.slurm    # SLURM 任务脚本
├── submit.sh               # 快速提交助手
├── asset/                  # 参考音频（需手动放入）
│   ├── zh_male_01.wav
│   ├── zh_male_01.txt
│   ├── de_female_01.wav
│   ├── de_female_01.txt
│   └── ...
└── README.md
```

## 使用步骤

### 1. 首次配置

```bash
bash superpod/setup_env.sh
```

### 2. 放入参考音频

将 `{lang}_{gender}_{id}.wav` + `.txt` 放入 `superpod/asset/`

### 3. 放入 Stage 1 数据

将 JSONL 文件放入 `output/` 目录，命名为 `{lang_pair}_dialogues.jsonl`

### 4. 提交任务

```bash
# 单个语言
bash superpod/submit.sh zh_en

# 限制条数
bash superpod/submit.sh de_en 100

# 所有语言
bash superpod/submit.sh all
```

### 5. 查看状态

```bash
squeue -u $USER              # 查看队列
tail -f logs/tts_JOBID.out   # 查看日志
```
