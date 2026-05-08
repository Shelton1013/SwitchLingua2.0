# Stage 2 Offline — Direct Model Loading Synthesis

与 `stage2/` 的区别：
- `stage2/`: 需要先部署 Fish Speech API 服务，通过 HTTP URL 调用
- `stage2_offline/`: 直接在脚本内加载模型到 GPU，一个命令完成全部合成

## 文件结构

```
stage2_offline/
├── synthesize.py   # 主合成脚本（直接加载模型）
├── setup.sh        # 一次性环境配置
├── job.slurm       # SLURM 任务提交脚本
├── asset/          # 参考音频（需手动放入）
│   ├── zh_male_01.wav + .txt
│   ├── zh_female_01.wav + .txt
│   └── ...
└── README.md
```

## 支持的模型版本

| 版本 | 模型 | VRAM | 质量 |
|------|------|------|------|
| v2 (默认) | Fish Speech S2-Pro | ~24GB | 最好 |
| v1.5 | Fish Speech 1.5 | ~8GB | 良好 |

## 使用方法

### 1. 环境配置（一次性）
```bash
bash stage2_offline/setup.sh v2    # 或 v1.5
```

### 2. 放入参考音频
```bash
# 命名格式：{语言}_{性别}_{编号}.wav + .txt
cp zh_male_01.wav stage2_offline/asset/
echo "参考音频的文字内容" > stage2_offline/asset/zh_male_01.txt
```

### 3. 运行合成

本地运行：
```bash
python stage2_offline/synthesize.py \
    --input output/zh_en_dialogues.jsonl \
    --output output/stage2/zh_en/ \
    --asset-dir stage2_offline/asset \
    --model-dir ~/models/s2-pro \
    --version v2 \
    --device cuda:0 \
    --limit 10
```

提交 SLURM 任务：
```bash
sbatch stage2_offline/job.slurm
sbatch --export=LANG_PAIR=de_en,LIMIT=50 stage2_offline/job.slurm
```
