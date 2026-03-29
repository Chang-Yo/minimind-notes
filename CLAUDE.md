# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MiniMind is a lightweight LLM training framework for training tiny language models (~64M parameters) from scratch. The project implements the complete training pipeline (Pretrain → SFT → RLHF/DPO/RLAIF) using native PyTorch without relying on high-level framework abstractions like `trl` or `peft` for core algorithms.

## Key Commands

### Training
```bash
# Pretraining (required first step)
cd trainer && python train_pretrain.py

# Supervised Fine-Tuning (required second step)
cd trainer && python train_full_sft.py

# Optional: RLHF/RLAIF training
cd trainer && python train_dpo.py       # DPO preference optimization
cd trainer && python train_ppo.py       # PPO reinforcement learning
cd trainer && python train_grpo.py      # GRPO (group relative policy optimization)

# Optional: Other training modes
cd trainer && python train_lora.py          # LoRA fine-tuning
cd trainer && python train_distillation.py  # Model distillation
cd trainer && python train_agent.py         # Agentic RL with tool use
```

### Multi-GPU Training
```bash
# Single machine, N GPUs
torchrun --nproc_per_node N trainer/train_xxx.py
```

### Checkpoint Resume
All training scripts support automatic checkpoint resume:
```bash
python train_xxx.py --from_resume 1
```
Checkpoints are saved in `./checkpoints/` with format `<weight>_<dim>_resume.pth`.

### Inference/Evaluation
```bash
# Using transformers format model
python eval_llm.py --load_from ./minimind-3

# Using native PyTorch weights from ./out/ directory
python eval_llm.py --load_from ./model --weight full_sft

# Web UI (requires streamlit, copy model to ./scripts/ first)
cd scripts && streamlit run web_demo.py
```

### Model Conversion
```bash
# Merge LoRA weights into base model
python scripts/convert_model.py --base_model ./minimind-3 --lora_path ./out/lora_xxx.pth --output_path ./minimind-3-merged
```

## Architecture

### Model Structure (`model/model_minimind.py`)
- **Base**: Transformer Decoder-Only with Pre-Norm + RMSNorm
- **Activation**: SwiGLU
- **Position Encoding**: RoPE with YaRN extrapolation support
- **Alignment**: Compatible with Qwen3/Qwen3-MoE ecosystem
- **Default Config**: `d_model=768`, `n_layers=8`, `q_heads=8`, `kv_heads=4`

### Model Variants
- **Dense**: Standard transformer (`minimind-3`, ~64M params)
- **MoE**: Mixture of Experts (`minimind-3-moe`, ~198M total/64M active, 4 experts/top-1)

### Training Pipeline
1. **Pretrain** (`train_pretrain.py`) - Learn from raw text data
2. **SFT** (`train_full_sft.py`) - Instruction following with chat format
3. **Optional RL**: DPO/PPO/GRPO/Agentic RL for alignment

## Data Format

### Pretrain Data (`dataset/pretrain_t2t*.jsonl`)
```jsonl
{"text": "Sample text for next-token prediction training..."}
```

### SFT Data (`dataset/sft_t2t*.jsonl`)
```jsonl
{
  "conversations": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
  ]
}
```

Tool Call format includes `tools`, `tool_calls`, and `reasoning_content` fields.

## Project Structure
```
minimind/
├── model/              # Model architectures (MiniMind, MoE, LoRA)
├── trainer/            # All training scripts and utilities
├── dataset/            # Dataset loading classes
├── scripts/            # Inference/serving scripts
├── eval_llm.py         # CLI evaluation and chat
├── out/                # Saved model weights (*.pth)
└── checkpoints/        # Training checkpoints for resume
```

## Important Notes
- The project uses a custom 6400-vocab tokenizer (`model/tokenizer.json`)
- Core algorithms (DPO, PPO, GRPO, LoRA) are implemented from scratch in PyTorch
- Models are compatible with `transformers`, `llama.cpp`, `vllm`, `ollama`
- Default training uses SwanLab for visualization (WandB-compatible API)
- Tool calling capability is included in the main SFT data (no separate training needed)
