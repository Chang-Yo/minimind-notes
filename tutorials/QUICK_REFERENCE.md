# MiniMind Quick Reference

A condensed guide for common tasks.

## Model Configurations

```python
# Dense model (64M params)
config = MiniMindConfig(
    hidden_size=768,
    num_hidden_layers=8,
    num_attention_heads=8,
    num_key_value_heads=4,
    use_moe=False
)

# MoE model (198M total / 64M active)
config = MiniMindConfig(
    hidden_size=768,
    num_hidden_layers=8,
    num_experts=4,
    num_experts_per_tok=1,
    use_moe=True
)
```

## Training Commands

```bash
# Pretraining
cd trainer && python train_pretrain.py --epochs 2 --batch_size 32

# SFT
cd trainer && python train_full_sft.py --from_weight pretrain --epochs 2

# LoRA
cd trainer && python train_lora.py --from_weight full_sft --lora_rank 16

# DPO
cd trainer && python train_dpo.py --from_weight full_sft --beta 0.1

# Resume training
python train_xxx.py --from_resume 1
```

## Inference Commands

```bash
# CLI chat
python eval_llm.py --weight full_sft --temperature 0.7

# With history
python eval_llm.py --weight full_sft --historys 4

# API server
cd scripts && python serve_openai_api.py --weight full_sft --port 8000
```

## Data Formats

### Pretrain (`pretrain_t2t*.jsonl`)
```jsonl
{"text": "Sample text for training..."}
```

### SFT (`sft_t2t*.jsonl`)
```jsonl
{"conversations": [
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]}
```

### DPO (`dpo.jsonl`)
```jsonl
{
  "chosen": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "Good"}],
  "rejected": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "Bad"}]
}
```

## Hyperparameter Cheat Sheet

| Stage | Learning Rate | Batch Size | Seq Len | Epochs |
|-------|---------------|------------|---------|--------|
| Pretrain | 5e-4 | 32 | 340 | 2 |
| SFT | 1e-5 | 16 | 768 | 2-5 |
| LoRA | 1e-5 | 16 | 768 | 2-3 |
| DPO | 1e-6 | 8 | 1024 | 2-3 |

## GPU Memory Optimization

```bash
# If OOM (Out of Memory)
--batch_size 8           # Reduce batch size
--accumulation_steps 4   # Increase accumulation
--max_seq_len 512        # Reduce sequence length
--num_workers 4          # Reduce data workers
```

## Generation Parameters

| Parameter | Range | Effect |
|-----------|-------|--------|
| temperature | 0.1-1.5 | Randomness (low=focused, high=creative) |
| top_p | 0.9-1.0 | Nucleus sampling |
| max_new_tokens | 128-8192 | Response length |
| repetition_penalty | 1.0-1.2 | Prevent repetition |

## File Locations

```
minimind/
├── model/                  # Model architecture & tokenizer
│   ├── model_minimind.py   # Main model
│   └── tokenizer.json      # Vocabulary
├── trainer/                # Training scripts
│   ├── train_pretrain.py   # Pretraining
│   ├── train_full_sft.py   # SFT
│   ├── train_lora.py       # LoRA
│   └── train_dpo.py        # DPO
├── dataset/                # Data files
│   └── *.jsonl             # Training data
├── scripts/                # Inference scripts
│   ├── serve_openai_api.py # API server
│   └── web_demo.py         # Web UI
├── out/                    # Saved weights
│   ├── pretrain_768.pth
│   └── full_sft_768.pth
└── checkpoints/            # Resume checkpoints
    └── *_resume.pth
```

## Common Issues

**Issue**: CUDA out of memory
**Fix**: Reduce `--batch_size`, increase `--accumulation_steps`

**Issue**: Loss not decreasing
**Fix**: Check learning rate, data quality, or try `--from_weight none` to train from scratch

**Issue**: Model not following instructions
**Fix**: Ensure SFT training completed, check data format

**Issue**: Repetitive outputs
**Fix**: Increase `--repetition_penalty`, lower `--temperature`

## Progress Tracking

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Check saved checkpoints
ls -lh out/
ls -lh checkpoints/

# Resume from checkpoint
python train_xxx.py --from_resume 1
```
