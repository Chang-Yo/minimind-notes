# MiniMind Learning Path

This guide will help you understand LLM training from scratch through the MiniMind codebase. MiniMind is perfect for learning because it's small enough to run on consumer hardware (like your RTX 4070 Laptop) while implementing all the key components of modern LLM training.

## Why Learn from MiniMind?

- **Tiny but complete**: ~64M parameters vs GPT-3's 175B (1/2700th the size)
- **Full pipeline**: Pretrain → SFT → RLHF/DPO/RLAIF
- **From scratch**: Core algorithms implemented in pure PyTorch
- **Fast training**: ~2 hours on a single 3090 (your 4070 Laptop will be slower but still feasible)
- **Production-ready**: Compatible with transformers, llama.cpp, vllm

## Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Verify CUDA is available
python -c "import torch; print(torch.cuda.is_available())"
```

## Learning Path Overview

```
Stage 1: Model Architecture (2-3 hours)
    ↓
Stage 2: Data & Tokenization (1-2 hours)
    ↓
Stage 3: Pretraining (4-6 hours actual training)
    ↓
Stage 4: Supervised Fine-Tuning (2-4 hours)
    ↓
Stage 5: Advanced Techniques (optional)
```

## Stage 1: Model Architecture

**Goal**: Understand how a Transformer-based LLM is constructed.

**Files to study**:
- `model/model_minimind.py` - Core model implementation
- `model/model_lora.py` - LoRA adaptation

**Key concepts to learn**:
1. **Transformer Decoder-Only architecture**
2. **Attention mechanism** (self-attention, multi-head, KV cache)
3. **Position encoding** (RoPE - Rotary Position Embedding)
4. **Feed-forward networks** (SwiGLU activation)
5. **Layer normalization** (RMSNorm)
6. **MoE (Mixture of Experts)** - optional advanced topic

**See**: `tutorials/01_model_architecture.md`

## Stage 2: Data & Tokenization

**Goal**: Understand how text is converted into model input.

**Files to study**:
- `dataset/lm_dataset.py` - Data loading classes
- `model/tokenizer.json` - The vocabulary

**Key concepts to learn**:
1. **Tokenization** - Text → Token IDs
2. **Chat templates** - Formatting conversations
3. **Data preprocessing** - Truncation, padding
4. **Tool calling format** - Function calling support

**See**: `tutorials/02_data_and_tokenization.md`

## Stage 3: Pretraining

**Goal**: Learn how models learn from raw text.

**Files to study**:
- `trainer/train_pretrain.py` - Pretraining script
- `trainer/trainer_utils.py` - Training utilities

**Key concepts to learn**:
1. **Next-token prediction** - The core LLM objective
2. **Loss calculation** - Cross-entropy loss
3. **Optimizer & learning rate schedule**
4. **Gradient accumulation** - Handling batch size constraints
5. **Checkpointing & resuming**

**Practice**: Run pretraining with mini dataset
```bash
cd trainer && python train_pretrain.py
```

**See**: `tutorials/03_pretraining.md`

## Stage 4: Supervised Fine-Tuning (SFT)

**Goal**: Learn how to make models follow instructions.

**Files to study**:
- `trainer/train_full_sft.py` - SFT training script

**Key concepts to learn**:
1. **Instruction format** - User/Assistant conversation structure
2. **Chat template system** - Special tokens for formatting
3. **Teacher forcing** - Training on ground truth responses
4. **Tool calling training** - Function invocation

**Practice**: Run SFT training
```bash
cd trainer && python train_full_sft.py
```

**See**: `tutorials/04_supervised_finetuning.md`

## Stage 5: Advanced Techniques (Optional)

**Goal**: Explore cutting-edge LLM training methods.

**Topics**:
- **LoRA** - Parameter-efficient fine-tuning (`train_lora.py`)
- **DPO** - Direct Preference Optimization (`train_dpo.py`)
- **PPO/GRPO** - Reinforcement learning from AI feedback (`train_ppo.py`, `train_grpo.py`)
- **Agentic RL** - Training for tool use (`train_agent.py`)
- **Distillation** - Knowledge transfer from larger models (`train_distillation.py`)

**See**: `tutorials/05_advanced_techniques.md`

## Stage 6: Inference & Deployment

**Goal**: Learn how to use trained models.

**Files to study**:
- `eval_llm.py` - CLI inference
- `scripts/serve_openai_api.py` - API server
- `scripts/web_demo.py` - Web UI

**Key concepts to learn**:
1. **Generation strategies** - Temperature, top-p, top-k sampling
2. **KV cache** - Efficient generation
3. **Streaming output** - Token-by-token generation
4. **OpenAI API compatibility** - Standard serving format

**See**: `tutorials/06_inference.md`

## Recommended Learning Order

1. **Read first** - Start with Stage 1-4 (architecture → pretrain → SFT)
2. **Run experiments** - Execute training scripts with mini datasets
3. **Observe** - Watch loss curves, model outputs change
4. **Modify** - Try changing hyperparameters, model size
5. **Advanced** - Explore RLHF, LoRA after understanding basics

## Tips for Your RTX 4070 Laptop

- **Use mini datasets**: `pretrain_t2t_mini.jsonl`, `sft_t2t_mini.jsonl`
- **Reduce batch size** if needed (adjust in training scripts)
- **Monitor GPU memory**: `nvidia-smi` in another terminal
- **Use checkpoint resume** if training gets interrupted
- **Start with smaller model** if needed (reduce `d_model` or `n_layers`)

## Expected Timeline

| Stage | Reading | Running (mini dataset) |
|-------|---------|------------------------|
| Architecture | 2-3 hours | - |
| Data & Tokenization | 1-2 hours | - |
| Pretraining | 1 hour | 4-6 hours |
| SFT | 1 hour | 2-4 hours |
| **Total (core)** | **5-7 hours** | **6-10 hours** |

## Next Steps

Start with `tutorials/01_model_architecture.md` to begin your journey!
