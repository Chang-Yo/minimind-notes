# Tutorial 3: Pretraining

This tutorial explains how LLMs learn from raw text through next-token prediction - the foundation of all language model training.

## Learning Objectives

After this tutorial, you will understand:
1. The pretraining objective (next-token prediction)
2. Training loop structure
3. Mixed precision training
4. Gradient accumulation
5. Learning rate scheduling
6. Checkpointing and resuming

## File: `trainer/train_pretrain.py`

---

## 1. What is Pretraining?

**Pretraining** is the process of training a model on large amounts of raw text to learn:
- Language patterns and grammar
- World knowledge and facts
- Reasoning abilities
- Text generation capabilities

**The Objective**: Predict the next token given previous tokens.

```
Input:  "The capital of France is"
Target: "Paris"
Loss:   CrossEntropy(predicted_distribution, "Paris")
```

**Why this works**: By learning to predict the next word, the model implicitly learns:
- Grammar (what words follow others)
- Facts (Paris follows "capital of France")
- Reasoning (logical connections between concepts)

---

## 2. Training Script Overview

```python
# Main structure (simplified)
if __name__ == "__main__":
    # 1. Initialize environment
    local_rank = init_distributed_mode()
    setup_seed(42)

    # 2. Create model and data
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)

    # 3. Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 4. Train for multiple epochs
    for epoch in range(args.epochs):
        train_epoch(epoch, loader, len(loader), wandb)
```

---

## 3. Training Loop - Step by Step

### 3.1 Forward Pass

```python
def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)

        # Forward pass
        with autocast_ctx:  # Mixed precision
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps
```

**What happens**:
1. Move batch to GPU
2. Forward pass through model
3. Calculate loss (cross-entropy for next-token prediction)
4. Scale loss by accumulation steps

**Loss calculation in model**:
```python
# In model_minimind.py
logits = self.lm_head(hidden_states)  # Shape: (batch, seq_len, vocab_size)

# Shift for next-token prediction
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()

# Cross-entropy loss
loss = F.cross_entropy(
    shift_logits.view(-1, vocab_size),
    shift_labels.view(-1),
    ignore_index=-100  # Don't compute loss on padding
)
```

### 3.2 Backward Pass

```python
        # Backward pass
        scaler.scale(loss).backward()
```

**Mixed precision training**:
```python
scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
```

**Why mixed precision?**
- Use `bfloat16` for forward/backward (faster, less memory)
- Keep `float32` master weights for stability
- `GradScaler` prevents underflow

### 3.3 Gradient Accumulation

```python
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
```

**What is gradient accumulation?**

Training with effective batch size = `batch_size × accumulation_steps`:

```
# Effective batch = 32 × 8 = 256
Step 1: backward(), accumulate gradients
Step 2: backward(), accumulate gradients
...
Step 8: backward(), accumulate gradients, then optimizer.step()
```

**Why accumulate?**
- Limited GPU memory can't fit large batches
- Larger batches = more stable training
- Accumulate gradients, then update once

### 3.4 Learning Rate Schedule

```python
lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
for param_group in optimizer.param_groups:
    param_group['lr'] = lr
```

**Cosine annealing with warmup**:
```python
def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))
```

**Shape**: Starts at ~0.55×lr, decays to ~0.1×lr

```
LR
│
│╲         ╱
│ ╲       ╱
│  ╲____╱    (cosine decay)
│
└─────────────── Step
```

### 3.5 Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
```

**What it does**: Limits gradient magnitude to prevent exploding gradients.

**Why**: Training can become unstable if gradients get too large.

### 3.6 Logging

```python
if step % args.log_interval == 0 or step == iters:
    current_loss = loss.item() * args.accumulation_steps
    current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
    current_logits_loss = current_loss - current_aux_loss
    Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}')
```

**Output example**:
```
Epoch:[1/2](100/1000), loss: 3.2456, logits_loss: 3.2456, aux_loss: 0.0000, lr: 0.00045000, epoch_time: 45.2min
```

### 3.7 Checkpointing

```python
if (step % args.save_interval == 0 or step == iters) and is_main_process():
    model.eval()
    state_dict = raw_model.state_dict()
    torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
    lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, ...)
    model.train()
```

**Two types of checkpoints**:
1. **Model weights** (`out/pretrain_768.pth`): Just the model parameters
2. **Resume checkpoint** (`checkpoints/pretrain_768_resume.pth`): Model + optimizer + epoch + step

---

## 4. Command Line Arguments

```bash
python train_pretrain.py \
    --epochs 2 \
    --batch_size 32 \
    --learning_rate 5e-4 \
    --accumulation_steps 8 \
    --max_seq_len 340 \
    --data_path ../dataset/pretrain_t2t_mini.jsonl \
    --save_weight pretrain
```

**Key arguments**:

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 2 | Number of training epochs |
| `--batch_size` | 32 | Batch size per GPU |
| `--learning_rate` | 5e-4 | Initial learning rate |
| `--accumulation_steps` | 8 | Gradient accumulation steps |
| `--max_seq_len` | 340 | Max sequence length (tokens) |
| `--data_path` | pretrain_t2t_mini.jsonl | Training data |
| `--save_weight` | pretrain | Checkpoint name prefix |
| `--from_resume` | 0 | Auto-resume from checkpoint |

---

## 5. Distributed Training

**Multi-GPU training**:
```bash
torchrun --nproc_per_node 4 trainer/train_pretrain.py
```

**What happens**:
1. Each GPU gets a portion of the batch
2. Gradients are synchronized across GPUs
3. Effective batch size = `batch_size × num_gpus × accumulation_steps`

---

## 6. Running Your First Training

### Step 1: Download data
```bash
# Download pretrain_t2t_mini.jsonl to ./dataset/
# From: https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files
```

### Step 2: Run training
```bash
cd trainer
python train_pretrain.py
```

### Step 3: Monitor progress
Watch the loss decrease:
```
Epoch:[1/2](100/1000), loss: 4.5234, lr: 0.00045000
Epoch:[1/2](200/1000), loss: 3.8212, lr: 0.00044800
Epoch:[1/2](300/1000), loss: 3.4521, lr: 0.00044300
...
```

**What to expect**:
- Loss starts high (~4-5)
- Gradually decreases
- Converges to ~2-3 for small models

### Step 4: Check output
```bash
ls -lh ../out/
# pretrain_768.pth  <- Your trained model!
```

---

## 7. Resume Training

If training is interrupted:
```bash
python train_pretrain.py --from_resume 1
```

**What happens**:
1. Automatically detects latest checkpoint
2. Restores model, optimizer, epoch, step
3. Continues from where it left off
4. WandB run continues (same run ID)

---

## 8. Tips for Your RTX 4070 Laptop

### Reduce memory usage:
```bash
python train_pretrain.py \
    --batch_size 16 \      # Reduce batch size
    --accumulation_steps 16 \  # Increase accumulation to compensate
    --max_seq_len 256 \     # Reduce sequence length
    --num_workers 4         # Reduce CPU workers
```

### Monitor GPU:
```bash
# In another terminal
watch -n 1 nvidia-smi
```

### Expected training time (mini dataset):
| GPU | Batch Size | Time per epoch |
|-----|------------|----------------|
| RTX 4090 | 32 | ~30 min |
| RTX 4070 Laptop | 16 | ~60-90 min |
| RTX 4060 Laptop | 8 | ~2-3 hours |

---

## Exercise: Modify and Train

Try these modifications:

1. **Change learning rate**:
   ```bash
   python train_pretrain.py --learning_rate 1e-3
   ```

2. **Train for more epochs**:
   ```bash
   python train_pretrain.py --epochs 5
   ```

3. **Use smaller model**:
   ```bash
   python train_pretrain.py --hidden_size 512 --num_hidden_layers 6
   ```

---

## Key Takeaways

1. **Next-token prediction**: The core LLM training objective
2. **Mixed precision**: bfloat16 for speed, float32 for stability
3. **Gradient accumulation**: Simulate larger batch sizes
4. **Learning rate schedule**: Cosine annealing for smooth convergence
5. **Checkpointing**: Save progress for resuming
6. **Distributed training**: Multiple GPUs for faster training

---

## Next Step

After pretraining, the model can generate text but doesn't follow instructions. Next, we'll do Supervised Fine-Tuning!

**Next**: `tutorials/04_supervised_finetuning.md`
