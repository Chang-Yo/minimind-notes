# Tutorial 5: Advanced Training Techniques

This tutorial covers advanced LLM training methods beyond basic pretraining and SFT.

## Learning Objectives

After this tutorial, you will understand:
1. LoRA - Parameter-efficient fine-tuning
2. DPO - Direct Preference Optimization
3. RLAIF - Reinforcement Learning from AI Feedback (PPO/GRPO)
4. When to use each technique

---

## 1. LoRA (Low-Rank Adaptation)

### What is LoRA?

**LoRA** freezes the main model weights and adds small trainable adapter matrices. Instead of updating all 64M parameters, you only train ~1-2M parameters.

**Key idea**: A weight update ΔW can be approximated by two low-rank matrices:
```
W_new = W_frozen + ΔW
ΔW ≈ A × B
```
Where A is (d × r) and B is (r × d), with r << d (rank).

### LoRA Implementation

```python
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank
        self.A = nn.Linear(in_features, rank, bias=False)  # d × r
        self.B = nn.Linear(rank, out_features, bias=False)  # r × d

        # Initialization
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        self.B.weight.data.zero_()  # Start with no change

    def forward(self, x):
        return self.B(self.A(x))
```

### Applying LoRA to a Model

```python
def apply_lora(model, rank=16):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank)
            setattr(module, "lora", lora)

            # Modify forward: original_output + lora_output
            original_forward = module.forward
            def forward_with_lora(x, orig=original_forward, lora_layer=lora):
                return orig(x) + lora_layer(x)
            module.forward = forward_with_lora
```

### Parameter Savings

| Component | Full Fine-tuning | LoRA (rank=16) | Savings |
|-----------|------------------|----------------|---------|
| Embedding | 5M params | Frozen | 100% |
| Attention | ~20M params | ~200K params | 99% |
| FFN | ~35M params | ~350K params | 99% |
| **Total trainable** | **64M** | **~550K** | **99.1%** |

### When to Use LoRA

✅ **Use LoRA when**:
- Limited GPU memory
- Training on specific tasks/domains
- Want to quickly experiment
- Need multiple task-specific adapters

❌ **Don't use LoRA when**:
- Training from scratch
- Want maximum quality
- Have abundant compute

### Running LoRA Training

```bash
cd trainer
python train_lora.py \
    --from_weight full_sft \
    --save_weight lora_mytask \
    --data_path ../dataset/sft_t2t_mini.jsonl \
    --lora_rank 16
```

### Merging LoRA Weights

After training, merge LoRA weights into the base model:

```bash
python scripts/convert_model.py \
    --base_model ./minimind-3 \
    --lora_path ./out/lora_mytask_768.pth \
    --output_path ./minimind-3-mytask
```

Or in code:
```python
def merge_lora(model, lora_path, save_path):
    load_lora(model, lora_path)

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and hasattr(module, 'lora'):
            # W_new = W_old + B × A
            module.weight.data += (module.lora.B.weight @ module.lora.A.weight)

    torch.save(model.state_dict(), save_path)
```

---

## 2. DPO (Direct Preference Optimization)

### What is DPO?

**DPO** trains models to prefer better responses without a separate reward model. It directly optimizes the probability that chosen responses are rated higher than rejected ones.

**vs RLHF**: DPO is simpler and more stable than PPO-based RLHF.

### DPO Loss Function

```python
def dpo_loss(ref_log_probs, policy_log_probs, mask, beta=0.1):
    # Compute log probabilities for chosen and rejected
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]

    # Log ratios
    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs

    # DPO loss
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)

    return loss.mean()
```

**Intuition**: Maximize `log(p_chosen/p_rejected)` relative to reference model.

### DPO Data Format

```jsonl
{
  "chosen": [
    {"content": "What's 2+2?", "role": "user"},
    {"content": "2+2 equals 4.", "role": "assistant"}
  ],
  "rejected": [
    {"content": "What's 2+2?", "role": "user"},
    {"content": "I don't know.", "role": "assistant"}
  ]
}
```

### DPO Training Loop

```python
# Two models: policy (trainable) and reference (frozen)
ref_model = init_model(lm_config, from_weight='full_sft')
for param in ref_model.parameters():
    param.requires_grad = False  # Freeze reference

# Training loop
for batch in loader:
    # Reference model (frozen)
    with torch.no_grad():
        ref_outputs = ref_model(x)
        ref_log_probs = logits_to_log_probs(ref_outputs.logits, y)

    # Policy model (trainable)
    outputs = model(x)
    policy_log_probs = logits_to_log_probs(outputs.logits, y)

    # DPO loss
    loss = dpo_loss(ref_log_probs, policy_log_probs, mask)
    loss.backward()
    optimizer.step()
```

### Running DPO Training

```bash
cd trainer
python train_dpo.py \
    --from_weight full_sft \
    --save_weight dpo \
    --data_path ../dataset/dpo.jsonl \
    --beta 0.1
```

---

## 3. RLAIF (RL from AI Feedback)

### Overview

**RLAIF** uses reinforcement learning where an AI model provides feedback/rewards instead of humans.

**Methods in MiniMind**:
- **PPO**: Proximal Policy Optimization
- **GRPO**: Group Relative Policy Optimization
- **CISPO**: CI-SP Optimization

### PPO Training Structure

```python
# 1. Generate responses using rollout engine
prompts = [sample['prompt'] for sample in batch]
responses = rollout_engine.generate(model, prompts)

# 2. Get rewards from reward model
rewards = reward_model.get_score(prompts, responses)

# 3. Compute PPO loss
policy_ratio = policy_probs / old_policy_probs
surrogate1 = policy_ratio * advantages
surrogate2 = torch.clamp(policy_ratio, 1 - eps, 1 + eps) * advantages
policy_loss = -torch.min(surrogate1, surrogate2).mean()

# 4. Value function loss (if using critic)
value_loss = F.mse_loss(value_preds, returns)

# 5. Entropy bonus (for exploration)
entropy_bonus = entropy.mean()

loss = policy_loss + value_loss - entropy_bonus
```

### GRPO (Group Relative Policy Optimization)

**GRPO** simplifies PPO by removing the value function and using group-wise normalization:

```python
# Group responses by prompt
groups = group_by_prompt(responses)

# Compute group-relative advantages
for group in groups:
    group_mean = mean(group.rewards)
    group_advantages = [r - group_mean for r in group.rewards]

# Optimize using these advantages
loss = -log_probs * group_advantages
```

### Running RLAIF Training

```bash
# PPO
python train_ppo.py \
    --from_weight full_sft \
    --save_weight ppo_actor \
    --data_path ../dataset/rlaif.jsonl

# GRPO
python train_grpo.py \
    --from_weight full_sft \
    --save_weight grpo \
    --data_path ../dataset/rlaif.jsonl
```

---

## 4. Agentic RL (Tool Use Training)

Training models to better use tools through reinforcement learning.

```bash
python train_agent.py \
    --from_weight full_sft \
    --save_weight agent \
    --data_path ../dataset/agent_rl.jsonl
```

**What it learns**:
- When to call tools
- Correct tool invocation format
- Handling tool responses
- Multi-step reasoning with tools

---

## 5. Model Distillation

Transfer knowledge from a larger model to MiniMind.

```bash
python train_distillation.py \
    --teacher_model qwen3-4b \
    --from_weight full_sft \
    --save_weight distilled
```

**Approaches**:
1. **Logit distillation**: Match teacher's output distribution
2. **Hidden state distillation**: Match intermediate representations
3. **Feature distillation**: Match attention patterns, etc.

---

## 6. Comparison of Techniques

| Technique | Data Needed | GPU Memory | Training Time | Quality | Use Case |
|-----------|-------------|------------|---------------|---------|----------|
| **Full SFT** | Conversations | High | Medium | High | General instruction following |
| **LoRA** | Conversations | Low | Fast | Medium | Task-specific, memory-constrained |
| **DPO** | Preferences | High | Medium | High | Alignment, quality improvement |
| **PPO/GRPO** | Prompts + Reward | Very High | Slow | High | Advanced alignment |
| **Distillation** | Teacher model | Medium | Medium | Medium | Knowledge transfer |

---

## 7. Training Pipeline with Advanced Techniques

```
Base Model (64M)
    ↓
[Pretrain] → Language capability
    ↓
[Full SFT] → Instruction following
    ↓
    ├──→ [DPO] → Better responses (optional)
    ├──→ [LoRA] → Task specialization (optional)
    ├──→ [PPO/GRPO] → Further alignment (optional)
    └──→ [Distillation] → Transfer from larger model (optional)
```

---

## 8. Tips for Your RTX 4070

### Memory-efficient training:

```bash
# Use LoRA for task-specific training
python train_lora.py --lora_rank 8 --batch_size 8

# Use gradient checkpointing for DPO
python train_dpo.py --batch_size 4 --accumulation_steps 4
```

### Expected training times (per epoch, mini dataset):

| Method | RTX 4070 Laptop |
|--------|-----------------|
| LoRA | ~15-20 min |
| DPO | ~30-40 min |
| PPO | ~60-90 min |

---

## Key Takeaways

1. **LoRA**: 99% fewer parameters, task-specific training
2. **DPO**: Direct preference optimization, simpler than RLHF
3. **RLAIF (PPO/GRPO)**: Reinforcement learning with AI feedback
4. **Agentic RL**: Training for tool use
5. **Choose based on**: Compute budget, data availability, quality requirements

---

## Next Step

Now let's learn how to use the trained models for inference!

**Next**: `tutorials/06_inference.md`
