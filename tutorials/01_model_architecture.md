# Tutorial 1: Model Architecture

This tutorial explains the MiniMind model architecture - a Transformer Decoder-Only LLM optimized for small scale training.

## Learning Objectives

After this tutorial, you will understand:
1. The overall Transformer architecture
2. Self-attention mechanism with RoPE
3. Feed-forward networks with SwiGLU
4. Layer normalization (RMSNorm)
5. MoE (Mixture of Experts) - optional

## File: `model/model_minimind.py`

Let's break down the model step by step.

---

## 1. Model Configuration

```python
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"
    def __init__(self, hidden_size=768, num_hidden_layers=8, ...):
```

**Key parameters explained**:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `hidden_size` | 768 | Dimension of embeddings and hidden states |
| `num_hidden_layers` | 8 | Number of transformer blocks |
| `num_attention_heads` | 8 | Number of attention heads |
| `num_key_value_heads` | 4 | Number of KV heads (for GQA - Grouped Query Attention) |
| `vocab_size` | 6400 | Size of the vocabulary |
| `max_position_embeddings` | 32768 | Maximum sequence length |
| `rope_theta` | 1e6 | RoPE base frequency |
| `use_moe` | False | Enable Mixture of Experts |

**Why these values?**
- `hidden_size=768`: Small enough for fast training, large enough for good capacity
- `num_hidden_layers=8`: Shallow network trains quickly while maintaining depth
- `q_heads=8, kv_heads=4`: Grouped Query Attention (GQA) reduces memory while maintaining quality
- `vocab_size=6400`: Tiny vocabulary keeps embedding layers small

---

## 2. RMSNorm - Layer Normalization

```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)
```

**What it does**: Normalizes activations to have zero mean and unit variance.

**Why RMSNorm instead of LayerNorm?**
- Simpler and faster (no mean centering)
- Works just as well for transformers
- Formula: `output = x / sqrt(mean(x^2) + eps) * weight`

**Where it's used**: Before attention and before feed-forward (Pre-Norm architecture)

---

## 3. RoPE - Rotary Position Embedding

```python
def precompute_freqs_cis(dim: int, end: int = int(32 * 1024),
                         rope_base: float = 1e6, rope_scaling: dict = None):
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # ... (YaRN scaling logic)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin
```

**What it does**: Encodes position information by rotating tokens in the embedding space.

**Why RoPE?**
- Absolute positions encoded relatively
- Better extrapolation to longer sequences
- No positional embedding parameters to learn
- Compatible with KV cache and relative position attention

**How it works**:
```
For position m and dimension d:
freq(m, d) = m / (base ^ (2d/d_model))
Apply rotation: [q*cos + q_rot*sin]
```

---

## 4. Attention Module

```python
class Attention(nn.Module):
    def __init__(self, config: MiniMindConfig):
        # Q, K, V projections
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        # QK normalization (GQA style)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
```

**Key components**:

1. **QKV Projections**: Transform input to Query, Key, Value vectors
2. **QK Normalization**: Normalize queries and keys before attention
3. **RoPE Application**: Apply rotary position embeddings
4. **Multi-Head Attention**: Split into multiple heads for parallel processing
5. **KV Cache**: Store past K, V for efficient generation

**GQA (Grouped Query Attention)**:
```
n_rep = n_local_heads // n_local_kv_heads  # 8 // 4 = 2
Each KV head is shared by 2 Q heads
Reduces memory during generation
```

**Flash Attention**:
```python
if self.flash and seq_len > 1 and past_key_value is None:
    output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
```
Uses optimized CUDA kernel when available.

---

## 5. Feed-Forward Network

```python
class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]  # SiLU

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

**SwiGLU Activation**:
```
output = down(SiLU(gate(x)) * up(x))
```

**Why SwiGLU?**
- Better performance than ReLU/GELU for LLMs
- Gating mechanism allows selective information flow
- Standard in modern LLMs (Llama, Qwen, etc.)

---

## 6. Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, config: MiniMindConfig):
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, ...):
        # Pre-Norm: Normalize before the operation
        h = x + self.attention(self.attention_norm(x), ...)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
```

**Pre-Norm Architecture**:
```
x = x + Attention(Norm(x))
x = x + FFN(Norm(x))
```

**Why Pre-Norm?**
- More stable training for deep networks
- Better gradient flow
- Standard in modern transformers

---

## 7. MoE (Mixture of Experts) - Optional

```python
class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(config) for _ in range(config.num_experts)])
```

**What it does**: Routes each token to a subset of experts instead of all experts.

**Benefits**:
- More capacity without proportional compute increase
- 198M total params but only 64M active during inference

**Trade-offs**:
- Slower training (routing overhead)
- More complex implementation
- Load balancing loss needed

---

## 8. The Full Model

```python
class MiniMindForCausalLM(PreTrainedModel):
    def __init__(self, config):
        self.model = MiniMindModel(config)  # Transformer blocks
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, labels=None):
        # Embed tokens
        hidden_states = self.model.embed_tokens(input_ids)

        # Apply transformer blocks
        hidden_states = self.model.layers(hidden_states, ...)

        # Project to vocabulary
        logits = self.lm_head(hidden_states)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(...)
```

**Causal LM Training**:
- Predict next token for each position
- Loss = cross_entropy(predicted, actual_next_token)
- Mask future tokens (can't see future)

---

## Exercise: Inspect the Model

Run this code to explore the model:

```python
import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

# Create model
config = MiniMindConfig(hidden_size=768, num_hidden_layers=8)
model = MiniMindForCausalLM(config)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params / 1e6:.2f}M")

# Print model structure
print(model)

# Test forward pass
batch_size, seq_len = 2, 64
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
logits = model(input_ids)
print(f"Output shape: {logits.logits.shape}")  # Should be (2, 64, 6400)
```

---

## Key Takeaways

1. **Transformer Decoder-Only**: No encoder, causal masking
2. **RoPE**: Rotary position encoding for better length generalization
3. **SwiGLU**: Gated activation for better performance
4. **Pre-Norm**: Normalize before attention/FFN for stability
5. **GQA**: Grouped Query Attention for memory efficiency
6. **MoE (optional)**: Sparse activation for more capacity

---

## Next Step

Now that you understand the model architecture, let's look at how data is prepared for training.

**Next**: `tutorials/02_data_and_tokenization.md`
