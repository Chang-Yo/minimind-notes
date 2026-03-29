# Tutorial 6: Inference and Deployment

This tutorial explains how to use your trained MiniMind model for text generation and serving.

## Learning Objectives

After this tutorial, you will understand:
1. Text generation strategies
2. Using the CLI inference script
3. Running an OpenAI-compatible API server
4. Generation parameters (temperature, top-p, etc.)
5. KV cache and streaming

---

## 1. Text Generation Basics

### Autoregressive Generation

LLMs generate text token by token:

```
Input: "The capital of France is"
Step 1: Model predicts "Paris"
Step 2: Input becomes "The capital of France is Paris"
Step 3: Model predicts "."
...continues until EOS token or max length
```

### Sampling vs Greedy

```python
# Greedy (deterministic)
token = argmax(probabilities)

# Sampling (stochastic, more creative)
token = sample(probabilities, temperature=0.8)
```

**Temperature** controls randomness:
- Low (0.1): Very focused, deterministic
- High (1.0+): More random, creative

**Top-p (nucleus sampling)**:
- Keep tokens that sum to probability p
- Example: p=0.95 means keep 95% probability mass

---

## 2. CLI Inference

### Basic Usage

```bash
# Using PyTorch weights from ./out/
python eval_llm.py --weight full_sft

# Using transformers format model
python eval_llm.py --load_from ./minimind-3
```

### Interactive Mode

```bash
python eval_llm.py --weight full_sft
# Select: [1] 手动输入
# Then type your messages!
```

### Auto Test Mode

```bash
python eval_llm.py --weight full_sft
# Select: [0] 自动测试
# Runs through predefined test prompts
```

### Key Parameters

```bash
python eval_llm.py \
    --weight full_sft \
    --max_new_tokens 512 \      # Max tokens to generate
    --temperature 0.7 \          # Lower = more focused
    --top_p 0.95 \               # Nucleus sampling threshold
    --historys 4                 # Number of conversation turns to remember
```

---

## 3. Generation Parameters

### Temperature

```python
--temperature 0.1   # Very focused, deterministic
--temperature 0.7   # Balanced (recommended)
--temperature 1.0   # More creative
--temperature 1.5+  # Very random, may be incoherent
```

**Effect**:
```
Temperature 0.1: "The capital of France is Paris."
Temperature 0.7: "The capital of France is Paris, known for the Eiffel Tower."
Temperature 1.5: "The capital of France might be Paris, or maybe somewhere else..."
```

### Top-p (Nucleus Sampling)

```python
--top_p 0.9    # Keep top 90% probability mass
--top_p 0.95   # Keep top 95% probability mass (default)
--top_p 1.0    # No filtering (pure sampling)
```

**Combined with temperature**:
```python
# Sample from distribution
probs = softmax(logits / temperature)

# Filter by top-p
sorted_probs = sort(probs, descending)
cumsum = cumulative_sum(sorted_probs)
cutoff = where(cumsum > top_p)
filtered_probs = sorted_probs[:cutoff]

# Normalize and sample
token = sample(normalize(filtered_probs))
```

### Repetition Penalty

Prevents the model from repeating itself:

```python
# In generate()
repetition_penalty=1.0   # No penalty
repetition_penalty=1.1   # Mild penalty
repetition_penalty=1.2   # Strong penalty
```

### Max New Tokens

```python
--max_new_tokens 128    # Short responses
--max_new_tokens 512    # Medium responses (default)
--max_new_tokens 2048   # Long responses
```

**Note**: This is the generation limit, not the model's actual context window.

---

## 4. Conversation History

### Multi-turn Conversations

```python
--historys 4    # Remember last 2 user + 2 assistant turns
--historys 0    # No memory (default)
```

**How it works**:

```python
# Without history
input = "What's the capital?"
# Model doesn't know previous context

# With history
conversation = [
    {"role": "user", "content": "I'm learning about countries."},
    {"role": "assistant", "content": "That's interesting!"},
    {"role": "user", "content": "What's the capital?"}
]
# Model can use context from previous turns
```

### Conversation Management in Code

```python
conversation = []

# Turn 1
conversation.append({"role": "user", "content": "Hello"})
response = generate(conversation)
conversation.append({"role": "assistant", "content": response})

# Turn 2
conversation.append({"role": "user", "content": "How are you?"})
response = generate(conversation)
# Model remembers "Hello" was said
```

---

## 5. OpenAI-Compatible API Server

### Starting the Server

```bash
cd scripts
python serve_openai_api.py \
    --load_from ../out \
    --weight full_sft \
    --port 8000
```

### API Endpoints

#### Chat Completion

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minimind",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

#### Streaming Response

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minimind",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

#### Tool Calling Support

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minimind",
    "messages": [
      {"role": "system", "content": "# Tools...", "tools": "[{...}]"},
      {"role": "user", "content": "What is 2+2?"}
    ]
  }'
```

### Response Format

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "minimind",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hello! How can I help you today?"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  }
}
```

---

## 6. Web UI Demo

### Starting Streamlit Demo

```bash
# Copy model to scripts directory first
cp -r minimind-3 scripts/

cd scripts
streamlit run web_demo.py
```

**Features**:
- Interactive chat interface
- Thinking/reasoning display
- Tool calling visualization
- Multi-turn conversation support

---

## 7. KV Cache for Efficient Generation

### What is KV Cache?

During generation, keys and values from previous tokens are cached:

```python
# Without KV cache (slow)
for i in range(max_tokens):
    # Recompute attention for all tokens each time
    output = model(input_ids + all_generated_tokens)

# With KV cache (fast)
past_kv = None
for i in range(max_tokens):
    # Only compute for new token
    output, past_kv = model(new_token, past_kv=past_kv)
```

### Speed Improvement

```
Without KV cache: O(n²) per token
With KV cache:    O(n) per token (after first)

Speedup: ~2-3x for long sequences
```

### In MiniMind

KV cache is enabled by default in `model.generate()`:

```python
# In model_minimind.py
def forward(self, x, past_key_value=None, use_cache=True):
    # ... attention computation
    past_kv = (xk, xv) if use_cache else None
    return output, past_kv
```

---

## 8. RoPE Scaling for Long Context

MiniMind supports YaRN (Yet another RoPE extensioN) for longer sequences:

```bash
python eval_llm.py \
    --weight full_sft \
    --inference_rope_scaling \    # Enable 4x RoPE scaling
    --max_new_tokens 4096
```

**What it does**:
- Extends context window beyond training length
- Training: 2048 tokens → Inference: 8192 tokens (4×)
- May degrade quality for very long sequences

---

## 9. Performance Tips

### Faster Inference

1. **Use bfloat16/float16**:
   ```python
   model = model.half()  # or bfloat16
   ```

2. **Batch requests** (API server):
   ```python
   # Process multiple requests together
   responses = model.generate(batch_inputs)
   ```

3. **Reduce max tokens**:
   ```bash
   --max_new_tokens 256  # Instead of 8192
   ```

4. **Use compiled model**:
   ```bash
   python train_xxx.py --use_compile 1
   ```

### Expected Speed (RTX 4070 Laptop)

| Model | Quantization | Tokens/sec |
|-------|--------------|------------|
| FP16 | None | ~20-30 |
| FP16 | KV cache | ~40-60 |
| INT8 | None | ~40-50 |
| INT8 | KV cache | ~70-100 |

---

## 10. Complete Inference Example

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model
tokenizer = AutoTokenizer.from_pretrained("./minimind-3")
model = AutoModelForCausalLM.from_pretrained(
    "./minimind-3",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Prepare input
messages = [
    {"role": "user", "content": "Explain quantum computing"}
]
input_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

# Generate
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )

# Decode response
response = tokenizer.decode(
    outputs[0][len(inputs.input_ids[0]):],
    skip_special_tokens=True
)
print(response)
```

---

## Key Takeaways

1. **Temperature**: Controls randomness (0.1-1.5)
2. **Top-p**: Nucleus sampling (0.9-1.0)
3. **KV cache**: Critical for fast generation
4. **History**: Enable for multi-turn conversations
5. **RoPE scaling**: Extend context length (with quality trade-off)
6. **API server**: OpenAI-compatible deployment

---

## Congratulations! 🎉

You've completed the MiniMind learning path! You now understand:

✅ Model architecture (Transformer, RoPE, SwiGLU)
✅ Data and tokenization
✅ Pretraining (next-token prediction)
✅ SFT (instruction following)
✅ Advanced techniques (LoRA, DPO, RLHF)
✅ Inference and deployment

**Next steps**:
- Experiment with different hyperparameters
- Train on your own data
- Try advanced techniques
- Contribute to the project!

---

## Additional Resources

- **Main README**: `README.md` for full documentation
- **Dataset info**: `dataset/dataset.md`
- **Model configs**: `model/model_minimind.py`
- **Training scripts**: `trainer/*.py`
- **GitHub**: https://github.com/jingyaogong/minimind
- **HuggingFace**: https://huggingface.co/jingyaogong
