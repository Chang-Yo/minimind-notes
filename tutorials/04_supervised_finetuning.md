# Tutorial 4: Supervised Fine-Tuning (SFT)

This tutorial explains how to teach a pretrained model to follow instructions through Supervised Fine-Tuning.

## Learning Objectives

After this tutorial, you will understand:
1. Why SFT is needed after pretraining
2. How SFT differs from pretraining
3. Chat templates and instruction format
4. Label masking for training only on responses
5. How to run SFT training

## File: `trainer/train_full_sft.py`

---

## 1. What is SFT?

**Supervised Fine-Tuning (SFT)** is the process of training a pretrained model on instruction-response pairs to teach it to follow instructions.

**Pretrain vs SFT**:

| Aspect | Pretraining | SFT |
|--------|-------------|-----|
| **Goal** | Learn language | Learn to follow instructions |
| **Data** | Raw text | Conversations |
| **Format** | Continuous text | User/Assistant turns |
| **Learning** | Predict next token | Predict response given instruction |
| **Output** | Continues text | Answers questions |

**Example**:
```
Pretrain: "The capital of France is Paris, which is..."
         → Predict: "known for the Eiffel Tower..."

SFT:      User: "What is the capital of France?"
          Assistant: "Paris"
          → Learn: When asked this question, answer "Paris"
```

---

## 2. SFT Training Loop

The SFT training loop is almost identical to pretraining, with key differences in:
1. **Data format** (conversations vs raw text)
2. **Label masking** (only train on assistant responses)
3. **Learning rate** (much lower: 1e-5 vs 5e-4)

```python
# Same structure as pretraining
for step, (input_ids, labels) in enumerate(loader):
    # Forward pass
    res = model(input_ids, labels=labels)
    loss = res.loss + res.aux_loss

    # Backward pass
    scaler.scale(loss).backward()

    # Update weights
    if step % args.accumulation_steps == 0:
        scaler.step(optimizer)
        optimizer.zero_grad()
```

**The magic is in the data preparation!**

---

## 3. Key Difference: Label Masking

### Pretrain Labels
```python
# All tokens are trained
input_ids = [BOS, The, capital, is, Paris, EOS, PAD, PAD]
labels    = [The, capital, is, Paris, EOS, -100, -100, -100]
#           ↑────────────────────────────↑   ↑─────↑
#           Train on all real tokens        Ignore padding
```

### SFT Labels
```python
# Only assistant responses are trained
input_ids = [BOS, user, Hello, EOS, BOS, assistant, Hi, there, !, EOS]
labels    = [-100, -100, -100, -100, -100, Hi, there, !, EOS, -100]
#           └────────────────┘ └──────────────────────┘
#           Ignore (user)       Train on this only!
```

**Why mask user input?**
- Model should learn to RESPOND, not REPEAT
- User input provides context
- Assistant output is what to generate

---

## 4. Chat Template System

MiniMind uses a chat template to format conversations:

```python
# Input: List of messages
messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
]

# Template format
formatted = tokenizer.apply_chat_template(messages, tokenize=False)
```

**Result**:
```
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

**Template components**:
- `<|im_start|>{role}`: Start of message
- `{content}`: Message content
- `<|im_end|>`: End of message
- Special tokens: BOS, EOS for training

---

## 5. SFT Dataset Implementation

```python
class SFTDataset(Dataset):
    def __getitem__(self, index):
        sample = self.samples[index]

        # 1. Apply chat template
        conversations = pre_processing_chat(sample['conversations'])
        prompt = self.create_chat_prompt(conversations)

        # 2. Tokenize
        input_ids = self.tokenizer(prompt).input_ids[:self.max_seq_len]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # 3. Generate labels (mask non-assistant tokens)
        labels = self.generate_labels(input_ids)

        return torch.tensor(input_ids), torch.tensor(labels)
```

### Label Generation Logic

```python
def generate_labels(self, input_ids):
    labels = [-100] * len(input_ids)  # Default: ignore all

    i = 0
    while i < len(input_ids):
        # Find "assistant\n" marker (BOS for assistant)
        if input_ids[i:i + len(self.bos_id)] == self.bos_id:
            start = i + len(self.bos_id)

            # Find EOS token
            end = start
            while end < len(input_ids):
                if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                    break
                end += 1

            # Only compute loss on assistant response
            for j in range(start, min(end + len(self.eos_id), self.max_length)):
                labels[j] = input_ids[j]

            i = end + len(self.eos_id)
        else:
            i += 1

    return labels
```

**Visual breakdown**:
```
Position: 0    1    2    3    4    5    6    7    8    9    10
Token:   <BOS>user Hello<EOS><BOS>assistant Hi !<EOS>
Label:   -100 -100 -100 -100 -100   Hi   !  -100
                              └────────┘
                              Train here only
```

---

## 6. Hyperparameter Differences

| Parameter | Pretrain | SFT | Why? |
|-----------|----------|-----|------|
| Learning rate | 5e-4 | 1e-5 | Model already knows language, just needs to learn format |
| Batch size | 32 | 16 | SFT data is longer (conversations) |
| Max seq len | 340 | 768 | Conversations need more tokens |
| Accumulation | 8 | 1 | SFT converges with smaller effective batch |
| Epochs | 2 | 2-5 | SFT may need more passes to learn format |

---

## 7. Running SFT Training

### Prerequisites

1. **Complete pretraining first**:
   ```bash
   # You should have: out/pretrain_768.pth
   ls -lh ../out/pretrain_768.pth
   ```

2. **Download SFT data**:
   ```bash
   # Download sft_t2t_mini.jsonl to ./dataset/
   ```

### Run Training

```bash
cd trainer
python train_full_sft.py
```

**Command breakdown**:
```bash
python train_full_sft.py \
    --from_weight pretrain \    # Load pretrained weights
    --save_weight full_sft \     # Save as full_sft
    --data_path ../dataset/sft_t2t_mini.jsonl \
    --epochs 2 \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --max_seq_len 768
```

### Monitor Training

```
Epoch:[1/2](100/1000), loss: 2.1234, logits_loss: 2.1234, aux_loss: 0.0000, lr: 0.00000900
Epoch:[1/2](200/1000), loss: 1.8921, logits_loss: 1.8921, aux_loss: 0.0000, lr: 0.00000850
...
```

**Expected loss curve**:
- Starts higher than pretrain (~2-3)
- Decreases more slowly (lower learning rate)
- Converges to ~1-2

### Check Output

```bash
ls -lh ../out/
# pretrain_768.pth    <- Pretrained model
# full_sft_768.pth    <- SFT model (instruction following!)
```

---

## 8. Testing the Trained Model

After SFT, test if the model follows instructions:

```bash
python eval_llm.py --weight full_sft
```

**Before SFT (pretrain only)**:
```
User: What is the capital of France?
Model: The capital of France is Paris, which is located in...
       (Just continues text, doesn't answer)
```

**After SFT**:
```
User: What is the capital of France?
Model: The capital of France is Paris.
       (Actually answers the question!)
```

---

## 9. System Prompt Augmentation

MiniMind randomly adds system prompts during training:

```python
def pre_processing_chat(conversations, add_system_ratio=0.2):
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "You are a helpful AI assistant.",
        # ... more prompts
    ]

    # 20% chance to add system prompt
    if random.random() < add_system_ratio:
        return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations

    return conversations
```

**Why?**
- Teaches model to handle system prompts
- Adds diversity to training data
- Improves instruction following

---

## 10. Tool Calling in SFT

MiniMind SFT data includes tool calling examples:

```jsonl
{
  "conversations": [
    {"role": "system", "content": "# Tools...", "tools": "[{...}]"},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "", "tool_calls": "[{...}]"},
    {"role": "tool", "content": "4"},
    {"role": "assistant", "content": "The answer is 4."}
  ]
}
```

**Template format**:
```
<|im_start|>system
# Tools: calculator(...)
<|im_end|>
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
<|tool_call|>calculator(expr="2+2")<|tool_call_end|><|im_end|>
<|im_start|>tool
4<|im_end|>
<|im_start|>assistant
The answer is 4.<|im_end|>
```

The model learns to:
1. Recognize when to use tools
2. Format tool calls correctly
3. Process tool responses
4. Generate final answers

---

## 11. Tips for Your RTX 4070

### Memory optimization:
```bash
python train_full_sft.py \
    --batch_size 8 \           # Reduce if OOM
    --max_seq_len 512 \         # Reduce if OOM
    --accumulation_steps 2 \    # Compensate for smaller batch
    --num_workers 4             # Reduce CPU usage
```

### Expected training time (mini dataset):
| GPU | Batch Size | Time per epoch |
|-----|------------|----------------|
| RTX 4090 | 16 | ~20 min |
| RTX 4070 Laptop | 8 | ~40-60 min |
| RTX 4060 Laptop | 4 | ~80-120 min |

---

## 12. Complete Training Pipeline

```bash
# Step 1: Pretraining
cd trainer
python train_pretrain.py
# Output: out/pretrain_768.pth

# Step 2: SFT
python train_full_sft.py --from_weight pretrain
# Output: out/full_sft_768.pth

# Step 3: Test
cd ..
python eval_llm.py --weight full_sft
```

---

## Exercise: Experiment with SFT

1. **Train without pretraining**:
   ```bash
   python train_full_sft.py --from_weight none
   ```
   Compare quality with vs without pretraining.

2. **Different learning rates**:
   ```bash
   python train_full_sft.py --learning_rate 5e-5
   ```

3. **More epochs**:
   ```bash
   python train_full_sft.py --epochs 5
   ```

---

## Key Takeaways

1. **SFT teaches instruction following**: Not just language modeling
2. **Label masking**: Only train on assistant responses
3. **Chat templates**: Format conversations with special tokens
4. **Lower learning rate**: Model already knows language (1e-5 vs 5e-4)
5. **Tool calling**: Can be learned during SFT
6. **Pretraining → SFT**: The standard LLM training pipeline

---

## Next Step

Now you have a trained chat model! Let's look at advanced techniques like DPO, LoRA, and RLHF.

**Next**: `tutorials/05_advanced_techniques.md`
