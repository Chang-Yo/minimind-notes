# Tutorial 2: Data and Tokenization

This tutorial explains how text data is prepared for LLM training - from raw text to model input.

## Learning Objectives

After this tutorial, you will understand:
1. Tokenization - converting text to numbers
2. Chat templates - formatting conversations
3. Data preprocessing - truncation, padding, labeling
4. Different dataset types (Pretrain, SFT, DPO, RLAIF)

## File: `dataset/lm_dataset.py`

---

## 1. What is Tokenization?

**Tokenization** is the process of breaking text into smaller units called tokens, which are then mapped to numerical IDs.

**Example**:
```
Text: "Hello, world!"
Tokens: ["Hello", ",", "world", "!"]
IDs:   [1234,   15,   5678,   89]
```

**Why tokenize?**
- Neural networks operate on numbers, not text
- Reduces vocabulary size (vs character-level)
- Captures subword patterns (vs word-level)

---

## 2. MiniMind's Tokenizer

**Location**: `model/tokenizer.json`

**Specs**:
- Type: BPE (Byte Pair Encoding) + ByteLevel
- Vocabulary size: 6400 tokens
- Special tokens:
  - `<|im_start|>` / `<|im_end|>`: Chat boundaries
  - `<|thinking|>` / `<|</thinking|>`: Reasoning tags
  - Tool calling tokens

**Why such a small vocabulary?**
- Large vocabularies (like GPT-3's 50k) require huge embedding layers
- For a 64M model, 6400 vocab keeps embeddings small (~5M params)
- Trade-off: slightly more tokens per text, but much smaller model

---

## 3. Pretrain Dataset

The simplest dataset format - raw text.

### Data Format (`pretrain_t2t*.jsonl`)

```jsonl
{"text": "如何才能摆脱拖延症？治愈拖延症并不容易，但以下建议可能有所帮助。"}
{"text": "清晨的阳光透过窗帘洒进房间，桌上的书页被风轻轻翻动。"}
```

### Dataset Implementation

```python
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __getitem__(self, index):
        sample = self.samples[index]

        # Tokenize text
        tokens = self.tokenizer(
            str(sample['text']),
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True
        ).input_ids

        # Add BOS/EOS tokens
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]

        # Pad to max_length
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # Labels = input_ids (predict next token)
        labels = input_ids.clone()

        # Don't compute loss on padding
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        return input_ids, labels
```

### Key Concepts

**BOS/EOS Tokens**:
```
<BOS> How are you? <EOS> <PAD> <PAD> ...
```
- BOS (Beginning of Sequence): Marks start
- EOS (End of Sequence): Marks end
- PAD: Padding for fixed-length batches

**Labels = -100**:
```python
labels[input_ids == pad_token_id] = -100
```
In PyTorch, `-100` is ignored by cross-entropy loss. This means:
- Loss is computed on real tokens
- Loss is NOT computed on padding

---

## 4. SFT Dataset

Supervised Fine-Tuning uses conversation format.

### Data Format (`sft_t2t*.jsonl`)

```jsonl
{
  "conversations": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！"},
    {"role": "user", "content": "再见"},
    {"role": "assistant", "content": "再见！"}
  ]
}
```

### Chat Template Application

```python
def create_chat_prompt(self, conversations):
    messages = []
    for message in conversations:
        messages.append(message)

    # Apply chat template (tokenizer converts conversations to text)
    return self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=tools
    )
```

**After template application**:
```
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
你好！<|im_end|>
<|im_start|>user
再见<|im_end|>
<|im_start|>assistant
再见！<|im_end|>
```

### Label Generation (Critical!)

**Only train on assistant responses!**

```python
def generate_labels(self, input_ids):
    labels = [-100] * len(input_ids)  # Default: ignore all

    i = 0
    while i < len(input_ids):
        # Find "assistant\n" marker
        if input_ids[i:i + len(self.bos_id)] == self.bos_id:
            start = i + len(self.bos_id)  # After "assistant\n"

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

**Visual example**:
```
Token:  <BOS>user Hello<EOS><BOS>assistant Hi there!<EOS>
Label:  -100  -100   -100  -100  -100    -100   Hi   there   !   -100
         └────────────────┘ └────────────────────────────────────┘
         Ignore (user)      Train on this (assistant response)
```

**Why?**
- Model should learn to RESPOND, not repeat user input
- User input is context, assistant output is what to learn
- This is "teacher forcing" - train model to predict correct response

---

## 5. Tool Calling Format

MiniMind supports function calling during SFT.

### Data Format

```jsonl
{
  "conversations": [
    {
      "role": "system",
      "content": "# Tools...",
      "tools": "[{\"name\": \"calculator\", \"parameters\": {...}}]"
    },
    {"role": "user", "content": "What is 2+2?"},
    {
      "role": "assistant",
      "content": "",
      "tool_calls": "[{\"name\": \"calculator\", \"arguments\": {\"expr\": \"2+2\"}}]"
    },
    {"role": "tool", "content": "4"},
    {"role": "assistant", "content": "The answer is 4."}
  ]
}
```

### Template Application

The tokenizer converts this to:
```
<|im_start|>system
# Tools: calculator, ...
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

---

## 6. Reasoning/Thinking Tags

MiniMind supports chain-of-thought reasoning.

```python
def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    # Remove empty thinking tags 80% of the time
    if '<|thinking|>\n\n<|/thinking|>' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<|thinking|>\n\n<|/thinking|>', '')
    return prompt_content
```

**Template with thinking**:
```
<|im_start|>user
Solve 2+2<|im_end|>
<|im_start|>assistant
<|thinking|>
I need to add 2 and 2 together.
<|/thinking|>
The answer is 4.<|im_end|>
```

---

## 7. DPO Dataset (Preference Learning)

Direct Preference Optimization uses paired (chosen, rejected) examples.

### Data Format

```jsonl
{
  "chosen": [
    {"content": "What's the capital of France?", "role": "user"},
    {"content": "The capital of France is Paris.", "role": "assistant"}
  ],
  "rejected": [
    {"content": "What's the capital of France?", "role": "user"},
    {"content": "I don't know.", "role": "assistant"}
  ]
}
```

### Dataset Implementation

```python
class DPODataset(Dataset):
    def __getitem__(self, index):
        sample = self.samples[index]

        # Tokenize both chosen and rejected
        chosen_prompt = self.tokenizer.apply_chat_template(sample['chosen'], ...)
        rejected_prompt = self.tokenizer.apply_chat_template(sample['rejected'], ...)

        # Generate loss masks (only on assistant responses)
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }
```

---

## 8. RLAIF Dataset (Reinforcement Learning from AI Feedback)

For RL training, we need prompts (with empty assistant response to be filled).

```python
class RLAIFDataset(Dataset):
    def __getitem__(self, index):
        sample = self.samples[index]

        # Create prompt with empty assistant response
        prompt = self.tokenizer.apply_chat_template(
            conversations[:-1],  # Exclude last (assistant) message
            tokenize=False,
            open_thinking=random.choice([True, False]),
            add_generation_prompt=True
        )

        return {'prompt': prompt, 'answer': ""}
```

**Output example**:
```
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
<|thinking|>
                           <- Model will fill this in
```

---

## Exercise: Explore Tokenization

```python
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("./model")

# Test tokenization
text = "你好，世界！Hello, world!"
tokens = tokenizer.encode(text)
print(f"Text: {text}")
print(f"Tokens: {tokens}")
print(f"Decoded: {tokenizer.decode(tokens)}")

# Test chat template
messages = [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！"}
]
formatted = tokenizer.apply_chat_template(messages, tokenize=False)
print(f"\nFormatted chat:\n{formatted}")

# Check vocabulary size
print(f"\nVocab size: {len(tokenizer)}")
print(f"Special tokens: {tokenizer.special_tokens_map}")
```

---

## Key Takeaways

1. **Tokenization**: Text → Tokens → IDs (using BPE)
2. **Chat templates**: Format conversations with special tokens
3. **Label masking**: Only train on assistant responses in SFT
4. **Padding**: `-100` is ignored by loss function
5. **Multiple dataset types**: Pretrain, SFT, DPO, RLAIF

---

## Next Step

Now let's look at the actual training process!

**Next**: `tutorials/03_pretraining.md`
