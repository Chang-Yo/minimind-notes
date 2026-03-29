import torch 
import sys

sys.path.append('/home/zhixi/minimind')
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

config =MiniMindConfig(hidden_size=768,num_hidden_layers=8)
model = MiniMindForCausalLM(config)
total_params=sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params/1e6:.2f}M")

print(model)

batch_size,seq_len=2,64
input_ids=torch.randint(0,config.vocab_size,(batch_size,seq_len))

logits=model(input_ids)
print(f"Output shape: {logits.logits.shape}")