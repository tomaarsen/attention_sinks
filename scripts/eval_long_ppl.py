"""
Adapted from https://github.com/mit-han-lab/streaming-llm
"""

import torch
from tqdm import tqdm
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
# from attention_sinks import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.nn import CrossEntropyLoss

data = load_dataset("emozilla/pg19-test", split="test")

# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", device_map="auto", torch_dtype=torch.bfloat16)
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("PY007/TinyLlama-1.1B-intermediate-step-480k-1T", device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("PY007/TinyLlama-1.1B-intermediate-step-480k-1T")
model.eval()

num_test_samples = 1
min_test_seq_length = 100
max_test_seq_length = 8192

os.makedirs("outputs", exist_ok=True)
with open(f"outputs/transformers.txt", "w") as log_file:
    nlls = []
    loss_fn = CrossEntropyLoss(reduction="none")
    past_key_values = None
    num_eval_tokens = 0
    for text in data["text"][:num_test_samples]:
        encodings = tokenizer(text, return_tensors="pt")

        seq_len = encodings.input_ids.size(1)
        print(f"seq_len: {seq_len}")
        pbar = tqdm(range(0, seq_len - 1))

        for idx in pbar:
            input_ids = encodings.input_ids[:, idx : idx + 1].to(model.device)
            with torch.no_grad():
                outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
                logits = outputs.logits.view(-1, model.config.vocab_size)
                past_key_values = outputs.past_key_values
                label = encodings.input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
                neg_log_likelihood = loss_fn(logits, label)
            nlls.append(neg_log_likelihood)
            pbar.set_description(
                f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}"
            )
            print(neg_log_likelihood.item(), file=log_file, flush=True)
            num_eval_tokens += 1
            if num_eval_tokens >= max_test_seq_length:
                break

ppl = torch.exp(torch.stack(nlls).mean())
print(ppl.item())
with open(f"outputs/transformers_ppl.txt", "w") as f:
    f.write(f"{ppl.item()}\n")
