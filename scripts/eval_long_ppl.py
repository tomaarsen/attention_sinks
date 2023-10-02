"""
Adapted from https://github.com/mit-han-lab/streaming-llm

This is a very rudimentary, ugly script. Sorry about that. Feel free to modify TEST to one of the three options.
Run all three and you'll be able to run `plot_perplexity.py` to get the figure from the README.
"""

TEST = "attention_sinks"
# TEST = "transformers"
# TEST = "window_attention"

assert TEST in ("attention_sinks", "transformers", "window_attention")

import os

import torch

if TEST in ("attention_sinks", "window_attention"):
    from attention_sinks import AutoModelForCausalLM, AutoTokenizer
else:
    from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

data = load_dataset("emozilla/pg19-test", split="test")

model_id = "meta-llama/Llama-2-7b-hf"
# model_id = "PY007/TinyLlama-1.1B-intermediate-step-480k-1T"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,  # <- or just float16 if you can't use bf16
    attention_sink_size=0,
    attention_sink_window_size=1024,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.eval()

# Just one book (which is about 65k tokens long)
num_test_samples = 1
min_test_seq_length = 100
max_test_seq_length = 8192

f = open(f"outputs/{TEST}_gpu_usage.txt", "w")

os.makedirs("outputs", exist_ok=True)
with open(f"outputs/{TEST}.txt", "w") as log_file:
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
            print(torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024, file=f, flush=True)
            num_eval_tokens += 1
            if num_eval_tokens >= max_test_seq_length:
                break

ppl = torch.exp(torch.stack(nlls).mean())
print(ppl.item())
with open(f"outputs/{TEST}_ppl.txt", "w") as f:
    f.write(f"{ppl.item()}\n")
