"""
Adapted from https://github.com/mit-han-lab/streaming-llm

Note: Although this script measures latency, it is not optimized whatsoever!
The latency is only tracked to see the impact of speed over time.

Usage:

python benchmark/perplexity.py --experiment attention_sinks
python benchmark/perplexity.py --experiment transformers
python benchmark/perplexity.py --experiment windowed
"""


import argparse
import itertools
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoTokenizer


def compute_perplexity(
    model,
    tokenizer,
    dataset,
    experiment: str,
    output_dir: str = "outputs",
    data_column: str = "text",
    num_samples: int = 1,
    num_tokens: Optional[int] = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{experiment}.csv"

    if output_file.exists():
        raise ValueError(
            f"The {output_file!r} output file already exists - if you really want to override it, then delete it manually first."
        )

    logs = defaultdict(list)
    loss_fn = CrossEntropyLoss(reduction="none")
    past_key_values = None
    num_processed_tokens = 0
    for text in itertools.islice(dataset, num_samples):
        encodings = tokenizer(text[data_column], return_tensors="pt")

        seq_len = encodings.input_ids.size(1)
        print(f"sequence length: {seq_len}")
        pbar = tqdm(range(0, seq_len - 1))

        for idx in pbar:
            start_t = time.time()
            input_ids = encodings.input_ids[:, idx : idx + 1].to(model.device)
            with torch.no_grad():
                outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
                logits = outputs.logits.view(-1, model.config.vocab_size)
                past_key_values = outputs.past_key_values
                label = encodings.input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
                neg_log_likelihood = loss_fn(logits, label)
                perplexity = neg_log_likelihood.exp()
            pbar.set_description(f"nll: {neg_log_likelihood.item():>5.2f}, ppl: {perplexity.item():>8.2f}")

            # Store data and save every 10 tokens
            logs["input_length"].append(idx + 1)
            logs["nll"].append(neg_log_likelihood.item())
            logs["ppl"].append(perplexity.item())
            logs["overall_ppl"].append(torch.tensor(logs["nll"]).mean().exp().item())
            logs["cuda_vram_allocated"].append(torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024)  # in GB
            logs["latency"].append(time.time() - start_t)
            if num_processed_tokens % 10 == 0:
                try:
                    pd.DataFrame(logs).to_csv(output_file, index=False)
                except KeyboardInterrupt as ex:
                    # If there's a Keyboard Interrupt, still write the file, and then stop
                    pd.DataFrame(logs).to_csv(output_file, index=False)
                    raise ex

            num_processed_tokens += 1
            if num_tokens and num_processed_tokens >= num_tokens:
                return


def main():
    parser = argparse.ArgumentParser()
    # Which experiment to run?
    parser.add_argument(
        "--experiment", choices=["attention_sinks", "transformers", "windowed"], default="attention_sinks"
    )

    # Model args
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--trust_remote_code", action="store_true")

    # Dataset args
    parser.add_argument("--dataset_name", type=str, default="emozilla/pg19-test")
    parser.add_argument("--data_column", type=str, default="text")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["validation", "test"])
    # parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--num_tokens", type=int, default=8192)

    # Where to log
    parser.add_argument("--output_dir", type=str, default="benchmark/outputs")

    # Window size for windowed and attention_sinks
    parser.add_argument("--window_size", type=int, default=1024)

    # Attention Sinks-only settings
    # Attention Sink window size is calculated with args.window_size - args.attention_sink_size
    parser.add_argument("--attention_sink_size", type=int, default=4)

    args = parser.parse_args()

    # Initialize the model, either via transformers or via attention_sinks
    if args.experiment == "transformers":
        from transformers import AutoModelForCausalLM
    else:
        from attention_sinks import AutoModelForCausalLM
    kwargs = {}
    if args.experiment == "attention_sinks":
        kwargs = {
            "attention_sink_size": args.attention_sink_size,
            "attention_sink_window_size": args.window_size - args.attention_sink_size,  # default: 1020
        }
    elif args.experiment == "windowed":
        kwargs = {
            "attention_sink_size": 0,
            "attention_sink_window_size": args.window_size,
        }
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        revision=args.revision,
        trust_remote_code=bool(args.trust_remote_code),
        torch_dtype=torch.float16,
        device_map="auto",
        **kwargs,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=bool(args.trust_remote_code))

    # Set up the dataset
    dataset = load_dataset(args.dataset_name, args.task, split=args.split, streaming=True)

    compute_perplexity(
        model,
        tokenizer,
        dataset,
        args.experiment,
        output_dir=args.output_dir,
        data_column=args.data_column,
        num_samples=1,  # <- No support for more than one instance now
        num_tokens=args.num_tokens,
    )


if __name__ == "__main__":
    main()
