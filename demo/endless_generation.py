import argparse
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from utils import FileStreamer


@torch.no_grad()
def endless_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    text: str,
    log_file: str,
    min_new_tokens: int = 1000,
    max_new_tokens: int = 10000,
) -> str:
    streamer = FileStreamer(tokenizer, log_file)
    input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
    generated_tokens = model.generate(
        input_ids,
        generation_config=GenerationConfig(
            # use_cache=True is required, the rest can be changed up.
            use_cache=True,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            penalty_alpha=0.6,
            top_k=4,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        ),
        streamer=streamer,
    )
    # Decode the final generated text
    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)


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

    # Input text
    parser.add_argument("--text", type=str, default="Vaswani et al. (2017) introduced the Transformers")

    # Generation args
    parser.add_argument("--min_new_tokens", type=int, default=1000)
    parser.add_argument("--max_new_tokens", type=int, default=10000)

    # Where to log
    parser.add_argument("--log_file", type=str, default=None)

    # Window size for windowed and attention_sinks
    parser.add_argument("--window_size", type=int, default=512)

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
    tokenizer.pad_token_id = tokenizer.eos_token_id

    log_file = args.log_file or Path("demo") / "endless_logs" / args.experiment / f"{args.model_name_or_path}.txt"
    endless_generate(
        model,
        tokenizer,
        args.text,
        log_file=log_file,
        min_new_tokens=args.min_new_tokens,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
