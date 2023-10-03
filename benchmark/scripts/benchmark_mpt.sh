
python benchmark/perplexity.py --model_name_or_path mosaicml/mpt-7b --experiment attention_sinks --output_dir benchmark/outputs_mpt_7b
python benchmark/perplexity.py --model_name_or_path mosaicml/mpt-7b --experiment transformers --output_dir benchmark/outputs_mpt_7b --num_tokens 2048
python benchmark/perplexity.py --model_name_or_path mosaicml/mpt-7b --experiment windowed --output_dir benchmark/outputs_mpt_7b --num_tokens 4000

# NOTE: Running mpt-7b with pure transformers with a sequence length of larger than 2048 requires the following:
# ```python
# config = AutoConfig.from_pretrained(args.model_name_or_path)
# config.max_seq_len = 8192
# model = AutoModelForCausalLM.from_pretrained(
#    model_name_or_path,
#    config=config,
#    ...,
# )
# ```
# This is not required for "attention_sinks" or "windowed". To prevent crashes I'll put `num_tokens` to 2048 for "transformers"


python benchmark/plot_perplexity.py --features perplexity vram --title "Log perplexity & VRAM usage of MPT-7B as a function of input lengths" --output_dir benchmark/outputs_mpt_7b --log_perplexity_limit 4
