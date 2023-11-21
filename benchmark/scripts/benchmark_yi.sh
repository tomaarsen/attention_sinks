
python benchmark/perplexity.py --model_name_or_path 01-ai/Yi-6B --experiment attention_sinks --output_dir benchmark/outputs_yi_6b --trust_remote_code
python benchmark/perplexity.py --model_name_or_path 01-ai/Yi-6B --experiment transformers --output_dir benchmark/outputs_yi_6b --trust_remote_code
python benchmark/perplexity.py --model_name_or_path 01-ai/Yi-6B --experiment windowed --output_dir benchmark/outputs_yi_6b --trust_remote_code

python benchmark/plot_perplexity.py --features perplexity vram --title "Log perplexity & VRAM usage of Yi-6B as a function of input lengths" --output_dir benchmark/outputs_yi_6b --log_perplexity_limit 4
