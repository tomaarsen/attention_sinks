
python benchmark/perplexity.py --model_name_or_path Qwen/Qwen-7B --experiment attention_sinks --output_dir benchmark/outputs_qwen_7b --trust_remote_code
python benchmark/perplexity.py --model_name_or_path Qwen/Qwen-7B --experiment transformers --output_dir benchmark/outputs_qwen_7b --trust_remote_code
python benchmark/perplexity.py --model_name_or_path Qwen/Qwen-7B --experiment windowed --output_dir benchmark/outputs_qwen_7b --trust_remote_code

python benchmark/plot_perplexity.py --features perplexity vram --title "Log perplexity & VRAM usage of Qwen 7B as a function of input lengths" --output_dir benchmark/outputs_qwen_7b --log_perplexity_limit 4
