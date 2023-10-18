python benchmark/perplexity.py --model_name_or_path stabilityai/stablelm-3b-4e1t --experiment attention_sinks --output_dir benchmark/outputs_stablelm_3b_4e1t --trust_remote_code
python benchmark/perplexity.py --model_name_or_path stabilityai/stablelm-3b-4e1t --experiment transformers --output_dir benchmark/outputs_stablelm_3b_4e1t --trust_remote_code
python benchmark/perplexity.py --model_name_or_path stabilityai/stablelm-3b-4e1t --experiment windowed --output_dir benchmark/outputs_stablelm_3b_4e1t --trust_remote_code

python benchmark/plot_perplexity.py --features perplexity vram --title "Log perplexity & VRAM usage of  StableLM-3B-4E1T as a function of input lengths" --output_dir benchmark/outputs_stablelm_3b_4e1t --log_perplexity_limit 4