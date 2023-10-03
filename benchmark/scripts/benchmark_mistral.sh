
python benchmark/perplexity.py --model_name_or_path mistralai/Mistral-7B-v0.1 --experiment attention_sinks --output_dir benchmark/outputs_mistral_7b --num_tokens 16384
python benchmark/perplexity.py --model_name_or_path mistralai/Mistral-7B-v0.1 --experiment transformers --output_dir benchmark/outputs_mistral_7b --num_tokens 14000
python benchmark/perplexity.py --model_name_or_path mistralai/Mistral-7B-v0.1 --experiment windowed --output_dir benchmark/outputs_mistral_7b

python benchmark/plot_perplexity.py --features perplexity vram --title "Log perplexity & VRAM usage of Mistral 7B as a function of input lengths" --output_dir benchmark/outputs_mistral_7b --log_perplexity_limit 4
