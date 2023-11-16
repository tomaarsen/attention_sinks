
python benchmark/perplexity.py --model_name_or_path cerebras/btlm-3b-8k-base --experiment attention_sinks --output_dir benchmark/outputs_btlm_3b_8k_base_2k_win --trust_remote_code --num_tokens 16384 --window_size 2048
python benchmark/perplexity.py --model_name_or_path cerebras/btlm-3b-8k-base --experiment transformers --output_dir benchmark/outputs_btlm_3b_8k_base_2k_win --trust_remote_code --num_tokens 16384
python benchmark/perplexity.py --model_name_or_path cerebras/btlm-3b-8k-base --experiment windowed --output_dir benchmark/outputs_btlm_3b_8k_base_2k_win --trust_remote_code --num_tokens 16384 --window_size 2048

python benchmark/plot_perplexity.py --features perplexity vram --title "Log perplexity & VRAM usage of BTLM-3B-8k-base\nas a function of input lengths" --output_dir benchmark/outputs_btlm_3b_8k_base_2k_win --log_perplexity_limit 4
