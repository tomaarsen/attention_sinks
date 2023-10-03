
python benchmark/perplexity.py --experiment attention_sinks --output_dir benchmark/outputs_llama_2_7b
python benchmark/perplexity.py --experiment transformers --output_dir benchmark/outputs_llama_2_7b --num_tokens 7000
python benchmark/perplexity.py --experiment windowed --output_dir benchmark/outputs_llama_2_7b --num_tokens 2000

python benchmark/plot_perplexity.py --features perplexity vram --title "Log perplexity & VRAM usage of Llama 2 7B as a function of input lengths" --output_dir benchmark/outputs_llama_2_7b --log_perplexity_limit 4
