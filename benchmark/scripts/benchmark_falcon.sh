
python benchmark/perplexity.py --model_name_or_path tiiuae/falcon-7b --experiment attention_sinks --output_dir benchmark/outputs_falcon_7b
python benchmark/perplexity.py --model_name_or_path tiiuae/falcon-7b --experiment transformers --output_dir benchmark/outputs_falcon_7b  --num_tokens 5000
python benchmark/perplexity.py --model_name_or_path tiiuae/falcon-7b --experiment windowed --output_dir benchmark/outputs_falcon_7b

python benchmark/plot_perplexity.py --features perplexity vram --title "Log perplexity & VRAM usage of Falcon-7B as a function of input lengths" --output_dir benchmark/outputs_falcon_7b --log_perplexity_limit 4
