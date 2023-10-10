
python benchmark/perplexity.py --model_name_or_path EleutherAI/gpt-j-6b --experiment attention_sinks --output_dir benchmark/outputs_gptj_6b
python benchmark/perplexity.py --model_name_or_path EleutherAI/gpt-j-6b --experiment transformers --output_dir benchmark/outputs_gptj_6b
python benchmark/perplexity.py --model_name_or_path EleutherAI/gpt-j-6b --experiment windowed --output_dir benchmark/outputs_gptj_6b

python benchmark/plot_perplexity.py --features perplexity vram --title "Log perplexity & VRAM usage of GPT-J 6B as a function of input lengths" --output_dir benchmark/outputs_gptj_6b --log_perplexity_limit 4
