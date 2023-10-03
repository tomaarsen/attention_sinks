
python benchmark/perplexity.py --model_name_or_path EleutherAI/pythia-6.9b-deduped --experiment attention_sinks --output_dir benchmark/outputs_pythia_6.9b
python benchmark/perplexity.py --model_name_or_path EleutherAI/pythia-6.9b-deduped --experiment transformers --output_dir benchmark/outputs_pythia_6.9b
python benchmark/perplexity.py --model_name_or_path EleutherAI/pythia-6.9b-deduped --experiment windowed --output_dir benchmark/outputs_pythia_6.9b

python benchmark/plot_perplexity.py --features perplexity vram --title "Log perplexity & VRAM usage of Pythia 6.9B as a function of input lengths" --output_dir benchmark/outputs_pythia_6.9b --log_perplexity_limit 4
