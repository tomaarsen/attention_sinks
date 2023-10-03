
# Attention Sinks in Transformers for Infinite-length LLMs

![llama_2_7b_ppl_vram](https://github.com/tomaarsen/attention_sinks/assets/37621491/1b99f29e-8d8d-4677-bef6-6a6e041776f6)

## Overview

* Extend existing LLMs (e.g. Llama 2) to infinite length without sacrificing efficiency and performance, without any retraining.
* The `attention_sinks` API allows for a drop-in replacement of the `transformers` API:
  ```python
  from attention_sinks import AutoModel

  model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-hf", device_map="auto")
  ```
* Support for Llama and Falcon models.
* New parameters to `AutoModel....from_pretrained`:
  * `attention_sink_size`, int, defaults to 4: The number of initial tokens to use as the attention sink. These tokens are always included in the Attention Sink KV Cache.
  * `attention_sink_window_size`, int, defaults to 1020: The size of the sliding window, i.e. the number of "recent tokens" to include in the Attention Sink KV Cache.

## Installation
You can install `attention_sinks` like so
```python
pip install attention_sinks
```

## Benchmarks
You can run a few benchmarks to compute the perplexity of various models over time using the provided [perplexity.py](benchmark/perplexity.py) benchmarking script. For example:
```
python benchmark/perplexity.py --experiment attention_sinks
```

<details><summary>Full argument list</summary>

```
usage: perplexity.py [-h] [--experiment {attention_sinks,transformers,windowed}] [--model_name_or_path MODEL_NAME_OR_PATH] [--revision REVISION]
                     [--trust_remote_code] [--dataset_name DATASET_NAME] [--data_column DATA_COLUMN] [--task TASK] [--split {validation,test}]
                     [--num_tokens NUM_TOKENS] [--output_dir OUTPUT_DIR] [--window_size WINDOW_SIZE] [--attention_sink_size ATTENTION_SINK_SIZE]

options:
  -h, --help            show this help message and exit
  --experiment {attention_sinks,transformers,windowed}
  --model_name_or_path MODEL_NAME_OR_PATH
  --revision REVISION
  --trust_remote_code
  --dataset_name DATASET_NAME
  --data_column DATA_COLUMN
  --task TASK
  --split {validation,test}
  --num_tokens NUM_TOKENS
  --output_dir OUTPUT_DIR
  --window_size WINDOW_SIZE
  --attention_sink_size ATTENTION_SINK_SIZE
```
</details>

This script will create a `csv` file in the output directory (`"benchmarks/outputs"` by default) for that experiment, with information about perplexities, CUDA VRAM usage and latencies.

This information can be plotted using the [plot_perplexity.py](benchmark\plot_perplexity.py) script. For example:
```
python benchmark/plot_perplexity.py --features perplexity latency --title "Log perplexity & latency of Llama 2 7B as a function of input lengths"
```

<details><summary>Full argument list</summary>

```
usage: plot_perplexity.py [-h] [--output_dir OUTPUT_DIR] [--features {perplexity,vram,latency} [{perplexity,vram,latency} ...]] [--title TITLE]
                          [--log_perplexity_limit LOG_PERPLEXITY_LIMIT] [--skip_first SKIP_FIRST]

options:
  -h, --help            show this help message and exit
  --output_dir OUTPUT_DIR
  --features {perplexity,vram,latency} [{perplexity,vram,latency} ...]
  --title TITLE
  --log_perplexity_limit LOG_PERPLEXITY_LIMIT
  --skip_first SKIP_FIRST
```
</details>

This script takes all `csv` files from the output directory (`"benchmark/outputs"` by default), and creates a plot like so:
```
python benchmark/plot_perplexity.py --features perplexity vram --title "Log perplexity & VRAM usage of Llama 2 7B as a function of input lengths" --output_dir benchmark/outputs_llama_2_7b --log_perplexity_limit 4
```

![llama_2_7b_ppl_vram_plotted](https://github.com/mit-han-lab/streaming-llm/assets/37621491/18802ec4-ed48-42be-ab26-ad9bfb83d0b7)

Clear as day:
1. `transformers`: The VRAM usage is linear as it doesn't do any windowing. The performance heavily falls after ~4096 tokens.
2. `windowed`: The VRAM is constant usage due to the windowing at 1024 tokens. However, it fails as soon as the first tokens leave the window.
3. `attention_sinks`: Constant VRAM usage due to windowing with 4 attention sink tokens + the 1020 most recent tokens. This approach never fails despite the constant VRAM usage.

I've uploaded [benchmark/outputs_llama_2_7b](benchmark/outputs_llama_2_7b) so you can reproduce this graph using the former command.

## Credits

Inspired by, and adapted from [StreamingLLM](https://github.com/mit-han-lab/streaming-llm).

### Citation

```
@article{xiao2023streamingllm,
    title={Efficient Streaming Language Models with Attention Sinks},
    author={Xiao, Guangxuan and Tian, Yuandong and Chen, Beidi and Han, Song and Lewis, Mike},
    journal={arXiv},
    year={2023}
}
```