
# Attention Sinks in Transformers for Infinite-length LLMs

| Llama 2 7B | Falcon 7B |
|:-------------:|:-------------:|
| ![llama_2_7b_ppl_vram_plotted](https://github.com/tomaarsen/attention_sinks/assets/37621491/8d2e5b88-7158-41ac-8b3a-5a7abe38020d)  | ![falcon_7b_ppl_vram_plotted](https://github.com/tomaarsen/attention_sinks/assets/37621491/1be07370-6de7-4a7e-b5ab-3092a5ecb412)  |
| **MPT 7B** | **Pythia 6.9B** |
| ![mpt_7b_ppl_vram_plotted](https://github.com/mit-han-lab/streaming-llm/assets/37621491/c96cff66-92a3-43ab-bc21-40232f2740a0) | ![pythia_6 8b_ppl_vram_plotted](https://github.com/tomaarsen/attention_sinks/assets/37621491/b0fee168-fa5a-457d-9e27-8395eb6dfb38) |
| **Mistral 7B** | |
| ![mistral_7b_ppl_vram_plotted](https://github.com/microsoft/torchscale/assets/37621491/3a4c5634-cc1b-42d1-a35a-afb376a4f970) | |

## Overview

This repository is an open-source implementation of the [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) paper.

* Extend existing LLMs (e.g. Llama 2) to infinite length without sacrificing efficiency and performance, without any retraining.
  * Model perplexities were stable even after 4 million tokens!
  * Unlike with regular `transformers`, memory usage is constant and thus the inference does not get extremely slow due to memory issues at higher sequence lengths.
  * Models using attention sinks have been shown to perform very well at the task of recalling a value from 20 lines back, even if the model has already processed hundreds of thousands of lines, whereas models using regular dense or window attention fall to 0% after having processed a few thousand tokens.
* The `attention_sinks` API allows for a drop-in replacement of the `transformers` API:
  ```python
  from attention_sinks import AutoModel

  model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-hf", device_map="auto")
  ```
* Support for Llama, Falcon, MPT, GPTNeoX (Pythia) and Mistral models.
* New parameters to `AutoModel....from_pretrained`:
  * `attention_sink_size`, `int`, defaults to 4: The number of initial tokens to use as the attention sink. These tokens are always included in the Attention Sink KV Cache.
  * `attention_sink_window_size`, `int`, defaults to 1020: The size of the sliding window, i.e. the number of "recent tokens" to include in the Attention Sink KV Cache. A larger window size costs more memory.

## Installation
You can install `attention_sinks` like so
```python
pip install attention_sinks
```

### Usage
Loading any Llama, Falcon, MPT, GPTNeoX (Pythia) or Mistral model is as simple as loading it in `transformers`, the only change is that the model class must be imported from `attention_sinks` rather than `transformers`, e.g.:
```python
from attention_sinks import AutoModel

model = AutoModel.from_pretrained("mosaicml/mpt-7b", device_map="auto")
```

Generation can be done like you would expect from `transformers`, e.g. like so:
```python
import torch
from transformers import AutoTokenizer, TextStreamer, GenerationConfig
from attention_sinks import AutoModelForCausalLM


# model_id = "meta-llama/Llama-2-7b-hf"
# model_id = "mistralai/Mistral-7B-v0.1"
model_id = "mosaicml/mpt-7b"
# model_id = "tiiuae/falcon-7b"
# model_id = "EleutherAI/pythia-6.9b-deduped"
# Note: instruct or chat models also work.

# Load the chosen model and corresponding tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # for efficiency:
    device_map="auto",
    torch_dtype=torch.float16,
    # `attention_sinks`-specific arguments:
    attention_sink_size=4,
    attention_sink_window_size=252, # <- Low for the sake of faster generation
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id

# Our input text
text = "Vaswani et al. (2017) introduced the Transformers"

# Encode the text
input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)

# Print tokens as they're being generated
streamer = TextStreamer(tokenizer)
generated_tokens = model.generate(
    input_ids,
    generation_config=GenerationConfig(
        # use_cache=True is required, the rest can be changed up.
        use_cache=True,
        min_new_tokens=100_000,
        max_new_tokens=1_000_000,
        penalty_alpha=0.6,
        top_k=5,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    ),
    streamer=streamer,
)
# Decode the final generated text
output_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
```
This example will happily generate between 100k and 1m tokens without forgetting how to speak, even on a low-VRAM environment like Google Colab when using `load_in_4bit=True` in the `AutoModelForCausalLM.from_pretrained`.

## Benchmarks

### Pre-prepared benchmarks
See [benchmark/scripts](benchmark/scripts) for a collection of ready-to-go scripts for various model architectures like Llama 2, Falcon, MPT and GPT-NeoX (Pythia). Each of these scripts runs the benchmarking and plotting tools described below for pure [`transformers`](https://github.com/huggingface/transformers), [`attention_sinks`](https://github.com/tomaarsen/attention_sinks) and a third alternative: `windowed`, which involves simple windowed attention at a window size of 1024 tokens. Upon completion, the script will plot the figures that you see at the top of this README.

### Benchmarking tool
You can run a few benchmarks to compute the perplexity of various models over time using the provided [perplexity.py](benchmark/perplexity.py) benchmarking script. This is done by computing the negative log likelihood losses of the chosen model when it is provided a full book with 60k+ tokens. By default, the scripts stop after 8192 tokens, but this can be modified. An ideal solution continuously has a low log perplexity and a constant CUDA VRAM usage.

To use the script, you can run:
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

### Plotting tool
The information from the benchmarking tool can be plotted using the [plot_perplexity.py](benchmark\plot_perplexity.py) script. In particular, you can plot any combination of the following features:
* `perplexity`,
* `vram`, i.e. CUDA VRAM usage,
* `latency`.

For example:
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

I've uploaded outputs of various benchmarks in [benchmark](benchmark) so you can reproduce this graph using the former command.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for all release information.

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