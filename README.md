
# Attention Sinks in Transformers for Infinite-length LLMs

![llama_2_7b_ppl_vram](https://github.com/tomaarsen/attention_sinks/assets/37621491/b888155e-af73-46d4-8519-a010ecd247b0)

## Overview

* Extend existing LLMs (e.g. Llama 2) to infinite length without sacrificing efficiency and performance, without any retraining.
* The `attention_sinks` API allows for a drop-in replacement of the `transformers` API:
  ```python
  from attention_sinks import AutoModel

  model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-hf", device_map="auto")
  ```
  Currently, only Llama-based models are supported. Support for other models will come soon.
* New parameters to `AutoModel....from_pretrained`:
  * `attention_sink_size`, int, defaults to 4: The number of initial tokens to use as the attention sink. These tokens are always included in the Attention Sink KV Cache.
  * `attention_sink_window_size`, int, defaults to 1020: The size of the sliding window, i.e. the number of "recent tokens" to include in the Attention Sink KV Cache.

## Note

I've yet to replicate all of the experiments by the original paper, although I've replicated some. I can't confirm that this indeed allows for infinite-length LLMs in theory nor in practice.

More details coming soon.

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