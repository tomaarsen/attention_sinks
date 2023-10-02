import torch
from matplotlib import pyplot as plt

fig, ppl_ax = plt.subplots()
vram_ax = ppl_ax.twinx()

for module in ("transformers", "attention_sinks", "window_attention"):
    with open(f"outputs/{module}.txt", "r") as f:
        nlls = [float(line) for line in f.readlines() if line]
    max_X = len(nlls)
    X = range(100, max_X)
    perplexities = [torch.tensor(nlls[:x]).mean() for x in X]

    ppl_ax.plot(X, perplexities, label=module)

    with open(f"outputs/{module}_gpu_usage.txt", "r") as f:
        usages = [float(line) for line in f.readlines() if line]
    usages = [usages[x] for x in X]
    vram_ax.plot(X, usages, "--", label=f"{module} GPU usage")


ppl_ax.set_title(
    "Log Perplexity of Llama 2 7B as a function of input lengths, with and without Attention Sink KV Caches"
)
ppl_ax.set_xlabel("Input Sequence Length")
ppl_ax.set_ylabel("Perplexity (log) - (lower is better)")
vram_ax.set_ylabel("VRAM (GB) - (lower is better)")
ppl_ax.set_ylim(top=6)

ppl_ax.legend(loc="upper left")
vram_ax.legend(loc="upper right")
fig.tight_layout()
plt.show()
