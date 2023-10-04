from pathlib import Path

from transformers import AutoTokenizer, TextStreamer


class FileStreamer(TextStreamer):
    def __init__(self, tokenizer: AutoTokenizer, log_file: str, skip_prompt: bool = False, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Prints the new text to stdout and writes it to a file. If the stream is ending, also prints a newline."""
        super().on_finalized_text(text, stream_end=stream_end)
        with open(self.log_file, "a+", encoding="utf8") as f:
            print(text, flush=True, end="" if not stream_end else None, file=f)
