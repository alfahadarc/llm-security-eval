"""Defense B: Llama Prompt Guard 2 86M injection detector (Step 16a).

The detector scores an input text and returns P(malicious) (LABEL_1). Texts longer than the 512-token window are
split into segments and the case score is the maximum over segments (the model card's advice for long inputs).
The model is loaded at a pinned revision and runs on the CPU unless a device is given.
"""
import time

MODEL_ID = "meta-llama/Llama-Prompt-Guard-2-86M"
REVISION = "a8ded8e697ce7c355e395a0df51f94adb4a2fd27"
MALICIOUS_LABEL = 1
WINDOW = 512  # model context window, special tokens included


def split_segments(token_ids, content_len, stride):
    """Split a list of token ids (no special tokens) into windows of content_len, advancing by stride."""
    if content_len <= 0 or not 0 < stride <= content_len:
        raise ValueError("need 0 < stride <= content_len")
    if len(token_ids) <= content_len:
        return [list(token_ids)]
    segments, start = [], 0
    while True:
        segments.append(list(token_ids[start:start + content_len]))
        if start + content_len >= len(token_ids):
            return segments
        start += stride


def decide(score, threshold):
    """Flag when the malicious probability reaches the threshold."""
    return "block" if score >= threshold else "allow"


class PromptGuard:
    def __init__(self, device="cpu", revision=REVISION):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.device = device
        self.revision = revision
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, revision=revision)
        self.model.to(device).eval()
        self.content_len = WINDOW - self.tokenizer.num_special_tokens_to_add(pair=False)

    def score_text(self, text):
        """Return (max P(malicious) over segments, number of segments, runtime in seconds)."""
        start = time.perf_counter()
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        segments = split_segments(ids, self.content_len, self.content_len // 2)  # 50% overlap
        best = 0.0
        for seg in segments:
            full = [self.tokenizer.cls_token_id] + seg + [self.tokenizer.sep_token_id]
            inputs = {"input_ids": self.torch.tensor([full], device=self.device),
                      "attention_mask": self.torch.ones(1, len(full), dtype=self.torch.long, device=self.device)}
            with self.torch.no_grad():
                probs = self.model(**inputs).logits.softmax(-1)[0]
            best = max(best, float(probs[MALICIOUS_LABEL]))
        return best, len(segments), time.perf_counter() - start
