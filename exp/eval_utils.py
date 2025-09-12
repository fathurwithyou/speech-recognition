# eval_utils.py
import re
import math
from typing import List

def normalize_text(s, remove_punct: bool = True, to_lower: bool = True) -> str:
    # Handle None/NaN/non-strings gracefully
    if s is None:
        return ""
    if not isinstance(s, str):
        # Treat NaN (float) as empty
        try:
            if isinstance(s, float) and math.isnan(s):
                return ""
        except Exception:
            pass
        s = str(s)

    if to_lower:
        s = s.lower()
    s = s.strip()
    if remove_punct:
        s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _edit_distance(seq_a: List[str], seq_b: List[str]) -> int:
    n, m = len(seq_a), len(seq_b)
    if n == 0: return m
    if m == 0: return n
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        ai = seq_a[i - 1]
        for j in range(1, m + 1):
            bj = seq_b[j - 1]
            cost = 0 if ai == bj else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[m]

def compute_wer(ref, hyp) -> float:
    ref_norm = normalize_text(ref)
    hyp_norm = normalize_text(hyp)
    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()
    if len(ref_words) == 0:
        return float("nan") if len(hyp_words) > 0 else 0.0
    dist = _edit_distance(ref_words, hyp_words)
    return dist / len(ref_words)

def compute_cer(ref, hyp) -> float:
    ref_norm = normalize_text(ref)
    hyp_norm = normalize_text(hyp)
    ref_chars = list(ref_norm.replace(" ", ""))
    hyp_chars = list(hyp_norm.replace(" ", ""))
    if len(ref_chars) == 0:
        return float("nan") if len(hyp_chars) > 0 else 0.0
    dist = _edit_distance(ref_chars, hyp_chars)
    return dist / len(ref_chars)
