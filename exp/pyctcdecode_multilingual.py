import os
import time
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from eval_utils import compute_wer, compute_cer

# --------------------- Threads ---------------------
def set_cpu_threads(threads: int | None):
    if threads and threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
        try:
            import torch
            torch.set_num_threads(threads)
        except Exception:
            pass

# --------------------- Labels & decoder ---------------------
def build_labels_from_processor(processor):
    """
    Susun label CTC sesuai id tokenizer. Map pemisah kata ke spasi.
    Drop token spesial (<s>, </s>, <pad>, <unk>) jika ada.
    """
    vocab = processor.tokenizer.get_vocab()  # token -> id
    id2tok = [None] * len(vocab)
    for tok, idx in vocab.items():
        if idx < 0:
            continue
        if idx >= len(id2tok):
            id2tok.extend([None] * (idx - len(id2tok) + 1))
        id2tok[idx] = tok

    labels = []
    word_delim = getattr(processor.tokenizer, "word_delimiter_token", None)
    for tok in id2tok:
        if tok is None:
            tok = ""
        if tok == word_delim or tok == "|":
            labels.append(" ")
        elif tok in ["<s>", "</s>", "<pad>", "<unk>"]:
            labels.append("")
        else:
            labels.append(tok)
    return labels

def build_decoder(processor, kenlm_path: str | None):
    from pyctcdecode import build_ctcdecoder
    labels = build_labels_from_processor(processor)
    decoder = build_ctcdecoder(labels=labels, kenlm_model_path=(kenlm_path or None))
    return decoder

# --------------------- LM routing ---------------------
def load_lm_map(path_json: str | None):
    if not path_json:
        return {}
    p = Path(path_json)
    if not p.exists():
        raise FileNotFoundError(f"LM map file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def find_lm_for_lang(lang: str, lm_dir: str | None, lm_map: dict) -> str | None:
    """Cari file LM berdasarkan bahasa; prioritas: lm_map > lm_dir/<lang>.arpa|.bin|.klm"""
    if not lang:
        lang = "und"
    lang = lang.lower()
    # 1) explicit map
    if lang in lm_map:
        return lm_map[lang]
    # 2) directory pattern
    if lm_dir:
        cand = [f"{lang}.arpa", f"{lang}.bin", f"{lang}.klm"]
        for name in cand:
            p = Path(lm_dir) / name
            if p.exists():
                return str(p)
    return None

# --------------------- Language detection (optional) ---------------------
def detect_language(text_or_path: str) -> str:
    """
    Deteksi bahasa ringan (pakai langid) dari transcript sementara ATAU gunakan file audio?
    Di pipeline ini kita deteksi dari path nama file sebagai fallback (kurang ideal).
    Lebih baik: jalankan deteksi dari hipotesis cepat dulu, lalu re-decode dgn LM per-bahasa jika perlu (2-pass).
    Untuk sederhana, kita pakai langid di 'reference' jika ada, atau fallback 'en'.
    """
    try:
        import langid
        langid.set_languages(None)  # default: semua
        lang, _ = langid.classify(text_or_path)
        return lang
    except Exception:
        return "en"

# --------------------- Core transcribe ---------------------
def transcribe_with_pyctcdecode(
    audio_path: str,
    model,
    processor,
    decoder,              # BeamSearchDecoderCTC (pyctcdecode)
    beam_width: int = 100
) -> str:
    import librosa
    import torch
    # Audio 16k mono
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding="longest")

    # Logits -> log-probs
    with torch.no_grad():
        logits = model(inputs.input_values).logits  # [B, T, V]
    logp = torch.log_softmax(logits, dim=-1)[0].cpu().numpy().astype(np.float32)  # [T, V]

    # Decode
    text = decoder.decode(logp, beam_width=beam_width)
    return text.strip()

# --------------------- Main ---------------------
def main():
    ap = argparse.ArgumentParser(description="Multilingual CTC + pyctcdecode over CSV (no hotwords)")
    ap.add_argument("--dataset", default="./one.csv", help="CSV columns: audio,reference[,language]")
    ap.add_argument("--ctc-model", default="facebook/mms-1b-all",
                    help="HF CTC multilingual model (e.g., facebook/mms-1b-all, facebook/mms-300m)")
    ap.add_argument("--beam-width", type=int, default=150)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--outdir", default="./pyctcdecode_multi_out")

    # LM options (optional)
    ap.add_argument("--lm-dir", default="", help="Directory with per-language LMs named <lang>.arpa/.bin/.klm")
    ap.add_argument("--lm-map", default="", help="JSON mapping: {\"en\": \"path/to/en.arpa\", \"id\": \"path/to/id.arpa\"}")

    # Language handling
    ap.add_argument("--language-col", default="language", help="CSV column with BCP-47/lang code (e.g., en, id)")
    ap.add_argument("--lang-detect", action="store_true", help="Auto-detect language if column missing/empty (via langid)")
    args = ap.parse_args()

    set_cpu_threads(args.threads)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    df = pd.read_csv(args.dataset)
    if "audio" not in df.columns:
        raise ValueError("Dataset missing required column: audio")
    if "reference" not in df.columns:
        df["reference"] = ""
    if args.language_col not in df.columns:
        df[args.language_col] = ""

    # Load multilingual CTC model & processor (shared)
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    print(f"[1/4] Loading multilingual CTC model: {args.ctc_model}")
    model = Wav2Vec2ForCTC.from_pretrained(args.ctc_model)
    processor = Wav2Vec2Processor.from_pretrained(args.ctc_model)
    model.eval()

    # Prepare LM routing
    lm_map = load_lm_map(args.lm_map) if args.lm_map else {}
    lm_dir = args.lm_dir or None

    # Cache decoders per LM path (including None)
    from pyctcdecode import BeamSearchDecoderCTC
    decoder_cache: dict[str | None, BeamSearchDecoderCTC] = {}

    def get_decoder_for_lang(lang_code: str) -> BeamSearchDecoderCTC:
        lm_path = find_lm_for_lang(lang_code, lm_dir, lm_map)
        key = lm_path or "__NO_LM__"
        if key not in decoder_cache:
            decoder_cache[key] = build_decoder(processor, lm_path)
        return decoder_cache[key]

    # Transcribe loop
    print("[2/4] Transcribing...")
    rows = []
    for i, r in df.iterrows():
        audio = str(r["audio"])
        ref = r["reference"]
        lang = str(r.get(args.language_col, "") or "").strip().lower()

        if not isinstance(ref, str):
            ref = "" if pd.isna(ref) else str(ref)

        # determine language
        if not lang and args.lang_detect:
            # try detect from filename or reference as heuristic
            seed_text = Path(audio).name + " " + (ref[:200] if isinstance(ref, str) else "")
            lang = detect_language(seed_text)

        # get decoder for this language (LM chosen per language if available)
        decoder = get_decoder_for_lang(lang)

        t0 = time.perf_counter()
        try:
            hyp = transcribe_with_pyctcdecode(
                audio_path=audio,
                model=model,
                processor=processor,
                decoder=decoder,
                beam_width=args.beam_width,
            )
            dt = time.perf_counter() - t0
        except Exception as e:
            hyp = f"[ERROR] {e}"
            dt = float("nan")

        rows.append({
            "index": i,
            "audio": audio,
            "language": lang or "",
            "reference": ref,
            "hypothesis": hyp,
            "wer": compute_wer(ref, hyp),
            "cer": compute_cer(ref, hyp),
            "latency_s": dt,
        })

    res_df = pd.DataFrame(rows)
    res_csv = outdir / "results_multilingual_pyctcdecode.csv"
    res_df.to_csv(res_csv, index=False, encoding="utf-8")

    # Summaries (overall + per language)
    print("[3/4] Summarizing...")
    overall = {
        "ctc_model": args.ctc_model,
        "threads": args.threads,
        "beam_width": args.beam_width,
        "lm_dir": bool(lm_dir),
        "lm_map": bool(lm_map),
        "samples": int(res_df.shape[0]),
        "wer_mean": float(res_df["wer"].mean()) if not res_df.empty else float("nan"),
        "cer_mean": float(res_df["cer"].mean()) if not res_df.empty else float("nan"),
        "latency_mean_s": float(res_df["latency_s"].mean()) if not res_df.empty else float("nan"),
    }

    by_lang = []
    if "language" in res_df.columns:
        for lang_code, grp in res_df.groupby("language", dropna=False):
            by_lang.append({
                "language": lang_code or "",
                "samples": int(grp.shape[0]),
                "wer_mean": float(grp["wer"].mean()),
                "cer_mean": float(grp["cer"].mean()),
                "latency_mean_s": float(grp["latency_s"].mean()),
            })

    out = {"overall": overall, "by_language": by_lang}
    with open(outdir / "summary_multilingual_pyctcdecode.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n=== Multilingual pyctcdecode Summary ===")
    print(json.dumps(out, indent=2))
    print(f"\n[4/4] Saved:\n- {res_csv}\n- {outdir / 'summary_multilingual_pyctcdecode.json'}")

if __name__ == "__main__":
    main()
