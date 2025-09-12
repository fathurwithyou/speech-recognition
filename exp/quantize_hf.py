import os
import json
import time
import argparse
from pathlib import Path

import pandas as pd
from eval_utils import compute_wer, compute_cer


# --------------------- Helpers ---------------------
def human_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    for unit in ["KB", "MB", "GB", "TB"]:
        n /= 1024.0
        if n < 1024.0:
            return f"{n:.2f} {unit}"
    return f"{n:.2f} PB"


def set_cpu_threads(threads: int | None):
    if not threads or threads <= 0:
        return
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
    try:
        import torch
        torch.set_num_threads(threads)
    except Exception:
        pass


def load_audio_16k(path: str):
    import librosa
    audio, sr = librosa.load(path, sr=16000, mono=True)
    return audio, 16000


def transcribe_generate(model, processor, audio, sr, language: str, beam_size: int, return_timestamps: bool) -> str:
    """Transcribe with HF Whisper using model.generate (no pipeline)."""
    import torch

    LANG_FIX = {"en": "en", "id": "id"}
    lang = LANG_FIX.get((language or "").lower(), language or "id")
    model.generation_config.task = "transcribe"
    model.generation_config.language = lang

    inputs = processor(
        audio,
        sampling_rate=sr,
        return_tensors="pt",
        padding="longest",
        truncation=False,
        return_attention_mask=True,
    )


    model.eval()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            return_timestamps=return_timestamps,  # True for >30s audio
        )

    sequences = getattr(out, "sequences", out)
    text = processor.batch_decode(sequences, skip_special_tokens=True)[0]
    return text.strip()


def quantize_model_int8(model_fp32, quant_mode: str = "dynamic"):
    """
    Quantize to INT8 with TorchAO eager API. Fallback to torch.ao (deprecated) if TorchAO unavailable.
    quant_mode:
      - "dynamic" (default): int8 weights + dynamic int8 activations for Linear layers
      - "weight_only": int8 weights only
    """
    import copy
    try:
        from torchao.quantization import (
            Int8DynamicActivationInt8WeightConfig,
            Int8WeightOnlyConfig,
            quantize_,
        )
        m_int8 = copy.deepcopy(model_fp32)
        if quant_mode == "weight_only":
            quantize_(m_int8, Int8WeightOnlyConfig())
            engine = "torchao.weight_only"
        else:
            quantize_(m_int8, Int8DynamicActivationInt8WeightConfig())
            engine = "torchao.dynamic"
        return m_int8, engine
    except Exception as e:
        # Fallback to deprecated torch.ao API if TorchAO isn't installed or fails
        import torch
        from warnings import warn
        warn(f"TorchAO not available, falling back to torch.ao dynamic INT8: {e}")
        from torch.ao.quantization import quantize_dynamic
        m_int8 = quantize_dynamic(model_fp32, {torch.nn.Linear}, dtype=torch.qint8)
        return m_int8, "torch.ao.dynamic (deprecated)"


# --------------------- Main ---------------------
def main():
    ap = argparse.ArgumentParser(description="HF Whisper FP32 vs INT8 (TorchAO) quantization & evaluation over CSV")
    ap.add_argument("--dataset", default="./one.csv", help="CSV with columns: audio,reference[,language]")
    ap.add_argument("--model-id", default="openai/whisper-small", help="HF model id (e.g., openai/whisper-small)")
    ap.add_argument("--language", default="id", help="Default language if dataset has no 'language' column")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--outdir", default="./quant_outputs")
    ap.add_argument("--quant-mode", choices=["dynamic", "weight_only"], default="dynamic",
                    help="INT8 quantization mode (TorchAO): dynamic (w8a8-dynamic) or weight_only (w8)")
    ap.add_argument("--return-timestamps", action="store_true",
                    help="Use Whisper long-form decoding (recommended for >30s audio)")
    args = ap.parse_args()

    set_cpu_threads(args.threads)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # -------- Load dataset --------
    df = pd.read_csv(args.dataset)
    if "audio" not in df.columns:
        raise ValueError("Dataset must contain column 'audio'.")
    if "reference" not in df.columns:
        df["reference"] = ""

    # -------- Load model & processor (FP32) --------
    import torch
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

    t_load0 = time.perf_counter()
    model_fp32 = AutoModelForSpeechSeq2Seq.from_pretrained(args.model_id, dtype=torch.float32)
    load_fp32_time = time.perf_counter() - t_load0
    processor = AutoProcessor.from_pretrained(args.model_id)

    # -------- Save FP32 as state_dict (safe) --------
    fp32_sd_path = outdir / "model_fp32_sd.pt"
    torch.save(model_fp32.state_dict(), fp32_sd_path)
    fp32_size = fp32_sd_path.stat().st_size

    # -------- Quantize (INT8) & save state_dict (safe) --------
    model_int8, quant_engine = quantize_model_int8(model_fp32, quant_mode=args.quant_mode)
    int8_sd_path = outdir / "model_int8_sd.pt"
    torch.save(model_int8.state_dict(), int8_sd_path)
    int8_size = int8_sd_path.stat().st_size

    # -------- Measure INT8 *load* time (rebuild graph + quantize + load sd) --------
    # This avoids pickling issues and keeps weights_only=True (safe) semantics.
    import gc
    del model_int8
    gc.collect()

    t_load_int8_0 = time.perf_counter()
    model_int8 = AutoModelForSpeechSeq2Seq.from_pretrained(args.model_id, dtype=torch.float32)
    model_int8, _ = quantize_model_int8(model_int8, quant_mode=args.quant_mode)
    sd_int8 = torch.load(int8_sd_path, map_location="cpu", weights_only=True)
    model_int8.load_state_dict(sd_int8, strict=False)
    load_int8_time = time.perf_counter() - t_load_int8_0

    # -------- Per-sample inference & metrics --------
    rows = []
    for i, r in df.iterrows():
        audio_path = str(r["audio"])
        ref = r["reference"]
        lang = (r["language"] if "language" in df.columns else "") or args.language

        # Coerce reference to string (handle NaN)
        if not isinstance(ref, str):
            ref = "" if pd.isna(ref) else str(ref)

        # Load audio once per sample
        audio, sr = load_audio_16k(audio_path)

        # FP32
        t0 = time.perf_counter()
        hyp_fp32 = transcribe_generate(
            model_fp32, processor, audio, sr, lang, args.beam_size, args.return_timestamps
        )
        t_fp32 = time.perf_counter() - t0

        # INT8 (reconstructed + loaded sd)
        t1 = time.perf_counter()
        hyp_int8 = transcribe_generate(
            model_int8, processor, audio, sr, lang, args.beam_size, args.return_timestamps
        )
        t_int8 = time.perf_counter() - t1

        # WER/CER
        if ref.strip():
            wer_fp32 = compute_wer(ref, hyp_fp32)
            cer_fp32 = compute_cer(ref, hyp_fp32)
            wer_int8 = compute_wer(ref, hyp_int8)
            cer_int8 = compute_cer(ref, hyp_int8)
        else:
            wer_fp32 = cer_fp32 = float("nan")
            wer_int8 = cer_int8 = float("nan")

        rows.append({
            "index": i,
            "audio": audio_path,
            "language": lang,
            "reference": ref,
            "hypothesis_fp32": hyp_fp32,
            "hypothesis_int8": hyp_int8,
            "wer_fp32": wer_fp32,
            "cer_fp32": cer_fp32,
            "wer_int8": wer_int8,
            "cer_int8": cer_int8,
            "infer_time_fp32_s": t_fp32,
            "infer_time_int8_s": t_int8,
            "speedup_fp32_over_int8": (t_fp32 / t_int8) if t_int8 > 0 else float("inf"),
        })

    res_df = pd.DataFrame(rows)
    res_csv = outdir / "per_sample_results.csv"
    res_df.to_csv(res_csv, index=False, encoding="utf-8")

    # -------- Summary --------
    size_delta = fp32_size - int8_size
    size_drop_pct = (size_delta / fp32_size * 100.0) if fp32_size > 0 else float("nan")

    summary = {
        "model_id": args.model_id,
        "threads": args.threads,
        "beam_size": args.beam_size,
        "return_timestamps": bool(args.return_timestamps),
        "quant_engine": quant_engine,
        "sizes": {
            "fp32_bytes": fp32_size,
            "fp32_human": human_bytes(fp32_size),
            "int8_bytes": int8_size,
            "int8_human": human_bytes(int8_size),
            "reduction_bytes": size_delta,
            "reduction_pct": round(size_drop_pct, 2),
        },
        "timing": {
            "model_load_fp32_s": round(load_fp32_time, 3),
            "model_load_int8_s": round(load_int8_time, 3),
            "infer_fp32_mean_s": round(float(res_df["infer_time_fp32_s"].mean()), 3) if not res_df.empty else float("nan"),
            "infer_int8_mean_s": round(float(res_df["infer_time_int8_s"].mean()), 3) if not res_df.empty else float("nan"),
            "speedup_mean": round(float(res_df["speedup_fp32_over_int8"].mean()), 3) if not res_df.empty else float("nan"),
        },
    }

    if res_df["reference"].str.strip().any():
        summary["wer_cer"] = {
            "fp32_wer_mean": round(float(res_df["wer_fp32"].mean()), 6),
            "fp32_cer_mean": round(float(res_df["cer_fp32"].mean()), 6),
            "int8_wer_mean": round(float(res_df["wer_int8"].mean()), 6),
            "int8_cer_mean": round(float(res_df["cer_int8"].mean()), 6),
        }
    else:
        summary["wer_cer"] = "no reference provided"

    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n==== SUMMARY ====")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved:\n- {fp32_sd_path}\n- {int8_sd_path}\n- {res_csv}\n- {outdir / 'summary.json'}")


if __name__ == "__main__":
    main()
