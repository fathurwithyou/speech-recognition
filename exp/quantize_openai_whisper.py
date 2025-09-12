import os
import json
import time
import argparse
from pathlib import Path

import pandas as pd
from eval_utils import compute_wer, compute_cer


# ===================== Helpers =====================
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


def quantize_model_int8(model_fp32, quant_mode: str = "dynamic", copy_model: bool = True):
    """
    Quantize INT8 dengan TorchAO eager.
    Fallback ke torch.ao (deprecated) bila TorchAO tak tersedia.

    quant_mode:
      - "dynamic" (default): int8 weights + dynamic int8 activations (Linear)
      - "weight_only": int8 weights only (Linear)
    copy_model:
      - True  -> quantize pada salinan (aman; dipakai saat membuat ckpt)
      - False -> in-place (cepat; dipakai saat 'prep graph' re-load)
    """
    import copy
    try:
        from torchao.quantization import (
            Int8DynamicActivationInt8WeightConfig,
            Int8WeightOnlyConfig,
            quantize_,
        )
        m = copy.deepcopy(model_fp32) if copy_model else model_fp32
        if quant_mode == "weight_only":
            quantize_(m, Int8WeightOnlyConfig())
            engine = "torchao.weight_only"
        else:
            quantize_(m, Int8DynamicActivationInt8WeightConfig())
            engine = "torchao.dynamic"
        return m, engine
    except Exception as e:
        import torch
        from warnings import warn
        warn(f"TorchAO unavailable, falling back to torch.ao dynamic INT8: {e}")
        from torch.ao.quantization import quantize_dynamic
        m = copy.deepcopy(model_fp32) if copy_model else model_fp32
        m = quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
        return m, "torch.ao.dynamic (deprecated)"


def transcribe_openai(model, audio_path: str, language: str, beam_size: int) -> str:
    """
    Transkripsi via openai-whisper (model.transcribe di CPU).
    """
    result = model.transcribe(
        audio_path,
        language=(language or None),
        task="transcribe",
        condition_on_previous_text=True,
        fp16=False,
        verbose=False,
    )
    return (result.get("text") or "").strip()


# ===================== Main =====================
def main():
    ap = argparse.ArgumentParser(description="Quantize OpenAI Whisper (TorchAO) + evaluate FP32 vs INT8 over CSV")
    ap.add_argument("--dataset", default="./one.csv", help="CSV: audio,reference[,language]")
    ap.add_argument("--model", default="small",
                    help="OpenAI Whisper model: tiny/base/small/medium/large/large-v2 (sesuaikan dengan yang tersedia)")
    ap.add_argument("--language", default="id", help="Default language jika CSV tidak punya kolom 'language'")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--outdir", default="./quant_outputs_openai")
    ap.add_argument("--quant-mode", choices=["dynamic", "weight_only"], default="dynamic",
                    help="TorchAO: dynamic (w8a8-dynamic) atau weight_only (w8)")
    ap.add_argument("--download-root", default=None,
                    help="Folder cache model openai-whisper (opsional).")
    args = ap.parse_args()

    set_cpu_threads(args.threads)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # -------- Load dataset --------
    df = pd.read_csv(args.dataset)
    if "audio" not in df.columns:
        raise ValueError("Dataset must contain column 'audio'.")
    if "reference" not in df.columns:
        df["reference"] = ""

    # -------- Load OpenAI Whisper FP32 --------
    import whisper
    from whisper.model import Whisper as WhisperCore  # untuk membangun 'blank' arsitektur
    import torch
    import gc

    print(f"[1/7] Loading FP32 Whisper model: {args.model}")
    t0 = time.perf_counter()
    model_fp32 = whisper.load_model(args.model, device="cpu", download_root=args.download_root)
    load_fp32_time = time.perf_counter() - t0

    # Simpan FP32 state_dict
    fp32_sd_path = outdir / "openai_model_fp32_sd.pt"
    torch.save(model_fp32.state_dict(), fp32_sd_path)
    fp32_size = fp32_sd_path.stat().st_size

    # -------- Quantize & save INT8 state_dict --------
    print("[2/7] Quantizing to INT8 (TorchAO)...")
    model_int8_ckpt, quant_engine = quantize_model_int8(model_fp32, quant_mode=args.quant_mode, copy_model=True)
    int8_sd_path = outdir / "openai_model_int8_sd.pt"
    torch.save(model_int8_ckpt.state_dict(), int8_sd_path)
    int8_size = int8_sd_path.stat().st_size
    del model_int8_ckpt
    gc.collect()

    # -------- Build blank graphs & measure prep/load split --------
    # Siapkan arsitektur kosong dari dims (tanpa bobot), supaya load-time tak ikut download FP32.
    print("[3/7] Preparing blank graphs for fair load-time measurement...")
    dims = model_fp32.dims  # ModelDimensions
    # FP32 blank graph
    t_prep_fp32_0 = time.perf_counter()
    model_fp32_blank = WhisperCore(dims)
    prep_fp32_graph_s = time.perf_counter() - t_prep_fp32_0

    t_load_fp32_w0 = time.perf_counter()
    sd_fp32 = torch.load(fp32_sd_path, map_location="cpu", weights_only=True)
    model_fp32_blank.load_state_dict(sd_fp32, strict=False)
    load_fp32_weights_s = time.perf_counter() - t_load_fp32_w0

    # INT8 blank graph + in-place quantize_
    t_prep_int8_0 = time.perf_counter()
    model_int8 = WhisperCore(dims)
    model_int8, _ = quantize_model_int8(model_int8, quant_mode=args.quant_mode, copy_model=False)  # in-place
    prep_int8_graph_s = time.perf_counter() - t_prep_int8_0

    t_load_int8_w0 = time.perf_counter()
    sd_int8 = torch.load(int8_sd_path, map_location="cpu", weights_only=True)
    model_int8.load_state_dict(sd_int8, strict=False)
    load_int8_weights_s = time.perf_counter() - t_load_int8_w0

    # model_fp32.eval(); model_int8.eval()

    # -------- Per-sample inference & metrics --------
    print("[4/7] Running per-sample inference...")
    rows = []
    for i, r in df.iterrows():
        audio_path = str(r["audio"])
        ref = r["reference"]
        lang = (r["language"] if "language" in df.columns else "") or args.language
        if not isinstance(ref, str):
            ref = "" if pd.isna(ref) else str(ref)

        # FP32
        t1 = time.perf_counter()
        hyp_fp32 = transcribe_openai(model_fp32, audio_path, lang, args.beam_size)
        t_fp32 = time.perf_counter() - t1

        # INT8
        t2 = time.perf_counter()
        hyp_int8 = transcribe_openai(model_int8, audio_path, lang, args.beam_size)
        t_int8 = time.perf_counter() - t2

        if ref.strip():
            wer_fp32 = compute_wer(ref, hyp_fp32); cer_fp32 = compute_cer(ref, hyp_fp32)
            wer_int8 = compute_wer(ref, hyp_int8); cer_int8 = compute_cer(ref, hyp_int8)
        else:
            wer_fp32 = cer_fp32 = float("nan"); wer_int8 = cer_int8 = float("nan")

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
    res_csv = outdir / "per_sample_results_openai.csv"
    res_df.to_csv(res_csv, index=False, encoding="utf-8")

    # -------- Summary --------
    print("[5/7] Summarizing results...")
    size_delta = fp32_size - int8_size
    size_drop_pct = (size_delta / fp32_size * 100.0) if fp32_size > 0 else float("nan")

    summary = {
        "openai_model": args.model,
        "threads": args.threads,
        "beam_size": args.beam_size,
        "quant_engine": "TorchAO (fallback torch.ao if needed)",
        "quant_mode": args.quant_mode,
        "sizes": {
            "fp32_bytes": fp32_size,
            "fp32_human": human_bytes(fp32_size),
            "int8_bytes": int8_size,
            "int8_human": human_bytes(int8_size),
            "reduction_bytes": size_delta,
            "reduction_pct": round(size_drop_pct, 2),
        },
        "timing": {
            "load_fp32_pretrained_s": round(load_fp32_time, 3),
            "prep_fp32_graph_s": round(prep_fp32_graph_s, 3),
            "load_fp32_weights_s": round(load_fp32_weights_s, 3),
            "prep_int8_graph_s": round(prep_int8_graph_s, 3),      # termasuk waktu quantize_
            "load_int8_weights_s": round(load_int8_weights_s, 3),  # murni load sd INT8
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

    # -------- Save summary --------
    print("[6/7] Writing outputs...")
    with open(outdir / "summary_openai.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n==== SUMMARY OPENAI WHISPER ====")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved:\n- {fp32_sd_path}\n- {int8_sd_path}\n- {res_csv}\n- {outdir / 'summary_openai.json'}")
    print("[7/7] Done.")


if __name__ == "__main__":
    main()
