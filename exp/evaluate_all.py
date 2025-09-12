import argparse
import os
import json
import time
import pandas as pd
from pathlib import Path
from eval_utils import compute_wer, compute_cer

# --------------------- CPU threads ---------------------
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

# --------------------- Backends ------------------------
def transcribe_transformers(audio_path: str, model_id: str, language: str, beam_size: int) -> str:
    import torch
    import librosa
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, dtype=torch.float32)
    processor = AutoProcessor.from_pretrained(model_id)

    LANG_FIX = {"en": "en", "id": "id"}
    lang = LANG_FIX.get((language or "").lower(), language or "id")

    # Configure generation (preferred over forced_decoder_ids)
    model.generation_config.task = "transcribe"
    model.generation_config.language = lang

    # Load audio at 16 kHz and build features WITHOUT truncation
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding="longest",
        truncation=False,           # important for long-form
        return_attention_mask=True, # recommended for generate
    )

    # For long-form audio, set return_timestamps=True to use Whisper’s long-form algorithm.
    # If your clips are short (<=30s), leaving it False is fine.
    return_timestamps = True

    with torch.no_grad():
        gen_out = model.generate(
            **inputs,
            return_timestamps=return_timestamps,
        )

    # Handle both tensor and ModelOutput/dict cases
    sequences = getattr(gen_out, "sequences", gen_out)
    text = processor.batch_decode(sequences, skip_special_tokens=True)[0]
    return text.strip()


def transcribe_faster_whisper(audio_path: str, model_name: str, language: str, beam_size: int, compute_type: str) -> str:
    from faster_whisper import WhisperModel
    model = WhisperModel(model_name, device="cpu", compute_type=compute_type, cpu_threads=16)
    segments, _ = model.transcribe(
        audio_path,
        language=language or None,
        task="transcribe",
        beam_size=beam_size,
        temperature=0.0,
        vad_filter=False,
        condition_on_previous_text=True,
    )
    return "".join(seg.text for seg in segments)


def transcribe_whisperx(
    audio_path: str,
    model_name: str,          # e.g. "large-v2" / "large-v3" / "medium" / "small"
    language: str,            # e.g. "en", "id" (or ""/None for auto)
    beam_size: int,
    compute_type: str,        # "float32" | "int16" | "int8" (use "float16" only on CUDA)
) -> str:
    import whisperx

    device = "cpu"            # change to "cuda" if you have GPU
    if device == "cpu" and compute_type.lower() == "float16":
        compute_type = "float32"  # avoid invalid type on CPU

    # Load audio like the reference
    audio = whisperx.load_audio(audio_path)

    # Accuracy-first decoding options
    asr_options = {
        "beam_size": beam_size,
        "condition_on_previous_text": True,
    }

    # IMPORTANT: use keyword args so types don't shift positions
    model = whisperx.load_model(
        whisper_arch=model_name,
        device=device,
        compute_type=compute_type,
        language=(language or None),
    )

    # Transcribe (batched like the reference)
    result = model.transcribe(audio, batch_size=16)

    # Robust: use "text" if present, else join segments
    text = result.get("text")
    if not text:
        segs = result.get("segments", []) or []
        text = " ".join((s.get("text", "") or "").strip() for s in segs).strip()

    return text


def transcribe_ct2(audio_path: str, model_name: str, language: str, beam_size: int, compute_type: str) -> str:
    """
    Use whisper-ctranslate2 CLI via subprocess (no Python import needed).
    Requires the package to be installed so the module entrypoint exists:
      uv pip install whisper-ctranslate2
    """
    import subprocess
    from pathlib import Path

    outdir = Path("./eval_ct2_tmp")
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "whisper-ctranslate2",
        str(audio_path),
        "--model", model_name,
        "--task", "transcribe",
        "--compute_type", compute_type,
        "--beam_size", str(beam_size),
        "--output_dir", str(outdir),
        "--threads", "16"
    ]
    if language:
        cmd += ["--language", language]

    # Inherit current env (so your thread limits from set_cpu_threads() apply)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-40:])
        raise RuntimeError(f"whisper-ctranslate2 CLI failed (exit {proc.returncode}). Log tail:\n{tail}")

    js = outdir / (Path(audio_path).stem + ".json")
    if js.exists():
        import json
        return json.loads(js.read_text(encoding="utf-8")).get("text","").strip()
    raise RuntimeError("CT2 produced no transcript file")



def transcribe_openai_whisper(audio_path: str, ow_model: str, language: str, beam_size: int) -> str:
    import whisper
    model = whisper.load_model(ow_model, device="cpu")
    result = model.transcribe(
        audio_path,
        language=language or None,
        task="transcribe",
        temperature=0.0,
        # beam_size=beam_size,
        condition_on_previous_text=True,
        fp16=False,
        verbose=False,
    )
    return result.get("text", "")

# ------------------- Eval runner -----------------------
def eval_backend(df: pd.DataFrame, backend: str, args) -> pd.DataFrame:
    rows = []
    for i, row in df.iterrows():
        audio = row["audio"]
        ref = row["reference"]
        lang = (row["language"] if "language" in df.columns else "") or args.language

        try:
            t0 = time.perf_counter()
            if backend == "transformers":
                hyp = transcribe_transformers(audio, args.hf_model, lang, args.beam_size)
            elif backend == "faster-whisper":
                hyp = transcribe_faster_whisper(audio, args.model, lang, args.beam_size, args.compute_type)
            elif backend == "whisperx":
                hyp = transcribe_whisperx(audio, args.model, lang, args.beam_size, args.compute_type)
            elif backend == "whisper-ctranslate2":
                hyp = transcribe_ct2(audio, args.model, lang, args.beam_size, args.compute_type)
            elif backend == "openai-whisper":
                hyp = transcribe_openai_whisper(audio, args.ow_model, lang, args.beam_size)
            else:
                raise ValueError(f"Unknown backend: {backend}")
            dt = time.perf_counter() - t0
        except Exception as e:
            hyp = f"[ERROR] {e}"
            dt = float("nan")

        rows.append({
            "index": i,
            "backend": backend,
            "audio": audio,
            "language": lang,
            "reference": ref,
            "hypothesis": hyp,
            "wer": compute_wer(ref, hyp),
            "cer": compute_cer(ref, hyp),
            "latency_s": dt,
        })
    return pd.DataFrame(rows)

def summarize(df: pd.DataFrame) -> dict:
    return {
        "samples": int(df.shape[0]),
        "wer_mean": float(df["wer"].mean() if not df.empty else float("nan")),
        "cer_mean": float(df["cer"].mean() if not df.empty else float("nan")),
        "latency_mean_s": float(df["latency_s"].mean() if not df.empty else float("nan")),
    }

# -------------------- CLI main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="CSV with columns: audio,reference[,language]")
    ap.add_argument("--language", default="id", help="Default language if dataset has no 'language' column")
    ap.add_argument("--model", default="large-v3", help="CT2-based backends model: tiny/base/small/medium/large-v3")
    ap.add_argument("--hf-model", default="openai/whisper-large-v3", help="Transformers model id")
    ap.add_argument("--ow-model", default="large-v2", help="OpenAI whisper model name (tiny/base/small/medium/large/large-v2)")
    ap.add_argument("--compute-type", default="float32", help="CT2 compute type: float32|int16|int8")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--backends", nargs="*", default=[
        "transformers", "faster-whisper", "whisperx", "whisper-ctranslate2", "openai-whisper"
    ])
    ap.add_argument("--outdir", default="./eval_outputs")
    args = ap.parse_args()

    set_cpu_threads(args.threads)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.dataset)
    for col in ["audio", "reference"]:
        if col not in df.columns:
            raise ValueError(f"Dataset missing required column: {col}")

    all_summaries = []
    for backend in args.backends:
        try:
            res_df = eval_backend(df, backend, args)
            (outdir / f"results_{backend}.csv").write_text(res_df.to_csv(index=False), encoding="utf-8")
            summ = summarize(res_df); summ["backend"] = backend
            with open(outdir / f"summary_{backend}.json", "w", encoding="utf-8") as f:
                json.dump(summ, f, indent=2)
            print(f"[{backend}] samples={summ['samples']}  WER={summ['wer_mean']:.4f}  CER={summ['cer_mean']:.4f}  latency_s={summ['latency_mean_s']:.3f}")
            all_summaries.append(summ)
        except Exception as e:
            print(f"[WARN] backend {backend} failed: {e}")

    if all_summaries:
        sum_df = pd.DataFrame(all_summaries)[["backend", "samples", "wer_mean", "cer_mean", "latency_mean_s"]]
        (outdir / "summary_all.csv").write_text(sum_df.to_csv(index=False), encoding="utf-8")
        print("\n=== Overall Summary ===")
        print(sum_df.to_string(index=False))

if __name__ == "__main__":
    main()
