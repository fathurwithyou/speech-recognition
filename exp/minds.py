from datasets import load_dataset
import pandas as pd
from pathlib import Path
import soundfile as sf

def export_timit(out_csv="timit_eval.csv", limit=100):
    ds = load_dataset("PolyAI/MINDs14", 'en-US', split="train[:300]")

    rows = []
    outdir = Path(out_csv).with_suffix("")
    outdir.mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        audio_path = outdir / f"{i}.wav"
        sf.write(audio_path, row["audio"]["array"], row["audio"]["sampling_rate"])

        print(row)

        rows.append({
            "audio": str(audio_path),
            "reference": row["transcription"],
            "language": "en"
        })

    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    print(f"✅ Saved {len(rows)} samples to {out_csv}")

# contoh pemakaian
export_timit(out_csv="timit_eval.csv", limit=200)
