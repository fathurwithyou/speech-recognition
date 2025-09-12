import os
from typing import Optional

def get_timit_file_path(file_id: str, base_path: str = "timit_eval") -> Optional[str]:
    """Get full path to TIMIT file by ID."""
    file_path = os.path.join(base_path, f"{file_id}.wav")
    if os.path.exists(file_path):
        return file_path
    return None

def model_based_speech_to_text(audio_path: str, file_id: str, use_model: bool = True, process_id: int = 0, use_onnx: bool = False) -> dict:
    """Model-based speech-to-text using loaded Whisper model.

    Fallback has been removed; failures raise exceptions for the caller to handle.
    """
    if not use_model:
        raise RuntimeError("Model inference disabled and no fallback available")

    try:
        from model_loader import get_model_loader, load_model_if_needed
        if not load_model_if_needed(process_id, use_onnx=use_onnx):
            model_type = "ONNX" if use_onnx else "PyTorch"
            raise RuntimeError(f"{model_type} model not loaded for process {process_id}")
        loader = get_model_loader(process_id, use_onnx=use_onnx)
        return loader.infer_audio(audio_path, file_id, language="en")
    except Exception as e:
        raise RuntimeError(f"Model inference error: {e}")
