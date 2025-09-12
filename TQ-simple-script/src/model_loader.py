import os
import time
import threading
import logging
from typing import Dict, Any

import torch

logger = logging.getLogger(__name__)

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("OpenAI Whisper not available. Install with: pip install openai-whisper")


def quantize_model_int8(model_fp32, quant_mode: str = "dynamic", copy_model: bool = True):
    """Placeholder for quantization; returns the model unchanged and engine='none'."""
    return model_fp32, "none"


class WhisperModelLoader:
    """Simple loader/runner for OpenAI Whisper.

    KISS: load the official model directly (vanilla). If a custom state dict is
    provided and compatible, that can be added later; for now we prioritize a
    reliable working path over quantization complexity.
    """

    def __init__(self, base_model: str = "small"):
        self.base_model = base_model
        self.model = None
        self.device = None
        self.model_info: Dict[str, Any] = {}
        self.load_lock = threading.Lock()
        self.is_loaded = False

    def detect_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def load_model(self, force_reload: bool = False) -> bool:
        with self.load_lock:
            if self.is_loaded and not force_reload:
                return True

            if not WHISPER_AVAILABLE:
                print("OpenAI Whisper not available. Install with: pip install openai-whisper")
                return False

            try:
                load_start = time.time()
                self.device = self.detect_device()
                print(f"Loading Whisper '{self.base_model}' on {self.device}...")
                self.model = whisper.load_model(self.base_model, device=self.device)
                self.model.eval()

                self.model_info = {
                    'device': self.device,
                    'load_time': time.time() - load_start,
                    'model_type': 'OpenAI Whisper (Vanilla)',
                    'base_model': self.base_model,
                    'quant_mode': 'none',
                    'quant_engine': 'none',
                    'torch_version': torch.__version__
                }

                self.is_loaded = True
                print(f"Whisper model loaded in {self.model_info['load_time']:.2f}s")
                return True

            except Exception as e:
                print(f"Failed to load Whisper model: {e}")
                logger.error(f"Model loading error: {e}")
                self.is_loaded = False
                return False

    def infer_audio(self, audio_path: str, file_id: str, language: str = "en") -> Dict[str, Any]:
        if not self.is_loaded:
            return {
                'transcription': f'ERROR: Model not loaded for {file_id}',
                'confidence': 0.0,
                'model_used': 'none'
            }

        try:
            t0 = time.time()
            with torch.no_grad():
                result = self.model.transcribe(
                    audio_path,
                    language=(language or None),
                    task="transcribe",
                    condition_on_previous_text=True,
                    fp16=False,
                    verbose=False,
                )

            text = (result.get("text") or "").strip()
            inference_time = time.time() - t0

            return {
                'transcription': text,
                'confidence': self._estimate_confidence(result),
                'model_used': f'whisper_{self.base_model}',
                'inference_time': inference_time,
                'language': result.get('language', language),
                'segments': len(result.get('segments', [])) if result.get('segments') else 0
            }

        except Exception as e:
            print(f"Whisper inference error for {file_id}: {e}")
            return {
                'transcription': f'WHISPER_ERROR: {str(e)[:80]}',
                'confidence': 0.0,
                'model_used': 'error',
                'inference_time': 0.0
            }

    def _estimate_confidence(self, result: dict) -> float:
        try:
            segs = result.get('segments') or []
            vals = []
            for s in segs:
                if 'avg_logprob' in s:
                    vals.append(max(0.0, min(1.0, s['avg_logprob'] + 1.0)))
                elif 'no_speech_prob' in s:
                    vals.append(1.0 - float(s['no_speech_prob']))
            if vals:
                return sum(vals) / len(vals)
            return 0.6 if (result.get('text') or '').strip() else 0.1
        except Exception:
            return 0.5


_model_loaders: Dict[int, WhisperModelLoader] = {}

def get_model_loader(process_id: int = 0) -> WhisperModelLoader:
    if process_id not in _model_loaders:
        _model_loaders[process_id] = WhisperModelLoader()
    return _model_loaders[process_id]

def get_multiprocess_loaders(num_processes: int = 4) -> Dict[int, WhisperModelLoader]:
    return {i: get_model_loader(i) for i in range(num_processes)}

def load_model_if_needed(process_id: int = 0) -> bool:
    loader = get_model_loader(process_id)
    if not loader.is_loaded:
        return loader.load_model()
    return True


def load_all_models(num_processes: int = 4) -> bool:
    ok = 0
    for i in range(num_processes):
        if load_model_if_needed(i):
            ok += 1
    print(f"Loaded {ok}/{num_processes} model instances")
    return ok > 0

