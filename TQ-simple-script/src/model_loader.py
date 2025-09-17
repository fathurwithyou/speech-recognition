import os
import time
import threading
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

try:
    import mlx_whisper
    MLX_WHISPER_AVAILABLE = True
except ImportError:
    MLX_WHISPER_AVAILABLE = False
    print("MLX Whisper not available. Install with: pip install mlx-whisper")



class WhisperModelLoader:
    """Simple loader/runner for MLX Whisper.

    Uses MLX Whisper for Apple Silicon optimized inference.
    """

    def __init__(self, base_model: str = "mlx-community/whisper-large-v2"):
        self.base_model = base_model
        self.model_info: Dict[str, Any] = {}
        self.load_lock = threading.Lock()
        self.is_loaded = False

    def load_model(self, force_reload: bool = False) -> bool:
        with self.load_lock:
            if self.is_loaded and not force_reload:
                return True

            if not MLX_WHISPER_AVAILABLE:
                print("MLX Whisper not available. Install with: pip install mlx-whisper")
                return False

            try:
                load_start = time.time()
                print(f"Loading MLX Whisper model '{self.base_model}'...")

                # MLX Whisper doesn't require explicit model loading - it loads on first use
                # We'll validate the model exists by doing a quick test
                try:
                    # Test if model is accessible (this will download if needed)
                    mlx_whisper.load_model(self.base_model)
                except Exception as e:
                    print(f"Failed to load model '{self.base_model}': {e}")
                    return False

                self.model_info = {
                    'load_time': time.time() - load_start,
                    'model_type': 'MLX Whisper',
                    'base_model': self.base_model,
                    'device': 'mps'  # MLX uses Metal Performance Shaders
                }
                self.is_loaded = True
                print(f"MLX Whisper model loaded in {self.model_info['load_time']:.2f}s")
                return True

            except Exception as e:
                e = str(e)[:500]  # Truncate long errors
                print(f"Failed to load MLX Whisper model: {e}")
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
            t0 = time.perf_counter()

            # Use MLX Whisper transcribe function
            result = mlx_whisper.transcribe(
                audio_path,
                path_or_hf_repo=self.base_model,
                language=language if language != "en" else None,  # Let it auto-detect if English
                verbose=False,
            )

            text = (result.get("text") or "").strip()
            inference_time = time.perf_counter() - t0

            return {
                'transcription': text,
                'confidence': self._estimate_confidence(result),
                'model_used': f'mlx_whisper_{self.base_model.replace("/", "_")}',
                'inference_time': inference_time,
                'language': result.get('language', language),
                'segments': len(result.get('segments', [])) if result.get('segments') else 0
            }

        except Exception as e:
            print(f"MLX Whisper inference error for {file_id}: {e}")
            return {
                'transcription': f'MLX_WHISPER_ERROR: {str(e)[:80]}',
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
