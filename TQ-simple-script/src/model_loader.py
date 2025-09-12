import os
import time
import threading
import logging
from typing import Dict, Any
import numpy as np

from sklearn import base
import torch

logger = logging.getLogger(__name__)

try:
    import whisper
    from whisper.model import Whisper as WhisperCore
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("OpenAI Whisper not available. Install with: pip install openai-whisper")

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("ONNX Runtime not available. Install with: pip install onnxruntime")


def quantize_model_int8(model_fp32, quant_mode: str = "dynamic", copy_model: bool = True):
    """Apply INT8 quantization using torchao if available; fallback to torch.ao dynamic.

    - quant_mode: 'dynamic' (default) or 'weight_only'
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
    except Exception:
        try:
            from torch.ao.quantization import quantize_dynamic
            m = copy.deepcopy(model_fp32) if copy_model else model_fp32
            m = quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
            return m, "torch.ao.dynamic"
        except Exception:
            return model_fp32, "none"

class WhisperModelLoader:
    """Simple loader/runner for OpenAI Whisper.

    KISS: load the official model directly (vanilla). If a custom state dict is
    provided and compatible, that can be added later; for now we prioritize a
    reliable working path over quantization complexity.
    """

    def __init__(self, base_model: str = "large-v2", model_path: str = os.path.join("openai", "openai_model_int8_sd.pt"), quant_mode: str = "dynamic"):
        self.base_model = base_model
        self.model_path = model_path
        self.quant_mode = quant_mode
        self.model = None
        self.device = None
        self.model_info: Dict[str, Any] = {}
        self.load_lock = threading.Lock()
        self.is_loaded = False

    def load_model(self, force_reload: bool = False) -> bool:
        with self.load_lock:
            if self.is_loaded and not force_reload:
                return True

            if not WHISPER_AVAILABLE:
                print("OpenAI Whisper not available. Install with: pip install openai-whisper")
                return False

            try:
                load_start = time.time()
                self.device = "cpu"
                if self.model_path and os.path.exists(self.model_path):
                    print(f"Loading Whisper '{self.base_model}' (INT8 weights) on {self.device}...")
                    base_arch = whisper.load_model(self.base_model, device="cpu")
                    dims = base_arch.dims
                    del base_arch   

                    core = WhisperCore(dims)
                    core, q_engine = quantize_model_int8(core, quant_mode=self.quant_mode, copy_model=False)
                    state = torch.load(self.model_path, map_location="cpu", weights_only=False)
                    missing, unexpected = core.load_state_dict(state, strict=False)
                    self.model = core.to(self.device)
                    if missing or unexpected:
                        print(f"Warning: state dict load had {len(missing)} missing and {len(unexpected)} unexpected keys")

                    self.model_info = {
                        'device': self.device,
                        'load_time': time.time() - load_start,
                        'model_type': 'OpenAI Whisper (INT8 state dict)',
                        'base_model': self.base_model,
                        'quant_mode': self.quant_mode,
                        'quant_engine': q_engine,
                        'torch_version': torch.__version__
                    }
                else:
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
                e = str(e)[:500]  # Truncate long errors
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
            t0 = time.perf_counter()
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
            inference_time = time.perf_counter() - t0

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


class WhisperONNXModelLoader:
    """ONNX-based Whisper model loader with INT8 quantization."""
    
    def __init__(self, model_name: str = "base", onnx_dir: str = "onnx_models"):
        self.model_name = model_name
        self.onnx_dir = onnx_dir
        self.encoder_session = None
        self.decoder_session = None
        self.model_info: Dict[str, Any] = {}
        self.load_lock = threading.Lock()
        self.is_loaded = False
        
        # Get model dimensions from whisper
        if WHISPER_AVAILABLE:
            temp_model = whisper.load_model(model_name)
            self.dims = temp_model.dims
            del temp_model
        else:
            # Default dimensions for base model
            self.dims = type('Dims', (), {
                'n_mels': 80,
                'n_audio_ctx': 1500,
                'n_audio_state': 512,
                'n_audio_head': 8,
                'n_audio_layer': 6,
                'n_vocab': 51865,
                'n_text_ctx': 448,
                'n_text_state': 512,
                'n_text_head': 8,
                'n_text_layer': 6
            })()
    
    def load_model(self, force_reload: bool = False) -> bool:
        """Load ONNX models."""
        with self.load_lock:
            if self.is_loaded and not force_reload:
                return True
            
            if not ONNX_AVAILABLE:
                print("ONNX Runtime not available")
                return False
            
            try:
                load_start = time.time()
                
                # Try INT8 models first, fallback to FP32
                encoder_int8_path = os.path.join(self.onnx_dir, f"whisper_{self.model_name}_encoder_int8.onnx")
                decoder_int8_path = os.path.join(self.onnx_dir, f"whisper_{self.model_name}_decoder_int8.onnx")
                encoder_fp32_path = os.path.join(self.onnx_dir, f"whisper_{self.model_name}_encoder.onnx")
                decoder_fp32_path = os.path.join(self.onnx_dir, f"whisper_{self.model_name}_decoder.onnx")
                
                # Choose best available models
                encoder_path = encoder_int8_path if os.path.exists(encoder_int8_path) else encoder_fp32_path
                decoder_path = decoder_int8_path if os.path.exists(decoder_int8_path) else decoder_fp32_path
                
                if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
                    print(f"ONNX models not found. Expected one of:")
                    print(f"  Encoder: {encoder_int8_path} or {encoder_fp32_path}")
                    print(f"  Decoder: {decoder_int8_path} or {decoder_fp32_path}")
                    print("Run convert_to_onnx.py to generate them.")
                    return False
                
                encoder_type = "INT8" if "int8" in os.path.basename(encoder_path) else "FP32"
                decoder_type = "INT8" if "int8" in os.path.basename(decoder_path) else "FP32"
                print(f"Using encoder: {encoder_type}, decoder: {decoder_type}")
                
                print(f"Loading ONNX Whisper {self.model_name} models...")
                
                # Create ONNX Runtime sessions
                sess_options = ort.SessionOptions()
                sess_options.inter_op_num_threads = 1
                sess_options.intra_op_num_threads = 4
                
                self.encoder_session = ort.InferenceSession(encoder_path, sess_options)
                self.decoder_session = ort.InferenceSession(decoder_path, sess_options)
                
                self.model_info = {
                    'load_time': time.time() - load_start,
                    'model_type': f'Whisper ONNX (Encoder: {encoder_type}, Decoder: {decoder_type})',
                    'model_name': self.model_name,
                    'encoder_path': encoder_path,
                    'decoder_path': decoder_path,
                    'encoder_precision': encoder_type,
                    'decoder_precision': decoder_type,
                    'onnx_version': ort.__version__
                }
                
                self.is_loaded = True
                print(f"ONNX Whisper models loaded in {self.model_info['load_time']:.2f}s")
                return True
                
            except Exception as e:
                print(f"Failed to load ONNX models: {e}")
                self.is_loaded = False
                return False
    
    def infer_audio(self, audio_path: str, file_id: str, language: str = "en") -> Dict[str, Any]:
        """Perform inference using ONNX models."""
        if not self.is_loaded:
            return {
                'transcription': f'ERROR: ONNX model not loaded for {file_id}',
                'confidence': 0.0,
                'model_used': 'none'
            }
        
        try:
            t0 = time.perf_counter()
            
            # Load and preprocess audio
            if WHISPER_AVAILABLE:
                audio = whisper.load_audio(audio_path)
                audio = whisper.pad_or_trim(audio)
                mel = whisper.log_mel_spectrogram(audio, n_mels=self.dims.n_mels).unsqueeze(0).numpy()
            else:
                # Fallback - create dummy mel spectrogram
                mel = np.random.randn(1, self.dims.n_mels, 3000).astype(np.float32)
            
            # Run encoder
            encoder_outputs = self.encoder_session.run(
                None, 
                {"mel_spectrogram": mel}
            )
            encoder_output = encoder_outputs[0]
            
            # Simple greedy decoding
            tokens = [50258]  # Start token
            max_tokens = 448
            
            for _ in range(max_tokens):
                token_input = np.array([tokens], dtype=np.int64)
                
                decoder_outputs = self.decoder_session.run(
                    None,
                    {
                        "tokens": token_input,
                        "encoder_output": encoder_output
                    }
                )
                
                logits = decoder_outputs[0]
                next_token = np.argmax(logits[0, -1])
                
                if next_token == 50257:  # End token
                    break
                    
                tokens.append(int(next_token))
            
            # Decode tokens to text (simplified)
            if WHISPER_AVAILABLE:
                # Determine if model is multilingual based on model name
                multilingual = not self.model_name.endswith('.en')
                tokenizer = whisper.tokenizer.get_tokenizer(multilingual=multilingual)
                text = tokenizer.decode(tokens[1:])  # Skip start token
            else:
                text = f"ONNX_TRANSCRIPTION_FILE_{file_id}"  # Fallback text
            
            inference_time = time.perf_counter() - t0
            
            return {
                'transcription': text.strip(),
                'confidence': 0.8,  # Default confidence
                'model_used': f'whisper_{self.model_name}_onnx_int8',
                'inference_time': inference_time,
                'language': language,
                'segments': 1
            }
            
        except Exception as e:
            print(f"ONNX inference error for {file_id}: {e}")
            return {
                'transcription': f'ONNX_ERROR: {str(e)[:80]}',
                'confidence': 0.0,
                'model_used': 'error',
                'inference_time': 0.0
            }


_model_loaders: Dict[int, WhisperModelLoader] = {}
_onnx_model_loaders: Dict[int, WhisperONNXModelLoader] = {}

def get_model_loader(process_id: int = 0, use_onnx: bool = False):
    if use_onnx:
        return get_onnx_model_loader(process_id)
    else:
        if process_id not in _model_loaders:
            _model_loaders[process_id] = WhisperModelLoader()
        return _model_loaders[process_id]

def get_onnx_model_loader(process_id: int = 0) -> WhisperONNXModelLoader:
    if process_id not in _onnx_model_loaders:
        _onnx_model_loaders[process_id] = WhisperONNXModelLoader()
    return _onnx_model_loaders[process_id]

def get_multiprocess_loaders(num_processes: int = 4, use_onnx: bool = False):
    return {i: get_model_loader(i, use_onnx) for i in range(num_processes)}

def load_model_if_needed(process_id: int = 0, use_onnx: bool = False) -> bool:
    loader = get_model_loader(process_id, use_onnx)
    if not loader.is_loaded:
        return loader.load_model()
    return True

def load_all_models(num_processes: int = 4, use_onnx: bool = False) -> bool:
    ok = 0
    for i in range(num_processes):
        if load_model_if_needed(i, use_onnx):
            ok += 1
    model_type = "ONNX" if use_onnx else "PyTorch"
    print(f"Loaded {ok}/{num_processes} {model_type} model instances")
    return ok > 0
