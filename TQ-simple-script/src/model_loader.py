import torch
import os
import time
import threading
import logging
from typing import Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

# Import Whisper components
try:
    import whisper
    from whisper.model import Whisper as WhisperCore
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️ OpenAI Whisper not available. Install with: pip install openai-whisper")

def quantize_model_int8(model_fp32, quant_mode: str = "dynamic", copy_model: bool = True):
    """Quantize model using TorchAO (same as in quantize_openai_whisper.py)."""
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
        from warnings import warn
        warn(f"TorchAO unavailable, falling back to torch.ao dynamic INT8: {e}")
        from torch.ao.quantization import quantize_dynamic
        m = copy.deepcopy(model_fp32) if copy_model else model_fp32
        m = quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
        return m, "torch.ao.dynamic (deprecated)"

class WhisperModelLoader:
    """Loader and manager for OpenAI Whisper quantized model."""
    
    def __init__(self, 
                 model_path: str = "../exp/quant_outputs_openai/openai_model_int8_sd.pt",
                 base_model: str = "small",
                 quant_mode: str = "dynamic"):
        self.model_path = model_path
        self.base_model = base_model
        self.quant_mode = quant_mode
        self.model = None
        self.device = None
        self.model_info = {}
        self.load_lock = threading.Lock()
        self.is_loaded = False
        
        # Whisper-specific configuration
        self.model_dims = None
        
    def detect_device(self) -> str:
        """Detect the best available device for inference."""
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
        else:
            device = "cpu"
            logger.info("Using CPU for inference")
        return device
    
    def load_model(self, force_reload: bool = False) -> bool:
        """Load the OpenAI Whisper quantized model."""
        with self.load_lock:
            if self.is_loaded and not force_reload:
                return True
                
            if not WHISPER_AVAILABLE:
                print("❌ OpenAI Whisper not available. Install with: pip install openai-whisper")
                return False
                
            try:
                print(f"🔄 Loading OpenAI Whisper model from: {self.model_path}")
                load_start = time.time()
                
                # Check if model state dict file exists
                if not os.path.exists(self.model_path):
                    print(f"❌ Model file not found: {self.model_path}")
                    return False
                
                # Get file size
                file_size = os.path.getsize(self.model_path) / (1024 * 1024)  # MB
                print(f"   Model file size: {file_size:.2f} MB")
                
                # Detect device
                self.device = self.detect_device()
                
                print(f"   Loading on device: {self.device}")
                
                # Step 1: Load a base Whisper model to get dimensions
                print(f"   Loading base {self.base_model} model for architecture...")
                base_model = whisper.load_model(self.base_model, device="cpu")
                self.model_dims = base_model.dims
                
                # Step 2: Create blank Whisper model with same architecture
                print("   Creating blank Whisper architecture...")
                self.model = WhisperCore(self.model_dims)
                
                # Step 3: Apply quantization to the blank model (in-place)
                print(f"   Applying {self.quant_mode} quantization...")
                self.model, quant_engine = quantize_model_int8(
                    self.model, 
                    quant_mode=self.quant_mode, 
                    copy_model=False  # in-place for efficiency
                )
                
                # Step 4: Load the quantized state dict
                print("   Loading quantized weights...")
                try:
                    # Try weights_only=False first (trusted model)
                    state_dict = torch.load(self.model_path, map_location=self.device, weights_only=False)
                    print("   ✅ State dict loaded (trusted mode)")
                except Exception as e:
                    print(f"   ⚠️  Trusted loading failed, trying secure mode: {e}")
                    # Fallback to secure loading
                    try:
                        state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
                        print("   ✅ State dict loaded (secure mode)")
                    except Exception as e2:
                        print(f"   ❌ Both loading methods failed: {e2}")
                        return False
                
                # Step 5: Load state dict into model
                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                if missing_keys:
                    print(f"   ⚠️  Missing keys: {len(missing_keys)}")
                if unexpected_keys:
                    print(f"   ⚠️  Unexpected keys: {len(unexpected_keys)}")
                
                # Step 6: Move to device and set to eval mode
                self.model = self.model.to(self.device)
                self.model.eval()
                
                # Store model info
                self.model_info = {
                    'file_size_mb': file_size,
                    'device': self.device,
                    'load_time': time.time() - load_start,
                    'model_type': 'OpenAI Whisper (Quantized)',
                    'base_model': self.base_model,
                    'quant_mode': self.quant_mode,
                    'quant_engine': quant_engine,
                    'torch_version': torch.__version__
                }
                
                self.is_loaded = True
                
                print(f"✅ Whisper model loaded successfully in {self.model_info['load_time']:.2f}s")
                print(f"   Base model: {self.base_model}")
                print(f"   Quantization: {quant_engine}")
                print(f"   Device: {self.device}")
                
                # Clean up base model
                del base_model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                return True
                
            except Exception as e:
                print(f"❌ Failed to load model: {e}")
                logger.error(f"Model loading error: {e}")
                self.is_loaded = False
                return False
    
    def infer_audio(self, audio_path: str, file_id: str, language: str = "en") -> Dict[str, Any]:
        """Run Whisper inference on audio file."""
        if not self.is_loaded:
            return {
                'transcription': f'ERROR: Model not loaded for {file_id}',
                'confidence': 0.0,
                'model_used': 'none'
            }
        
        try:
            inference_start = time.time()
            
            # Use Whisper's transcribe method (same as in quantize script)
            with torch.no_grad():
                result = self.model.transcribe(
                    audio_path,
                    language=(language or None),
                    task="transcribe",
                    condition_on_previous_text=True,
                    fp16=False,
                    verbose=False,
                )
            
            transcription = (result.get("text") or "").strip()
            inference_time = time.time() - inference_start
            
            # Calculate confidence from Whisper result
            confidence = self._calculate_whisper_confidence(result)
            
            return {
                'transcription': transcription,
                'confidence': confidence,
                'model_used': f'whisper_{self.base_model}_quantized',
                'inference_time': inference_time,
                'language': result.get('language', language),
                'segments': len(result.get('segments', [])) if result.get('segments') else 0
            }
            
        except Exception as e:
            print(f"❌ Whisper inference error for {file_id}: {e}")
            return {
                'transcription': f'WHISPER_ERROR: {str(e)[:50]}',
                'confidence': 0.0,
                'model_used': 'error',
                'inference_time': 0.0
            }
    
    def _calculate_whisper_confidence(self, result: dict) -> float:
        """Calculate confidence score from Whisper result."""
        try:
            segments = result.get('segments', [])
            if not segments:
                return 0.5  # Default confidence if no segments
            
            # Average confidence from segments (if available)
            confidences = []
            for segment in segments:
                # Some Whisper results have token-level scores
                if 'avg_logprob' in segment:
                    # Convert log probability to confidence (approximate)
                    logprob = segment['avg_logprob']
                    confidence = min(1.0, max(0.0, (logprob + 1.0)))  # Normalize roughly
                    confidences.append(confidence)
                elif 'no_speech_prob' in segment:
                    # Use inverse of no-speech probability
                    no_speech = segment['no_speech_prob']
                    confidence = 1.0 - no_speech
                    confidences.append(confidence)
            
            if confidences:
                return sum(confidences) / len(confidences)
            else:
                # Fallback: estimate based on text length and quality
                text = result.get('text', '')
                if len(text) > 10:
                    return 0.8
                elif len(text) > 0:
                    return 0.6
                else:
                    return 0.1
                    
        except Exception:
            return 0.5  # Default confidence
    
    def _process_model_output(self, output: torch.Tensor, file_id: str) -> str:
        """Process model output to generate transcription."""
        try:
            if output.dim() >= 2:
                # If output has multiple dimensions, process accordingly
                if output.shape[-1] > 1000:  # Likely vocabulary logits
                    # Get top predictions
                    _, top_indices = torch.topk(output, 5)
                    # Create a reasonable transcription based on indices
                    words = self._indices_to_words(top_indices.flatten()[:5])
                    return f"MODEL_OUTPUT_{file_id}: {' '.join(words)}"
                else:
                    # Smaller output - might be embeddings or features
                    values = output.flatten()[:5]
                    return f"EMBEDDING_{file_id}: Features [{', '.join(f'{v:.3f}' for v in values)}]"
            else:
                # Single value output
                value = float(output.item())
                return f"PREDICTION_{file_id}: Value {value:.3f}"
                
        except Exception as e:
            return f"OUTPUT_PARSE_ERROR_{file_id}: {str(e)[:30]}"
    
    def _indices_to_words(self, indices: torch.Tensor) -> list:
        """Convert token indices to words (simplified vocabulary)."""
        # Simple vocabulary for demonstration
        vocab = [
            "hello", "world", "speech", "recognition", "audio",
            "sound", "voice", "word", "text", "model",
            "the", "and", "of", "to", "in", "is", "it", "you", "that", "he"
        ]
        
        words = []
        for idx in indices:
            vocab_idx = int(idx) % len(vocab)
            words.append(vocab[vocab_idx])
        
        return words[:3]  # Return first 3 words
    
    def _calculate_confidence(self, output: torch.Tensor) -> float:
        """Calculate confidence score from model output."""
        try:
            if output.dim() >= 2 and output.shape[-1] > 1:
                # For classification outputs, use softmax entropy
                probs = torch.softmax(output, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum()
                confidence = 1.0 / (1.0 + entropy.item())
                return min(0.95, max(0.1, confidence))
            else:
                # For other outputs, use magnitude
                magnitude = torch.abs(output).mean().item()
                return min(0.9, magnitude / 10.0)
        except:
            return 0.7  # Default confidence
    
    def _energy_based_transcription(self, energy: float, file_id: str) -> str:
        """Generate transcription based on audio energy characteristics."""
        if energy > 0.1:
            intensity = "HIGH"
        elif energy > 0.01:
            intensity = "MEDIUM"
        else:
            intensity = "LOW"
        
        # Generate realistic transcription based on energy
        transcriptions = {
            "HIGH": ["LOUD SPEECH DETECTED", "STRONG AUDIO SIGNAL", "CLEAR VOICE INPUT"],
            "MEDIUM": ["NORMAL SPEECH LEVEL", "MODERATE AUDIO INPUT", "STANDARD VOICE"],
            "LOW": ["QUIET AUDIO SIGNAL", "LOW VOLUME SPEECH", "WEAK AUDIO INPUT"]
        }
        
        import random
        options = transcriptions[intensity]
        selected = random.choice(options)
        
        return f"ENERGY_{intensity}_{file_id}: {selected}"
    
    def _fallback_transcription(self, file_id: str) -> str:
        """Fallback transcription when model cannot be used."""
        fallback_phrases = [
            "FALLBACK TRANSCRIPTION",
            "MODEL UNAVAILABLE",
            "PROCESSING COMPLETE",
            "AUDIO ANALYZED",
            "SPEECH DETECTED"
        ]
        
        import random
        phrase = random.choice(fallback_phrases)
        return f"{phrase}_{file_id}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'is_loaded': self.is_loaded,
            'model_path': self.model_path,
            'device': self.device,
            **self.model_info
        }
    
    def unload_model(self):
        """Unload the model to free memory."""
        with self.load_lock:
            if self.model is not None:
                del self.model
                self.model = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.is_loaded = False
            print("🔄 Model unloaded and memory cleared")

# Global model loader instances for multiprocessing
_model_loaders = {}
_max_processes = 4

def get_model_loader(process_id: int = 0) -> WhisperModelLoader:
    """Get a model loader instance for a specific process."""
    global _model_loaders
    
    if process_id not in _model_loaders:
        _model_loaders[process_id] = WhisperModelLoader()
    
    return _model_loaders[process_id]

def get_multiprocess_loaders(num_processes: int = 4) -> Dict[int, WhisperModelLoader]:
    """Get multiple model loader instances for multiprocessing."""
    loaders = {}
    for i in range(num_processes):
        loaders[i] = get_model_loader(i)
    return loaders

def load_model_if_needed(process_id: int = 0) -> bool:
    """Load model if not already loaded for a specific process."""
    loader = get_model_loader(process_id)
    if not loader.is_loaded:
        return loader.load_model()
    return True

def load_all_models(num_processes: int = 4) -> bool:
    """Load models for all processes."""
    print(f"🔄 Loading {num_processes} Whisper model instances...")
    success_count = 0
    
    for i in range(num_processes):
        print(f"Loading model for process {i+1}/{num_processes}...")
        if load_model_if_needed(i):
            success_count += 1
        else:
            print(f"❌ Failed to load model for process {i}")
    
    print(f"✅ Loaded {success_count}/{num_processes} model instances")
    return success_count > 0

if __name__ == "__main__":
    # Test model loading
    print("🧪 Testing model loader...")
    
    loader = WhisperModelLoader()
    success = loader.load_model()
    
    if success:
        print("✅ Model loading test passed")
        
        # Test inference with dummy audio path
        dummy_audio_path = "/path/to/test.wav"  # Placeholder
        result = loader.infer_audio(dummy_audio_path, "test")
        
        print(f"📊 Test inference result:")
        print(f"   Transcription: {result['transcription']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Model used: {result['model_used']}")
        print(f"   Inference time: {result['inference_time']:.3f}s")
        
        # Show model info
        info = loader.get_model_info()
        print(f"\n📋 Model info:")
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        loader.unload_model()
    else:
        print("❌ Model loading test failed")