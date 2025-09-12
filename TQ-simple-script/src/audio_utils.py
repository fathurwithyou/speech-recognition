import os
import wave
import numpy as np
from typing import Tuple, Optional

def load_audio_file(file_path: str) -> Tuple[np.ndarray, int]:
    """Load WAV audio file and return audio data and sample rate."""
    try:
        with wave.open(file_path, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            audio_data = wav_file.readframes(n_frames)
            
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Normalize to [-1, 1]
            audio_array = audio_array.astype(np.float32) / 32768.0
            
            return audio_array, sample_rate
    except Exception as e:
        raise ValueError(f"Failed to load audio file {file_path}: {e}")

def extract_mfcc_features(audio_data: np.ndarray, sample_rate: int, n_mfcc: int = 13) -> np.ndarray:
    """Extract MFCC features from audio data (simplified version)."""
    # Simple energy-based features as a placeholder
    # In a real implementation, you'd use librosa or similar
    frame_length = int(0.025 * sample_rate)  # 25ms frames
    hop_length = int(0.010 * sample_rate)    # 10ms hop
    
    features = []
    for i in range(0, len(audio_data) - frame_length, hop_length):
        frame = audio_data[i:i + frame_length]
        # Simple energy feature
        energy = np.mean(frame ** 2)
        # Simple spectral centroid approximation
        weighted_freq = np.mean(frame * np.arange(len(frame)))
        features.append([energy, weighted_freq])
    
    return np.array(features)

def get_timit_file_path(file_id: str, base_path: str = "timit_eval") -> Optional[str]:
    """Get full path to TIMIT file by ID."""
    file_path = os.path.join(base_path, f"{file_id}.wav")
    if os.path.exists(file_path):
        return file_path
    return None

def model_based_speech_to_text(audio_path: str, file_id: str, use_model: bool = True, process_id: int = 0) -> dict:
    """Model-based speech-to-text using loaded Whisper model."""
    if use_model:
        try:
            from model_loader import get_model_loader, load_model_if_needed
            
            # Ensure model is loaded for this process
            if load_model_if_needed(process_id):
                loader = get_model_loader(process_id)
                result = loader.infer_audio(audio_path, file_id, language="en")
                return result
            else:
                print(f"⚠️ Model loading failed for {file_id}, using fallback")
        except ImportError as e:
            print(f"⚠️ Model loader not available: {e}")
        except Exception as e:
            print(f"⚠️ Model inference error: {e}")
    
    # Fallback to energy-based transcription
    return fallback_speech_to_text_from_path(audio_path, file_id)

def fallback_speech_to_text_from_path(audio_path: str, file_id: str) -> dict:
    """Fallback transcription when model is unavailable (from audio path)."""
    try:
        # Load audio to get basic characteristics
        audio_data, sample_rate = load_audio_file(audio_path)
        features = extract_mfcc_features(audio_data, sample_rate)
        energy_mean = np.mean(features[:, 0]) if len(features) > 0 else 0
    except:
        energy_mean = 0.5  # Default if audio loading fails
    
    # Simple heuristic based on energy patterns
    if energy_mean > 0.01:
        words = ["HELLO", "WORLD", "SPEECH", "RECOGNITION", "TEST"]
        word_idx = int(file_id) % len(words)
        transcription = f"FALLBACK_{file_id}: {words[word_idx]} SAMPLE"
        confidence = min(0.8, energy_mean * 10)
    else:
        transcription = f"FALLBACK_{file_id}: LOW ENERGY AUDIO"
        confidence = 0.3
    
    return {
        'transcription': transcription,
        'confidence': confidence,
        'model_used': 'fallback_heuristic',
        'inference_time': 0.001  # Very fast fallback
    }

def fallback_speech_to_text(audio_features: np.ndarray, file_id: str) -> dict:
    """Fallback rule-based speech-to-text when model is unavailable."""
    energy_mean = np.mean(audio_features[:, 0]) if len(audio_features) > 0 else 0
    
    # Simple heuristic based on energy patterns
    if energy_mean > 0.01:
        words = ["HELLO", "WORLD", "SPEECH", "RECOGNITION", "TEST"]
        word_idx = int(file_id) % len(words)
        transcription = f"FALLBACK_{file_id}: {words[word_idx]} SAMPLE"
        confidence = min(0.8, energy_mean * 10)
    else:
        transcription = f"FALLBACK_{file_id}: LOW ENERGY AUDIO"
        confidence = 0.3
    
    return {
        'transcription': transcription,
        'confidence': confidence,
        'model_used': 'fallback_heuristic',
        'inference_time': 0.001  # Very fast fallback
    }

def simple_speech_to_text(audio_features: np.ndarray, file_id: str) -> str:
    """Legacy function retained for compatibility.

    Uses the heuristic fallback on precomputed features instead of the model,
    because the model-based path expects an audio file path, not feature arrays.
    """
    result = fallback_speech_to_text(audio_features, file_id)
    return result['transcription']
