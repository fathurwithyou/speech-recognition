#!/usr/bin/env python3
"""
Benchmark the OpenAI speech recognition model performance.
Tests model loading, inference speed, and accuracy patterns.
"""

import sys
import os
import time
import json
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model_loader import OpenAIModelLoader
from audio_utils import load_audio_file, extract_mfcc_features, get_timit_file_path

class ModelBenchmark:
    """Benchmark suite for OpenAI speech recognition model."""
    
    def __init__(self, model_path="../exp/quant_outputs_openai/openai_model_int8_sd.pt"):
        self.model_path = model_path
        self.loader = OpenAIModelLoader(model_path)
        self.results = []
        
    def test_model_loading(self):
        """Test model loading performance."""
        print("🔄 Testing model loading...")
        
        # Test initial load
        start_time = time.time()
        success = self.loader.load_model()
        load_time = time.time() - start_time
        
        result = {
            'test': 'model_loading',
            'success': success,
            'load_time': load_time,
            'timestamp': datetime.now().isoformat()
        }
        
        if success:
            model_info = self.loader.get_model_info()
            result.update(model_info)
            
            print(f"✅ Model loaded successfully in {load_time:.3f}s")
            print(f"   Model size: {model_info.get('file_size_mb', 'unknown'):.2f} MB")
            print(f"   Device: {model_info.get('device', 'unknown')}")
        else:
            print(f"❌ Model loading failed")
        
        # Test reload performance
        if success:
            print("🔄 Testing model reload...")
            start_time = time.time()
            self.loader.load_model(force_reload=True)
            reload_time = time.time() - start_time
            result['reload_time'] = reload_time
            print(f"✅ Model reload: {reload_time:.3f}s")
        
        self.results.append(result)
        return result
    
    def test_inference_speed(self, num_samples=10):
        """Test inference speed with different input sizes."""
        print(f"\n🚀 Testing inference speed with {num_samples} samples...")
        
        if not self.loader.is_loaded:
            print("❌ Model not loaded")
            return None
        
        # Test different input sizes
        input_sizes = [
            (10, 2),    # Small: 10 frames, 2 features
            (50, 2),    # Medium: 50 frames, 2 features  
            (100, 2),   # Large: 100 frames, 2 features
            (200, 2),   # Very large: 200 frames, 2 features
        ]
        
        speed_results = []
        
        for size_name, (frames, features) in zip(['small', 'medium', 'large', 'very_large'], input_sizes):
            print(f"   Testing {size_name} input ({frames}x{features})...")
            
            times = []
            for i in range(num_samples):
                # Generate random audio features
                dummy_features = np.random.randn(frames, features)
                
                start_time = time.time()
                result = self.loader.infer_audio(dummy_features, f"test_{i}")
                inference_time = time.time() - start_time
                times.append(inference_time)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            
            size_result = {
                'input_size': size_name,
                'frames': frames,
                'features': features,
                'num_samples': num_samples,
                'avg_time': avg_time,
                'std_time': std_time,
                'min_time': min_time,
                'max_time': max_time,
                'throughput': 1.0 / avg_time if avg_time > 0 else 0
            }
            
            speed_results.append(size_result)
            
            print(f"     Average: {avg_time:.4f}s ± {std_time:.4f}s")
            print(f"     Range: {min_time:.4f}s - {max_time:.4f}s")
            print(f"     Throughput: {size_result['throughput']:.1f} inferences/sec")
        
        result = {
            'test': 'inference_speed',
            'speed_results': speed_results,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results.append(result)
        return result
    
    def test_real_audio_inference(self, num_files=5):
        """Test inference on real TIMIT audio files."""
        print(f"\n🎵 Testing real audio inference with {num_files} TIMIT files...")
        
        if not self.loader.is_loaded:
            print("❌ Model not loaded")
            return None
        
        # Get first N TIMIT files
        test_files = [str(i) for i in range(min(num_files, 20))]
        audio_results = []
        
        total_audio_duration = 0
        total_inference_time = 0
        
        for file_id in test_files:
            try:
                # Load and process audio file
                file_path = get_timit_file_path(file_id, base_path="timit_eval")
                if not file_path:
                    print(f"⚠️ File {file_id}.wav not found, skipping...")
                    continue
                
                print(f"   Processing {file_id}.wav...")
                
                # Load audio
                audio_start = time.time()
                audio_data, sample_rate = load_audio_file(file_path)
                features = extract_mfcc_features(audio_data, sample_rate)
                feature_time = time.time() - audio_start
                
                # Run inference
                inference_start = time.time()
                result = self.loader.infer_audio(features, file_id)
                inference_time = time.time() - inference_start
                
                # Calculate metrics
                audio_duration = len(audio_data) / sample_rate
                total_audio_duration += audio_duration
                total_inference_time += inference_time
                
                # Real-time factor (inference time / audio duration)
                rtf = inference_time / audio_duration if audio_duration > 0 else float('inf')
                
                file_result = {
                    'file_id': file_id,
                    'audio_duration': audio_duration,
                    'num_features': len(features),
                    'feature_extraction_time': feature_time,
                    'inference_time': inference_time,
                    'total_processing_time': feature_time + inference_time,
                    'real_time_factor': rtf,
                    'transcription': result['transcription'],
                    'confidence': result.get('confidence', 0.0),
                    'model_used': result.get('model_used', 'unknown')
                }
                
                audio_results.append(file_result)
                
                print(f"     Duration: {audio_duration:.2f}s, Inference: {inference_time:.3f}s, RTF: {rtf:.2f}")
                print(f"     Model: {result.get('model_used', 'unknown')}, Confidence: {result.get('confidence', 0):.3f}")
                
            except Exception as e:
                print(f"❌ Error processing {file_id}: {e}")
        
        if audio_results:
            # Calculate aggregate metrics
            avg_rtf = np.mean([r['real_time_factor'] for r in audio_results if r['real_time_factor'] != float('inf')])
            avg_confidence = np.mean([r['confidence'] for r in audio_results])
            total_rtf = total_inference_time / total_audio_duration if total_audio_duration > 0 else float('inf')
            
            print(f"\n📊 Real audio results:")
            print(f"   Files processed: {len(audio_results)}")
            print(f"   Total audio duration: {total_audio_duration:.2f}s")
            print(f"   Total inference time: {total_inference_time:.3f}s")
            print(f"   Average real-time factor: {avg_rtf:.3f}")
            print(f"   Overall real-time factor: {total_rtf:.3f}")
            print(f"   Average confidence: {avg_confidence:.3f}")
            
            result = {
                'test': 'real_audio_inference',
                'files_processed': len(audio_results),
                'total_audio_duration': total_audio_duration,
                'total_inference_time': total_inference_time,
                'avg_real_time_factor': avg_rtf,
                'overall_real_time_factor': total_rtf,
                'avg_confidence': avg_confidence,
                'audio_results': audio_results,
                'timestamp': datetime.now().isoformat()
            }
        else:
            result = {
                'test': 'real_audio_inference',
                'error': 'No files processed successfully',
                'timestamp': datetime.now().isoformat()
            }
        
        self.results.append(result)
        return result
    
    def test_model_memory_usage(self):
        """Test model memory usage patterns."""
        print(f"\n💾 Testing model memory usage...")
        
        try:
            import psutil
            process = psutil.Process()
            
            # Memory before model loading
            self.loader.unload_model()
            time.sleep(1)  # Let garbage collection happen
            memory_before = process.memory_info().rss / (1024 * 1024)  # MB
            
            # Memory after model loading
            self.loader.load_model()
            time.sleep(1)
            memory_after = process.memory_info().rss / (1024 * 1024)  # MB
            
            memory_usage = memory_after - memory_before
            
            print(f"   Memory before loading: {memory_before:.1f} MB")
            print(f"   Memory after loading: {memory_after:.1f} MB")
            print(f"   Model memory usage: {memory_usage:.1f} MB")
            
            result = {
                'test': 'memory_usage',
                'memory_before_mb': memory_before,
                'memory_after_mb': memory_after,
                'model_memory_usage_mb': memory_usage,
                'timestamp': datetime.now().isoformat()
            }
            
        except ImportError:
            print("   psutil not available for memory testing")
            result = {
                'test': 'memory_usage',
                'error': 'psutil not available',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"   Memory test error: {e}")
            result = {
                'test': 'memory_usage',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        
        self.results.append(result)
        return result
    
    def run_full_benchmark(self):
        """Run complete benchmark suite."""
        print("🎯 OpenAI Model Benchmark Suite")
        print("=" * 50)
        print(f"Model path: {self.model_path}")
        
        # Test 1: Model loading
        loading_result = self.test_model_loading()
        
        if loading_result and loading_result['success']:
            # Test 2: Inference speed
            self.test_inference_speed(num_samples=5)
            
            # Test 3: Real audio inference
            self.test_real_audio_inference(num_files=5)
            
            # Test 4: Memory usage
            self.test_model_memory_usage()
        
        return self.results
    
    def print_summary(self):
        """Print benchmark summary."""
        print(f"\n{'='*60}")
        print("📊 BENCHMARK SUMMARY")
        print(f"{'='*60}")
        
        for result in self.results:
            test_name = result['test']
            print(f"\n🔸 {test_name.upper()}")
            
            if test_name == 'model_loading':
                if result['success']:
                    print(f"   ✅ Load time: {result['load_time']:.3f}s")
                    if 'reload_time' in result:
                        print(f"   ✅ Reload time: {result['reload_time']:.3f}s")
                    print(f"   📦 Model size: {result.get('file_size_mb', 0):.2f} MB")
                    print(f"   🖥️  Device: {result.get('device', 'unknown')}")
                else:
                    print(f"   ❌ Loading failed")
            
            elif test_name == 'inference_speed':
                print(f"   Speed results:")
                for speed in result['speed_results']:
                    print(f"     {speed['input_size']}: {speed['avg_time']:.4f}s avg "
                          f"({speed['throughput']:.1f} fps)")
            
            elif test_name == 'real_audio_inference':
                if 'error' not in result:
                    print(f"   📁 Files processed: {result['files_processed']}")
                    print(f"   🎵 Audio duration: {result['total_audio_duration']:.2f}s")
                    print(f"   ⚡ Overall RTF: {result['overall_real_time_factor']:.3f}")
                    print(f"   🎯 Avg confidence: {result['avg_confidence']:.3f}")
                else:
                    print(f"   ❌ {result['error']}")
            
            elif test_name == 'memory_usage':
                if 'error' not in result:
                    print(f"   💾 Model memory: {result['model_memory_usage_mb']:.1f} MB")
                else:
                    print(f"   ❌ {result['error']}")
    
    def save_results(self, filename=None):
        """Save benchmark results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_benchmark_{timestamp}.json"
        
        benchmark_data = {
            'benchmark_info': {
                'model_path': self.model_path,
                'timestamp': datetime.now().isoformat(),
                'total_tests': len(self.results)
            },
            'results': self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(benchmark_data, f, indent=2)
        
        print(f"\n💾 Benchmark results saved to: {filename}")
        return filename

def main():
    """Main benchmark execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='OpenAI Model Benchmark')
    parser.add_argument('--model-path', default="../exp/quant_outputs_openai/openai_model_int8_sd.pt",
                       help='Path to model file')
    parser.add_argument('--quick', action='store_true', help='Quick benchmark with fewer samples')
    parser.add_argument('--inference-only', action='store_true', help='Only test inference speed')
    
    args = parser.parse_args()
    
    benchmark = ModelBenchmark(args.model_path)
    
    if args.inference_only:
        # Only test inference speed
        benchmark.test_model_loading()
        if benchmark.loader.is_loaded:
            samples = 3 if args.quick else 5
            benchmark.test_inference_speed(num_samples=samples)
    elif args.quick:
        # Quick benchmark
        print("🚀 Running quick model benchmark...")
        benchmark.test_model_loading()
        if benchmark.loader.is_loaded:
            benchmark.test_inference_speed(num_samples=3)
            benchmark.test_real_audio_inference(num_files=3)
    else:
        # Full benchmark
        benchmark.run_full_benchmark()
    
    # Show results
    benchmark.print_summary()
    
    # Save results
    if benchmark.results:
        benchmark.save_results()
        print("\n🎉 Model benchmark completed!")
    else:
        print("\n❌ Model benchmark failed!")

if __name__ == "__main__":
    main()