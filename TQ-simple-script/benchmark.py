#!/usr/bin/env python3
"""
Parallel Processing Resource Scaling Benchmark
Implements Task 4 from CLAUDE.md - benchmarks transcription service scaling
"""
import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
import threading
import psutil
import wave
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import websockets

class ResourceMonitor:
    """Monitor CPU and memory usage during benchmarks."""

    def __init__(self, interval=1.0):
        self.interval = interval
        self.monitoring = False
        self.data = []
        self.monitor_thread = None

    def start_monitoring(self):
        """Start resource monitoring in background thread."""
        self.monitoring = True
        self.data = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop resource monitoring and return collected data."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        return self.data

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]

            self.data.append({
                'timestamp': time.time(),
                'cpu_percent': cpu_percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_percent': memory.percent,
                'load_avg_1min': load_avg[0] if load_avg else 0
            })
            time.sleep(self.interval)

class WhisperBenchmark:
    """Main benchmarking class for parallel Whisper transcription."""

    def __init__(self, audio_dir="../timit_eval", server_uri="ws://localhost:8765"):
        self.audio_dir = Path(audio_dir)
        self.server_uri = server_uri
        self.results = []
        self.monitor = ResourceMonitor()

        # Find available audio files
        self.audio_files = list(self.audio_dir.glob("*.wav"))[:50]  # Limit for testing
        if not self.audio_files:
            raise FileNotFoundError(f"No WAV files found in {self.audio_dir}")

        print(f"Found {len(self.audio_files)} audio files for testing")

    def get_wav_metadata(self, wav_path):
        """Extract metadata from WAV file."""
        try:
            with wave.open(str(wav_path), 'rb') as wav_file:
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                duration = frames / float(sample_rate)
                file_size = wav_path.stat().st_size

                return {
                    'duration': duration,
                    'sample_rate': sample_rate,
                    'file_size': file_size,
                    'frames': frames
                }
        except Exception as e:
            print(f"Error reading WAV metadata from {wav_path}: {e}")
            return None

    async def run_transcription_batch(self, num_files=10, timeout=300):
        """Run a batch of transcription requests and measure performance."""
        # Select files for this test
        test_files = self.audio_files[:num_files]

        print(f"Running transcription batch with {len(test_files)} files")

        # Collect audio metadata
        audio_durations = []
        audio_file_sizes = []
        total_audio_time = 0

        for wav_path in test_files:
            metadata = self.get_wav_metadata(wav_path)
            if metadata:
                audio_durations.append(metadata['duration'])
                audio_file_sizes.append(metadata['file_size'])
                total_audio_time += metadata['duration']

        start_time = time.perf_counter()
        results_received = 0

        try:
            async with websockets.connect(self.server_uri, timeout=30) as websocket:
                print(f"Connected to server at {self.server_uri}")

                # Send all requests
                for wav_path in test_files:
                    file_id = wav_path.stem

                    with open(wav_path, "rb") as f:
                        wav_data = f.read()

                    message = {
                        "file_id": file_id,
                        "data": wav_data.hex()
                    }
                    await websocket.send(json.dumps(message))
                    response = await websocket.recv()
                    print(f"Queued: {file_id}.wav")

                print(f"All {len(test_files)} requests sent, waiting for results...")

                # Collect results
                for i in range(len(test_files)):
                    try:
                        result = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                        results_received += 1
                        print(f"Result {results_received}/{len(test_files)}: {result[:100]}...")
                    except asyncio.TimeoutError:
                        print(f"Timeout waiting for result {i+1}")
                        break

        except Exception as e:
            print(f"Error during transcription batch: {e}")
            return None

        end_time = time.perf_counter()
        elapsed = end_time - start_time

        return {
            'num_files': len(test_files),
            'total_time': elapsed,
            'results_received': results_received,
            'success_rate': results_received / len(test_files) if test_files else 0,
            'avg_time_per_file': elapsed / len(test_files) if test_files else 0,
            'total_audio_time': total_audio_time,
            'processing_rate': total_audio_time / elapsed if elapsed > 0 else 0,
            'audio_durations': audio_durations,
            'audio_file_sizes': audio_file_sizes
        }

    def start_consumer_worker(self, pool_size=1, batch_size=5, threads_per_model=4):
        """Start the consumer worker with specified pool size."""
        # Check if running in Docker (producer service exists)
        rabbitmq_host = os.environ.get('RABBITMQ_HOST', 'localhost')

        if rabbitmq_host != 'localhost':
            # In Docker environment - consumer runs in same container
            cmd = [
                "uv", "run", "python", "/app/src/mp_consumer_worker.py",
                "--pool-size", str(pool_size),
                "--batch-size", str(batch_size),
                "--threads-per-model", str(threads_per_model)
            ]
        else:
            # Local environment
            cmd = [
                "uv", "run", "python", "src/mp_consumer_worker.py",
                "--pool-size", str(pool_size),
                "--batch-size", str(batch_size),
                "--threads-per-model", str(threads_per_model)
            ]

        print(f"Starting consumer worker: {' '.join(cmd)}")

        # Set environment variables for RabbitMQ connection
        env = os.environ.copy()
        env.update({
            'RABBITMQ_HOST': rabbitmq_host,
            'RABBITMQ_PORT': os.environ.get('RABBITMQ_PORT', '5672'),
        })

        # Start worker as subprocess
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env
        )

        # Give worker time to start
        time.sleep(3)

        return process

    def start_producer_server(self):
        """Start the WebSocket producer server."""
        # Check if running in Docker environment
        rabbitmq_host = os.environ.get('RABBITMQ_HOST', 'localhost')

        if rabbitmq_host != 'localhost':
            # In Docker environment - producer runs as separate service
            # No need to start it here, just wait for it to be available
            print("Producer service should already be running in Docker environment")
            time.sleep(2)
            return None
        else:
            # Local environment - start producer
            cmd = ["uv", "run", "python", "src/producer.py"]

            print(f"Starting producer server: {' '.join(cmd)}")

            # Set environment variables for RabbitMQ connection
            env = os.environ.copy()
            env.update({
                'RABBITMQ_HOST': rabbitmq_host,
                'RABBITMQ_PORT': os.environ.get('RABBITMQ_PORT', '5672'),
            })

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env
            )

            # Give server time to start
            time.sleep(2)

            return process

    async def run_baseline_test(self, num_files=10):
        """Run baseline test with single process (pool_size=1)."""
        print("\n" + "="*60)
        print("BASELINE TEST (Single Process)")
        print("="*60)

        # Start services
        producer_process = self.start_producer_server()
        consumer_process = self.start_consumer_worker(pool_size=1)

        try:
            # Start monitoring
            self.monitor.start_monitoring()

            # Run transcription batch
            result = await self.run_transcription_batch(num_files)

            # Stop monitoring
            resource_data = self.monitor.stop_monitoring()

            if result:
                result['pool_size'] = 1
                result['resource_data'] = resource_data
                result['test_type'] = 'baseline'

                # Calculate resource stats
                if resource_data:
                    result['max_cpu_percent'] = max(d['cpu_percent'] for d in resource_data)
                    result['avg_cpu_percent'] = statistics.mean(d['cpu_percent'] for d in resource_data)
                    result['max_memory_gb'] = max(d['memory_used_gb'] for d in resource_data)
                    result['avg_memory_gb'] = statistics.mean(d['memory_used_gb'] for d in resource_data)
                    result['max_load_avg'] = max(d['load_avg_1min'] for d in resource_data)

                self.results.append(result)
                self.print_test_result(result)

                return result
            else:
                print("Baseline test failed")
                return None

        finally:
            # Clean up processes
            consumer_process.terminate()
            if producer_process:
                producer_process.terminate()
            time.sleep(2)

            try:
                consumer_process.kill()
                if producer_process:
                    producer_process.kill()
            except:
                pass

    async def run_parallel_scaling_test(self, pool_sizes=[2, 4, 8, 16], num_files=20):
        """Run parallel scaling tests with different pool sizes."""
        print("\n" + "="*60)
        print("PARALLEL SCALING TESTS")
        print("="*60)

        for pool_size in pool_sizes:
            print(f"\nTesting pool size: {pool_size}")
            print("-" * 40)

            # Start services
            producer_process = self.start_producer_server()
            consumer_process = self.start_consumer_worker(pool_size=pool_size)

            try:
                # Start monitoring
                self.monitor.start_monitoring()

                # Run transcription batch
                result = await self.run_transcription_batch(num_files)

                # Stop monitoring
                resource_data = self.monitor.stop_monitoring()

                if result:
                    result['pool_size'] = pool_size
                    result['resource_data'] = resource_data
                    result['test_type'] = 'scaling'

                    # Calculate resource stats
                    if resource_data:
                        result['max_cpu_percent'] = max(d['cpu_percent'] for d in resource_data)
                        result['avg_cpu_percent'] = statistics.mean(d['cpu_percent'] for d in resource_data)
                        result['max_memory_gb'] = max(d['memory_used_gb'] for d in resource_data)
                        result['avg_memory_gb'] = statistics.mean(d['memory_used_gb'] for d in resource_data)
                        result['max_load_avg'] = max(d['load_avg_1min'] for d in resource_data)

                    self.results.append(result)
                    self.print_test_result(result)
                else:
                    print(f"Test failed for pool size {pool_size}")

            finally:
                # Clean up processes
                consumer_process.terminate()
                if producer_process:
                    producer_process.terminate()
                time.sleep(2)

                try:
                    consumer_process.kill()
                    if producer_process:
                        producer_process.kill()
                except:
                    pass

                # Wait between tests
                time.sleep(3)

    def print_test_result(self, result):
        """Print formatted test results."""
        print(f"\nRESULTS (Pool Size: {result['pool_size']}):")
        print(f"  Files processed: {result['results_received']}/{result['num_files']}")
        print(f"  Success rate: {result['success_rate']:.1%}")
        print(f"  Total time: {result['total_time']:.2f}s")
        print(f"  Avg time per file: {result['avg_time_per_file']:.2f}s")
        print(f"  Processing rate: {result['processing_rate']:.2f}x real-time")

        if 'max_cpu_percent' in result:
            print(f"  Peak CPU: {result['max_cpu_percent']:.1f}%")
            print(f"  Avg CPU: {result['avg_cpu_percent']:.1f}%")
            print(f"  Peak Memory: {result['max_memory_gb']:.2f}GB")
            print(f"  Peak Load Avg: {result['max_load_avg']:.2f}")

    def generate_report(self):
        """Generate comprehensive benchmark report."""
        if not self.results:
            print("No results to report")
            return

        print("\n" + "="*80)
        print("COMPREHENSIVE BENCHMARK REPORT")
        print("="*80)

        # System info
        print("\nSYSTEM INFORMATION:")
        print(f"  CPU Cores: {psutil.cpu_count()}")
        print(f"  Memory: {psutil.virtual_memory().total / (1024**3):.1f}GB")
        print(f"  Audio files tested: {len(self.audio_files)}")

        # Results summary table
        print(f"\n{'Pool Size':<10} {'Files':<8} {'Total Time':<12} {'Avg Time/File':<15} {'Processing Rate':<15} {'Peak CPU':<10} {'Peak Memory':<12}")
        print("-" * 100)

        for result in sorted(self.results, key=lambda x: x['pool_size']):
            pool_size = result['pool_size']
            files = result['results_received']
            total_time = result['total_time']
            avg_time = result['avg_time_per_file']
            proc_rate = result['processing_rate']
            peak_cpu = result.get('max_cpu_percent', 0)
            peak_mem = result.get('max_memory_gb', 0)

            print(f"{pool_size:<10} {files:<8} {total_time:<12.2f} {avg_time:<15.2f} {proc_rate:<15.2f} {peak_cpu:<10.1f}% {peak_mem:<12.2f}GB")

        # Find optimal configuration
        self.find_optimal_configuration()

        # Generate visualizations
        self.create_visualizations()

    def find_optimal_configuration(self):
        """Analyze results to find optimal configuration."""
        if len(self.results) < 2:
            return

        print("\n" + "="*60)
        print("OPTIMAL CONFIGURATION ANALYSIS")
        print("="*60)

        # Sort by pool size
        sorted_results = sorted(self.results, key=lambda x: x['pool_size'])

        # Find best throughput
        best_rate = max(r['processing_rate'] for r in sorted_results if r['success_rate'] > 0.8)
        best_config = next(r for r in sorted_results if r['processing_rate'] == best_rate)

        print(f"\nBEST PERFORMANCE:")
        print(f"  Pool Size: {best_config['pool_size']}")
        print(f"  Processing Rate: {best_config['processing_rate']:.2f}x real-time")
        print(f"  Total Time: {best_config['total_time']:.2f}s for {best_config['num_files']} files")
        print(f"  Resource Usage: {best_config.get('max_cpu_percent', 0):.1f}% CPU, {best_config.get('max_memory_gb', 0):.2f}GB RAM")

        # Efficiency analysis
        print(f"\nEFFICIENCY ANALYSIS:")
        baseline = next((r for r in sorted_results if r['pool_size'] == 1), None)
        if baseline:
            for result in sorted_results[1:]:  # Skip baseline
                speedup = baseline['total_time'] / result['total_time'] if result['total_time'] > 0 else 0
                efficiency = speedup / result['pool_size']
                print(f"  Pool Size {result['pool_size']:2d}: {speedup:.2f}x speedup, {efficiency:.2f} efficiency")

        # Recommendations
        print(f"\nRECOMMENDATIONS:")
        print(f"  Optimal pool size for this hardware: {best_config['pool_size']}")
        print(f"  Expected throughput: {best_config['processing_rate']:.1f}x real-time")

        if best_config['pool_size'] >= psutil.cpu_count():
            print(f"  Note: Optimal pool size equals or exceeds CPU cores ({psutil.cpu_count()})")
            print(f"  Consider horizontal scaling (more machines) for higher throughput")
        else:
            print(f"  Vertical scaling potential: Try pool sizes up to {psutil.cpu_count()} cores")

    def create_visualizations(self):
        """Create performance visualization plots."""
        if len(self.results) < 2:
            return

        # Sort results by pool size
        sorted_results = sorted(self.results, key=lambda x: x['pool_size'])

        pool_sizes = [r['pool_size'] for r in sorted_results]
        total_times = [r['total_time'] for r in sorted_results]
        processing_rates = [r['processing_rate'] for r in sorted_results]
        cpu_usage = [r.get('max_cpu_percent', 0) for r in sorted_results]
        memory_usage = [r.get('max_memory_gb', 0) for r in sorted_results]

        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Total time vs pool size
        ax1.plot(pool_sizes, total_times, 'b-o', linewidth=2, markersize=8)
        ax1.set_xlabel('Pool Size (Number of Processes)')
        ax1.set_ylabel('Total Processing Time (seconds)')
        ax1.set_title('Total Processing Time vs Pool Size')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Processing rate vs pool size
        ax2.plot(pool_sizes, processing_rates, 'g-o', linewidth=2, markersize=8)
        ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Real-time (1.0x)')
        ax2.set_xlabel('Pool Size (Number of Processes)')
        ax2.set_ylabel('Processing Rate (x real-time)')
        ax2.set_title('Processing Efficiency vs Pool Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: CPU usage vs pool size
        ax3.plot(pool_sizes, cpu_usage, 'r-o', linewidth=2, markersize=8)
        ax3.set_xlabel('Pool Size (Number of Processes)')
        ax3.set_ylabel('Peak CPU Usage (%)')
        ax3.set_title('CPU Usage vs Pool Size')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Memory usage vs pool size
        ax4.plot(pool_sizes, memory_usage, 'm-o', linewidth=2, markersize=8)
        ax4.set_xlabel('Pool Size (Number of Processes)')
        ax4.set_ylabel('Peak Memory Usage (GB)')
        ax4.set_title('Memory Usage vs Pool Size')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"whisper_benchmark_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved as: {plot_filename}")

        # Save data as CSV
        df = pd.DataFrame([{
            'pool_size': r['pool_size'],
            'total_time': r['total_time'],
            'processing_rate': r['processing_rate'],
            'peak_cpu_percent': r.get('max_cpu_percent', 0),
            'peak_memory_gb': r.get('max_memory_gb', 0),
            'success_rate': r['success_rate'],
            'files_processed': r['results_received']
        } for r in sorted_results])

        csv_filename = f"whisper_benchmark_data_{timestamp}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"Data saved as: {csv_filename}")

async def main():
    """Main benchmarking function."""
    parser = argparse.ArgumentParser(description='Whisper Parallel Processing Benchmark')
    parser.add_argument('--test-type', choices=['baseline', 'scaling', 'full'], default='full',
                       help='Type of benchmark to run')
    parser.add_argument('--pool-sizes', nargs='+', type=int, default=[1, 2, 4, 8],
                       help='Pool sizes to test (default: 1 2 4 8)')
    parser.add_argument('--num-files', type=int, default=20,
                       help='Number of files to test per configuration')
    parser.add_argument('--audio-dir', default='../timit_eval',
                       help='Directory containing audio files')
    parser.add_argument('--server-uri', default='ws://localhost:8765',
                       help='WebSocket server URI')

    args = parser.parse_args()

    print("Whisper Parallel Processing Benchmark")
    print("=" * 50)
    print(f"Test type: {args.test_type}")
    print(f"Pool sizes: {args.pool_sizes}")
    print(f"Files per test: {args.num_files}")
    print(f"Audio directory: {args.audio_dir}")

    try:
        benchmark = WhisperBenchmark(args.audio_dir, args.server_uri)

        if args.test_type in ['baseline', 'full']:
            await benchmark.run_baseline_test(args.num_files)

        if args.test_type in ['scaling', 'full']:
            scaling_sizes = [p for p in args.pool_sizes if p > 1]
            if scaling_sizes:
                await benchmark.run_parallel_scaling_test(scaling_sizes, args.num_files)

        benchmark.generate_report()

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"Benchmark error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())