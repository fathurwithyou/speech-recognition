import asyncio
import websockets
import json
import time
import subprocess
import signal
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import pika

class HybridBenchmark:
    """Benchmark comparing different RabbitMQ worker configurations."""
    
    def __init__(self, websocket_url="ws://localhost:8765"):
        self.websocket_url = websocket_url
        self.worker_processes = []
        
    def start_producer_if_needed(self):
        """Start producer if not running."""
        try:
            # Check if producer is running
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', 8765))
            sock.close()
            
            if result == 0:
                print("✓ Producer already running")
                return None
            else:
                print("⏳ Starting producer...")
                process = subprocess.Popen([
                    'python', 'src/producer.py'
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                time.sleep(3)  # Give producer time to start
                return process
                
        except Exception as e:
            print(f"Error checking/starting producer: {e}")
            return None
    
    def start_workers(self, worker_config):
        """Start worker processes based on configuration."""
        processes = []
        
        if worker_config['type'] == 'single':
            # Single-threaded workers
            for i in range(worker_config['num_workers']):
                print(f"Starting single-threaded worker {i+1}/{worker_config['num_workers']}")
                process = subprocess.Popen([
                    'python', 'src/worker.py'
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                processes.append(process)
                time.sleep(0.5)
                
        elif worker_config['type'] == 'multiprocessing':
            # Multiprocessing workers
            for i in range(worker_config['num_workers']):
                print(f"Starting multiprocessing worker {i+1}/{worker_config['num_workers']} "
                      f"(pool size: {worker_config['pool_size']})")
                process = subprocess.Popen([
                    'python', 'src/mp_consumer_worker.py',
                    '--pool-size', str(worker_config['pool_size']),
                    '--batch-size', str(worker_config.get('batch_size', 5)),
                    '--batch-timeout', str(worker_config.get('batch_timeout', 2.0))
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                processes.append(process)
                time.sleep(1.0)
        
        print(f"✓ Started {len(processes)} worker processes")
        return processes
    
    def stop_workers(self, processes):
        """Stop worker processes."""
        print("🔄 Stopping worker processes...")
        
        for process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            except:
                pass
        
        print(f"✅ Stopped {len(processes)} worker processes")
    
    async def run_benchmark_test(self, file_ids, test_name):
        """Run a single benchmark test."""
        print(f"\n🎯 Running {test_name}")
        print(f"   Files to process: {len(file_ids)}")
        
        start_time = time.time()
        
        try:
            async with websockets.connect(self.websocket_url, timeout=10) as websocket:
                # Send all requests
                print("📤 Sending requests...")
                for file_id in file_ids:
                    await websocket.send(file_id)
                    response = await websocket.recv()  # Acknowledgment
                
                # Collect results
                print("📥 Collecting results...")
                results = []
                for i in range(len(file_ids)):
                    result = await asyncio.wait_for(websocket.recv(), timeout=120)
                    results.append({
                        'raw_result': result,
                        'received_at': time.time()
                    })
                    
                    if (i + 1) % 10 == 0 or (i + 1) == len(file_ids):
                        print(f"   Received: {i + 1}/{len(file_ids)}")
                
                end_time = time.time()
                total_time = end_time - start_time
                
                # Parse results
                successful_tasks = []
                failed_tasks = []
                
                for result in results:
                    try:
                        if ": " in result['raw_result']:
                            json_str = result['raw_result'].split(": ", 1)[1]
                            data = json.loads(json_str)
                            if 'error' not in data:
                                successful_tasks.append(data)
                            else:
                                failed_tasks.append(data)
                    except:
                        failed_tasks.append({'error': 'Parse error'})
                
                return {
                    'test_name': test_name,
                    'total_time': total_time,
                    'successful_tasks': len(successful_tasks),
                    'failed_tasks': len(failed_tasks),
                    'throughput': len(successful_tasks) / total_time if total_time > 0 else 0,
                    'avg_processing_time': sum(t.get('processing_time', 0) for t in successful_tasks) / len(successful_tasks) if successful_tasks else 0
                }
                
        except asyncio.TimeoutError:
            print(f"❌ Benchmark timed out")
            return None
        except ConnectionRefusedError:
            print(f"❌ Could not connect to WebSocket server")
            return None
        except Exception as e:
            print(f"❌ Benchmark error: {e}")
            return None
    
    def run_comparison_benchmark(self, file_ids):
        """Run comprehensive comparison benchmark."""
        print("🔬 HYBRID WORKER COMPARISON BENCHMARK")
        print("=" * 60)
        
        # Test configurations
        test_configs = [
            {
                'name': 'Single Worker (Baseline)',
                'type': 'single',
                'num_workers': 1
            },
            {
                'name': '2 Single Workers',
                'type': 'single',
                'num_workers': 2
            },
            {
                'name': '4 Single Workers',
                'type': 'single',
                'num_workers': 4
            },
            {
                'name': '1 MP Worker (4 processes)',
                'type': 'multiprocessing',
                'num_workers': 1,
                'pool_size': 4,
                'batch_size': 5,
                'batch_timeout': 1.0
            },
            {
                'name': '2 MP Workers (2 processes each)',
                'type': 'multiprocessing',
                'num_workers': 2,
                'pool_size': 2,
                'batch_size': 3,
                'batch_timeout': 1.0
            },
            {
                'name': '1 MP Worker (8 processes)',
                'type': 'multiprocessing',
                'num_workers': 1,
                'pool_size': 8,
                'batch_size': 8,
                'batch_timeout': 0.5
            }
        ]
        
        # Start producer
        producer_process = self.start_producer_if_needed()
        
        results = []
        
        try:
            for config in test_configs:
                print(f"\n{'='*60}")
                print(f"🧪 Testing: {config['name']}")
                print(f"{'='*60}")
                
                # Start workers for this configuration
                worker_processes = self.start_workers(config)
                
                # Give workers time to start
                print("⏳ Waiting for workers to initialize...")
                time.sleep(5)
                
                # Run benchmark
                result = asyncio.run(self.run_benchmark_test(file_ids, config['name']))
                
                if result:
                    result['config'] = config
                    results.append(result)
                    
                    print(f"✅ {config['name']} completed:")
                    print(f"   Total time: {result['total_time']:.2f}s")
                    print(f"   Throughput: {result['throughput']:.2f} tasks/sec")
                    print(f"   Success rate: {result['successful_tasks']}/{result['successful_tasks'] + result['failed_tasks']}")
                else:
                    print(f"❌ {config['name']} failed")
                
                # Stop workers
                self.stop_workers(worker_processes)
                
                # Brief pause between tests
                time.sleep(3)
                
        finally:
            # Clean up producer
            if producer_process:
                try:
                    producer_process.terminate()
                    producer_process.wait(timeout=5)
                except:
                    producer_process.kill()
        
        return results
    
    def print_comparison_results(self, results):
        """Print comparison results in a nice format."""
        if not results:
            print("❌ No results to display")
            return
        
        print(f"\n{'='*80}")
        print("🏆 HYBRID BENCHMARK RESULTS")
        print(f"{'='*80}")
        
        # Table header
        print(f"{'Configuration':<30} {'Time(s)':<10} {'Success':<8} {'Failed':<8} {'Throughput':<12} {'Avg Proc':<10}")
        print("-" * 80)
        
        # Sort by throughput (best first)
        sorted_results = sorted(results, key=lambda x: x['throughput'], reverse=True)
        
        for result in sorted_results:
            print(f"{result['test_name']:<30} {result['total_time']:<10.2f} {result['successful_tasks']:<8} "
                  f"{result['failed_tasks']:<8} {result['throughput']:<12.2f} {result['avg_processing_time']:<10.2f}")
        
        # Analysis
        best_throughput = sorted_results[0]
        fastest_time = min(results, key=lambda x: x['total_time'])
        
        print(f"\n📊 ANALYSIS:")
        print(f"🚀 Highest throughput: {best_throughput['test_name']} - {best_throughput['throughput']:.2f} tasks/sec")
        print(f"⚡ Fastest completion: {fastest_time['test_name']} - {fastest_time['total_time']:.2f} seconds")
        
        # Compare multiprocessing vs single-threaded
        mp_results = [r for r in results if 'MP' in r['test_name']]
        single_results = [r for r in results if 'Single' in r['test_name']]
        
        if mp_results and single_results:
            best_mp = max(mp_results, key=lambda x: x['throughput'])
            best_single = max(single_results, key=lambda x: x['throughput'])
            
            improvement = ((best_mp['throughput'] - best_single['throughput']) / best_single['throughput']) * 100
            
            if improvement > 0:
                print(f"🎯 Best multiprocessing worker is {improvement:.1f}% faster than best single worker")
            else:
                print(f"🎯 Best single worker setup is {abs(improvement):.1f}% faster than multiprocessing")
        
        return sorted_results

def check_system_ready():
    """Check if RabbitMQ is ready."""
    try:
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host='localhost', port=5672)
        )
        connection.close()
        print("✓ RabbitMQ is ready")
        return True
    except:
        print("❌ RabbitMQ not available")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Hybrid RabbitMQ Worker Benchmark')
    parser.add_argument('--files', type=int, default=30, help='Number of test files to process')
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer configurations')
    
    args = parser.parse_args()
    
    # Check system
    if not check_system_ready():
        print("Please start RabbitMQ:")
        print("docker run --name my-rabbitmq -p 5672:5672 -p 15672:15672 -d rabbitmq:3-management")
        exit(1)
    
    # Generate test files
    test_files = [str(i) for i in range(min(args.files, 200))]
    
    benchmark = HybridBenchmark()
    
    if args.quick:
        print("🚀 Running quick benchmark...")
        test_files = test_files[:10]  # Only 10 files for quick test
    
    results = benchmark.run_comparison_benchmark(test_files)
    benchmark.print_comparison_results(results)
    
    # Save results
    if results:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"hybrid_benchmark_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'test_files_count': len(test_files),
                'results': results
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")