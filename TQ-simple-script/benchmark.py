#!/usr/bin/env python3
"""
Comprehensive benchmark comparing RabbitMQ vs Multiprocessing queues for speech recognition.
"""

import time
import json
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from mp_worker import run_multiprocessing_benchmark
from rabbitmq_benchmark import run_rabbitmq_benchmark, check_rabbitmq_system

def save_benchmark_results(results, filename=None):
    """Save benchmark results to JSON file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {filename}")
    return filename

def compare_results(mp_results, rabbitmq_result):
    """Compare and analyze benchmark results."""
    print(f"\n{'='*80}")
    print("BENCHMARK COMPARISON")
    print(f"{'='*80}")
    
    if not rabbitmq_result:
        print("⚠️  RabbitMQ benchmark failed - cannot compare")
        return
    
    print(f"{'Method':<15} {'Workers':<8} {'Time(s)':<8} {'Success':<8} {'Failed':<8} {'Throughput':<12} {'Avg Proc':<10}")
    print("-" * 80)
    
    # RabbitMQ results
    print(f"{'RabbitMQ':<15} {'N/A':<8} {rabbitmq_result['total_time']:<8.2f} "
          f"{rabbitmq_result['successful_tasks']:<8} {rabbitmq_result['failed_tasks']:<8} "
          f"{rabbitmq_result['throughput']:<12.2f} {rabbitmq_result['avg_processing_time']:<10.2f}")
    
    # Multiprocessing results
    for result in mp_results:
        print(f"{'Multiprocess':<15} {result['num_workers']:<8} {result['total_time']:<8.2f} "
              f"{result['successful_tasks']:<8} {result['failed_tasks']:<8} "
              f"{result['throughput']:<12.2f} {result['avg_processing_time']:<10.2f}")
    
    # Find best performing configurations
    all_results = [rabbitmq_result] + mp_results
    best_throughput = max(all_results, key=lambda x: x['throughput'])
    fastest_total = min(all_results, key=lambda x: x['total_time'])
    
    print(f"\n📊 ANALYSIS:")
    print(f"🚀 Highest throughput: {best_throughput['method']} "
          f"({best_throughput.get('num_workers', 'N/A')} workers) - {best_throughput['throughput']:.2f} tasks/sec")
    print(f"⚡ Fastest total time: {fastest_total['method']} "
          f"({fastest_total.get('num_workers', 'N/A')} workers) - {fastest_total['total_time']:.2f} seconds")
    
    # Efficiency analysis
    rabbitmq_throughput = rabbitmq_result['throughput']
    mp_throughputs = [r['throughput'] for r in mp_results]
    
    if mp_throughputs:
        best_mp_throughput = max(mp_throughputs)
        if rabbitmq_throughput > best_mp_throughput:
            improvement = ((rabbitmq_throughput - best_mp_throughput) / best_mp_throughput) * 100
            print(f"🏆 RabbitMQ is {improvement:.1f}% faster than best multiprocessing config")
        else:
            improvement = ((best_mp_throughput - rabbitmq_throughput) / rabbitmq_throughput) * 100
            print(f"🏆 Best multiprocessing is {improvement:.1f}% faster than RabbitMQ")

def run_full_benchmark():
    """Run comprehensive benchmark comparing both approaches."""
    print("🎯 SPEECH RECOGNITION QUEUE BENCHMARK")
    print("=" * 80)
    print("Comparing RabbitMQ distributed queue vs Python multiprocessing")
    print("=" * 80)
    
    # Test configuration
    test_files = [str(i) for i in range(30)]  # Test with 30 TIMIT files
    worker_configs = [1, 2, 4, 8]  # Different worker counts for multiprocessing
    
    print(f"📁 Testing with {len(test_files)} TIMIT audio files")
    print(f"⚙️  Multiprocessing worker configs: {worker_configs}")
    
    all_results = []
    
    # 1. Run multiprocessing benchmarks
    print(f"\n🔄 Running multiprocessing benchmarks...")
    mp_results = []
    
    for num_workers in worker_configs:
        print(f"\nTesting multiprocessing with {num_workers} workers...")
        result = run_multiprocessing_benchmark(test_files, num_workers=num_workers)
        if result:
            mp_results.append(result)
            all_results.append(result)
        time.sleep(2)  # Brief pause between tests
    
    # 2. Run RabbitMQ benchmark
    print(f"\n🔄 Running RabbitMQ benchmark...")
    
    if not check_rabbitmq_system():
        print("\n❌ RabbitMQ system not ready. Please start:")
        print("   1. docker run --name my-rabbitmq -p 5672:5672 -p 15672:15672 -d rabbitmq:3-management")
        print("   2. python src/producer.py")
        print("   3. python src/worker.py")
        rabbitmq_result = None
    else:
        rabbitmq_result = run_rabbitmq_benchmark(test_files)
        if rabbitmq_result:
            all_results.append(rabbitmq_result)
    
    # 3. Compare results
    if mp_results or rabbitmq_result:
        compare_results(mp_results, rabbitmq_result)
        
        # Save results
        benchmark_data = {
            'timestamp': datetime.now().isoformat(),
            'test_files_count': len(test_files),
            'multiprocessing_results': mp_results,
            'rabbitmq_result': rabbitmq_result,
            'summary': {
                'total_tests': len(all_results),
                'best_throughput': max(all_results, key=lambda x: x['throughput'])['throughput'] if all_results else 0
            }
        }
        
        filename = save_benchmark_results(benchmark_data)
        return filename
    else:
        print("\n❌ No successful benchmarks completed")
        return None

def run_quick_benchmark():
    """Run a quick benchmark with fewer files."""
    print("🚀 QUICK BENCHMARK (10 files)")
    test_files = [str(i) for i in range(10)]
    
    # Quick multiprocessing test
    mp_result = run_multiprocessing_benchmark(test_files, num_workers=4)
    
    # Quick RabbitMQ test (if available)
    if check_rabbitmq_system():
        rabbitmq_result = run_rabbitmq_benchmark(test_files)
        if mp_result and rabbitmq_result:
            compare_results([mp_result], rabbitmq_result)
    else:
        print("RabbitMQ system not available for comparison")

if __name__ == "__main__":
    print("Speech Recognition Queue Benchmark")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            run_quick_benchmark()
        elif sys.argv[1] == "mp-only":
            # Multiprocessing only
            test_files = [str(i) for i in range(20)]
            for workers in [1, 2, 4, 8]:
                run_multiprocessing_benchmark(test_files, num_workers=workers)
        elif sys.argv[1] == "rabbitmq-only":
            # RabbitMQ only
            if check_rabbitmq_system():
                test_files = [str(i) for i in range(20)]
                run_rabbitmq_benchmark(test_files)
            else:
                print("RabbitMQ system not ready")
        else:
            print("Usage: python benchmark.py [quick|mp-only|rabbitmq-only]")
    else:
        # Full benchmark
        run_full_benchmark()