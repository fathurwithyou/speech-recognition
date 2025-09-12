#!/usr/bin/env python3
"""
Scaling test for multiple RabbitMQ workers with different configurations.
Tests how the system scales with different numbers and types of workers.
"""

import time
import json
import subprocess
import os
import signal
import asyncio
import websockets
from datetime import datetime

class ScalingTest:
    """Test scaling characteristics of different worker configurations."""
    
    def __init__(self):
        self.producer_process = None
        self.worker_processes = []
        
    def start_producer(self):
        """Start the producer process."""
        print("🚀 Starting producer...")
        self.producer_process = subprocess.Popen([
            'python', 'src/producer.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(3)
        return True
    
    def stop_producer(self):
        """Stop the producer process."""
        if self.producer_process:
            try:
                self.producer_process.terminate()
                self.producer_process.wait(timeout=5)
            except:
                self.producer_process.kill()
        self.producer_process = None
    
    def start_workers(self, config):
        """Start workers based on configuration."""
        processes = []
        
        for worker_spec in config['workers']:
            if worker_spec['type'] == 'single':
                cmd = ['python', 'src/worker.py']
            elif worker_spec['type'] == 'multiprocessing':
                cmd = [
                    'python', 'src/mp_consumer_worker.py',
                    '--pool-size', str(worker_spec['pool_size']),
                    '--batch-size', str(worker_spec.get('batch_size', 5)),
                    '--batch-timeout', str(worker_spec.get('batch_timeout', 2.0))
                ]
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            processes.append({
                'process': process,
                'spec': worker_spec
            })
            
            time.sleep(0.5)  # Brief delay between worker starts
        
        print(f"✅ Started {len(processes)} workers")
        return processes
    
    def stop_workers(self, worker_processes):
        """Stop all worker processes."""
        print("🔄 Stopping workers...")
        
        for worker in worker_processes:
            try:
                worker['process'].terminate()
                worker['process'].wait(timeout=5)
            except:
                worker['process'].kill()
        
        print(f"✅ Stopped {len(worker_processes)} workers")
    
    async def send_load_test(self, file_ids, test_name):
        """Send load test via WebSocket."""
        print(f"\n📊 Running load test: {test_name}")
        print(f"   Files to process: {len(file_ids)}")
        
        start_time = time.time()
        
        try:
            async with websockets.connect("ws://localhost:8765", timeout=10) as websocket:
                # Send all requests rapidly
                send_start = time.time()
                for file_id in file_ids:
                    await websocket.send(file_id)
                    await websocket.recv()  # Get acknowledgment
                send_time = time.time() - send_start
                
                print(f"   📤 All requests sent in {send_time:.2f}s")
                
                # Collect results
                results = []
                collect_start = time.time()
                
                for i in range(len(file_ids)):
                    result = await asyncio.wait_for(websocket.recv(), timeout=180)
                    results.append(result)
                    
                    if (i + 1) % 20 == 0:
                        elapsed = time.time() - collect_start
                        rate = (i + 1) / elapsed
                        print(f"   📥 Collected {i + 1}/{len(file_ids)} results ({rate:.1f} results/sec)")
                
                total_time = time.time() - start_time
                collection_time = time.time() - collect_start
                
                # Parse results
                successful = 0
                failed = 0
                total_processing_time = 0
                
                for result in results:
                    try:
                        if ": " in result:
                            json_str = result.split(": ", 1)[1]
                            data = json.loads(json_str)
                            if 'error' not in data:
                                successful += 1
                                total_processing_time += data.get('processing_time', 0)
                            else:
                                failed += 1
                    except:
                        failed += 1
                
                avg_processing_time = total_processing_time / successful if successful > 0 else 0
                
                return {
                    'test_name': test_name,
                    'total_time': total_time,
                    'send_time': send_time,
                    'collection_time': collection_time,
                    'successful_tasks': successful,
                    'failed_tasks': failed,
                    'throughput': successful / total_time,
                    'avg_processing_time': avg_processing_time,
                    'queue_efficiency': successful / collection_time  # How fast results come back
                }
                
        except Exception as e:
            print(f"❌ Load test failed: {e}")
            return None
    
    def run_scaling_tests(self, test_files):
        """Run comprehensive scaling tests."""
        print("🎯 SCALING TEST SUITE")
        print("=" * 60)
        print(f"Testing with {len(test_files)} files")
        
        # Define scaling test configurations
        test_configs = [
            {
                'name': '1 Single Worker',
                'workers': [{'type': 'single'}]
            },
            {
                'name': '2 Single Workers',
                'workers': [{'type': 'single'}, {'type': 'single'}]
            },
            {
                'name': '4 Single Workers',
                'workers': [{'type': 'single'} for _ in range(4)]
            },
            {
                'name': '1 MP Worker (2 processes)',
                'workers': [{'type': 'multiprocessing', 'pool_size': 2, 'batch_size': 3}]
            },
            {
                'name': '1 MP Worker (4 processes)',
                'workers': [{'type': 'multiprocessing', 'pool_size': 4, 'batch_size': 5}]
            },
            {
                'name': '1 MP Worker (8 processes)',
                'workers': [{'type': 'multiprocessing', 'pool_size': 8, 'batch_size': 8}]
            },
            {
                'name': '2 MP Workers (4 processes each)',
                'workers': [
                    {'type': 'multiprocessing', 'pool_size': 4, 'batch_size': 5},
                    {'type': 'multiprocessing', 'pool_size': 4, 'batch_size': 5}
                ]
            },
            {
                'name': 'Mixed: 2 Single + 1 MP (4 proc)',
                'workers': [
                    {'type': 'single'},
                    {'type': 'single'},
                    {'type': 'multiprocessing', 'pool_size': 4, 'batch_size': 5}
                ]
            }
        ]
        
        results = []
        
        # Start producer once
        if not self.start_producer():
            print("❌ Failed to start producer")
            return []
        
        try:
            for i, config in enumerate(test_configs, 1):
                print(f"\n{'='*60}")
                print(f"🧪 Test {i}/{len(test_configs)}: {config['name']}")
                print(f"{'='*60}")
                
                # Calculate total worker capacity
                total_single = sum(1 for w in config['workers'] if w['type'] == 'single')
                total_mp_capacity = sum(w.get('pool_size', 0) for w in config['workers'] if w['type'] == 'multiprocessing')
                total_capacity = total_single + total_mp_capacity
                
                print(f"   Workers: {len(config['workers'])}")
                print(f"   Total processing capacity: {total_capacity}")
                
                # Start workers
                worker_processes = self.start_workers(config)
                
                # Wait for workers to initialize
                print("⏳ Waiting for workers to initialize...")
                time.sleep(8)
                
                # Run load test
                result = asyncio.run(self.send_load_test(test_files, config['name']))
                
                if result:
                    result['config'] = config
                    result['worker_count'] = len(config['workers'])
                    result['total_capacity'] = total_capacity
                    results.append(result)
                    
                    print(f"✅ Test completed:")
                    print(f"   Total time: {result['total_time']:.2f}s")
                    print(f"   Throughput: {result['throughput']:.2f} tasks/sec")
                    print(f"   Queue efficiency: {result['queue_efficiency']:.2f} results/sec")
                    print(f"   Success rate: {result['successful_tasks']}/{result['successful_tasks'] + result['failed_tasks']}")
                    print(f"   Efficiency per capacity: {result['throughput']/total_capacity:.3f} tasks/sec/unit")
                else:
                    print("❌ Test failed")
                
                # Stop workers
                self.stop_workers(worker_processes)
                
                # Brief pause between tests
                if i < len(test_configs):
                    print("\n⏸️  Pausing before next test...")
                    time.sleep(5)
                    
        finally:
            self.stop_producer()
        
        return results
    
    def analyze_scaling_results(self, results):
        """Analyze and display scaling results."""
        if not results:
            print("❌ No results to analyze")
            return
        
        print(f"\n{'='*100}")
        print("🏆 SCALING ANALYSIS RESULTS")
        print(f"{'='*100}")
        
        # Table header
        print(f"{'Configuration':<25} {'Workers':<8} {'Capacity':<9} {'Time(s)':<8} {'Throughput':<11} {'Efficiency':<11} {'Success':<8}")
        print("-" * 100)
        
        # Sort by throughput
        sorted_results = sorted(results, key=lambda x: x['throughput'], reverse=True)
        
        for result in sorted_results:
            efficiency = result['throughput'] / result['total_capacity']
            print(f"{result['test_name']:<25} {result['worker_count']:<8} {result['total_capacity']:<9} "
                  f"{result['total_time']:<8.2f} {result['throughput']:<11.2f} {efficiency:<11.3f} "
                  f"{result['successful_tasks']:<8}")
        
        # Performance analysis
        print(f"\n📈 PERFORMANCE INSIGHTS:")
        
        best_throughput = max(results, key=lambda x: x['throughput'])
        best_efficiency = max(results, key=lambda x: x['throughput']/x['total_capacity'])
        fastest_completion = min(results, key=lambda x: x['total_time'])
        
        print(f"🚀 Highest throughput: {best_throughput['test_name']} - {best_throughput['throughput']:.2f} tasks/sec")
        print(f"⚡ Fastest completion: {fastest_completion['test_name']} - {fastest_completion['total_time']:.2f} seconds")
        print(f"🎯 Best efficiency: {best_efficiency['test_name']} - {(best_efficiency['throughput']/best_efficiency['total_capacity']):.3f} tasks/sec per unit")
        
        # Scaling patterns
        single_worker_results = [r for r in results if 'Single' in r['test_name'] and 'Mixed' not in r['test_name']]
        mp_worker_results = [r for r in results if 'MP' in r['test_name'] and len(r['config']['workers']) == 1]
        
        if len(single_worker_results) >= 2:
            print(f"\n🔢 SINGLE WORKER SCALING:")
            for result in sorted(single_worker_results, key=lambda x: x['worker_count']):
                scaling_efficiency = result['throughput'] / result['worker_count']
                print(f"   {result['worker_count']} workers: {result['throughput']:.2f} tasks/sec (efficiency: {scaling_efficiency:.2f})")
        
        if len(mp_worker_results) >= 2:
            print(f"\n🔀 MULTIPROCESSING SCALING:")
            for result in sorted(mp_worker_results, key=lambda x: x['total_capacity']):
                print(f"   {result['total_capacity']} processes: {result['throughput']:.2f} tasks/sec "
                      f"(efficiency: {result['throughput']/result['total_capacity']:.3f})")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if best_throughput['throughput'] > 0:
            if 'MP' in best_throughput['test_name']:
                print(f"✅ For maximum throughput: Use multiprocessing workers")
                print(f"   Best configuration: {best_throughput['test_name']}")
            else:
                print(f"✅ For maximum throughput: Use single workers")
                print(f"   Best configuration: {best_throughput['test_name']}")
        
        if best_efficiency['throughput']/best_efficiency['total_capacity'] > 0:
            print(f"✅ For resource efficiency: {best_efficiency['test_name']}")
            print(f"   Achieves {(best_efficiency['throughput']/best_efficiency['total_capacity']):.3f} tasks/sec per processing unit")
        
        return sorted_results
    
    def save_results(self, results, filename=None):
        """Save scaling test results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scaling_test_{timestamp}.json"
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'test_type': 'scaling_test',
            'results': results
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
        return filename

def main():
    """Main scaling test execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='RabbitMQ Worker Scaling Test')
    parser.add_argument('--files', type=int, default=50, help='Number of test files (default: 50)')
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer files and configs')
    
    args = parser.parse_args()
    
    # Check if RabbitMQ is available
    try:
        import pika
        connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost', port=5672))
        connection.close()
    except:
        print("❌ RabbitMQ not available. Please start:")
        print("docker run --name my-rabbitmq -p 5672:5672 -p 15672:15672 -d rabbitmq:3-management")
        return
    
    # Generate test files
    if args.quick:
        test_files = [str(i) for i in range(15)]  # Quick test with 15 files
        print("🚀 Running quick scaling test...")
    else:
        test_files = [str(i) for i in range(min(args.files, 200))]
        print(f"🎯 Running full scaling test with {len(test_files)} files...")
    
    # Run scaling test
    scaling_test = ScalingTest()
    results = scaling_test.run_scaling_tests(test_files)
    
    if results:
        analyzed_results = scaling_test.analyze_scaling_results(results)
        scaling_test.save_results(analyzed_results)
        print("\n🎉 Scaling test completed successfully!")
    else:
        print("\n❌ Scaling test failed!")

if __name__ == "__main__":
    main()