import asyncio
import websockets
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import pika

class RabbitMQBenchmark:
    """Benchmark RabbitMQ-based speech recognition system."""
    
    def __init__(self, websocket_url="ws://localhost:8765"):
        self.websocket_url = websocket_url
        self.results = []
        self.completed_tasks = 0
        self.start_time = None
        self.lock = threading.Lock()
        
    async def send_requests(self, file_ids):
        """Send all requests via WebSocket."""
        try:
            async with websockets.connect(self.websocket_url) as websocket:
                print(f"Connected to {self.websocket_url}")
                
                # Send all requests
                for file_id in file_ids:
                    await websocket.send(file_id)
                    response = await websocket.recv()  # Acknowledgment
                    print(f"Queued: {file_id}.wav")
                
                print(f"All {len(file_ids)} requests sent. Waiting for results...")
                
                # Collect results
                results = []
                for i in range(len(file_ids)):
                    result = await websocket.recv()
                    results.append({
                        'raw_result': result,
                        'received_at': time.time()
                    })
                    
                    if (i + 1) % 5 == 0 or (i + 1) == len(file_ids):
                        print(f"Received: {i + 1}/{len(file_ids)} results")
                
                return results
                
        except ConnectionRefusedError:
            print("ERROR: Could not connect to WebSocket server")
            print("Make sure to start: python src/producer.py")
            return []
        except Exception as e:
            print(f"WebSocket error: {e}")
            return []
    
    def parse_result(self, result):
        """Parse result from WebSocket response."""
        try:
            # Extract JSON from result string
            if ": " in result['raw_result']:
                json_str = result['raw_result'].split(": ", 1)[1]
                data = json.loads(json_str)
                data['received_at'] = result['received_at']
                return data
        except:
            pass
        
        return {
            'raw': result['raw_result'],
            'received_at': result['received_at'],
            'error': 'Failed to parse result'
        }

def run_rabbitmq_benchmark(file_ids):
    """Run benchmark using RabbitMQ system."""
    print(f"\n{'='*60}")
    print(f"RABBITMQ BENCHMARK")
    print(f"{'='*60}")
    
    benchmark = RabbitMQBenchmark()
    
    start_time = time.time()
    
    # Run the async WebSocket client
    results = asyncio.run(benchmark.send_requests(file_ids))
    
    end_time = time.time()
    total_time = end_time - start_time
    
    if not results:
        return None
    
    # Parse results
    parsed_results = [benchmark.parse_result(r) for r in results]
    successful_tasks = [r for r in parsed_results if 'error' not in r or r.get('transcription')]
    failed_tasks = [r for r in parsed_results if 'error' in r and not r.get('transcription')]
    
    if successful_tasks:
        avg_processing_time = sum(r.get('processing_time', 0) for r in successful_tasks) / len(successful_tasks)
        avg_audio_duration = sum(r.get('duration_seconds', 0) for r in successful_tasks) / len(successful_tasks)
    else:
        avg_processing_time = 0
        avg_audio_duration = 0
    
    print(f"\nRABBITMQ RESULTS:")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Tasks completed: {len(successful_tasks)}/{len(file_ids)}")
    print(f"Tasks failed: {len(failed_tasks)}")
    print(f"Average processing time per task: {avg_processing_time:.2f}s")
    print(f"Average audio duration: {avg_audio_duration:.2f}s")
    print(f"Throughput: {len(successful_tasks)/total_time:.2f} tasks/second")
    
    if failed_tasks:
        print(f"\nFailed tasks:")
        for task in failed_tasks[:5]:  # Show first 5 failures
            print(f"  - {task.get('file_id', 'unknown')}: {task.get('error', 'Unknown error')}")
    
    return {
        'method': 'rabbitmq',
        'total_time': total_time,
        'successful_tasks': len(successful_tasks),
        'failed_tasks': len(failed_tasks),
        'avg_processing_time': avg_processing_time,
        'avg_audio_duration': avg_audio_duration,
        'throughput': len(successful_tasks)/total_time,
        'results': parsed_results
    }

def check_rabbitmq_system():
    """Check if RabbitMQ system components are running."""
    print("Checking RabbitMQ system components...")
    
    # Check RabbitMQ connection
    try:
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host='localhost', port=5672)
        )
        connection.close()
        print("✓ RabbitMQ server is running")
    except:
        print("✗ RabbitMQ server is NOT running")
        print("  Start with: docker run --name my-rabbitmq -p 5672:5672 -p 15672:15672 -d rabbitmq:3-management")
        return False
    
    # Check WebSocket server
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 8765))
        sock.close()
        if result == 0:
            print("✓ WebSocket server is running")
        else:
            print("✗ WebSocket server is NOT running")
            print("  Start with: python src/producer.py")
            return False
    except:
        print("✗ Could not check WebSocket server")
        return False
    
    print("✓ All components are ready for benchmarking")
    return True

if __name__ == "__main__":
    # Check system before running benchmark
    if not check_rabbitmq_system():
        print("\nPlease start the required components and try again.")
        exit(1)
    
    # Test with subset of TIMIT files
    test_files = [str(i) for i in range(20)]  # First 20 files
    
    result = run_rabbitmq_benchmark(test_files)
    if result:
        print(f"\nBenchmark completed successfully!")
    else:
        print(f"\nBenchmark failed!")