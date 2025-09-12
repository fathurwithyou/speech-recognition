import multiprocessing as mp
import time
import json
import os
from queue import Empty
from audio_utils import load_audio_file, extract_mfcc_features, get_timit_file_path, simple_speech_to_text

def process_speech_task_mp(task_data):
    """Process speech recognition task for multiprocessing."""
    task_id, file_id, start_time = task_data
    
    try:
        # Get the file path
        file_path = get_timit_file_path(file_id, base_path="../timit_eval")
        if not file_path:
            return {
                'task_id': task_id,
                'file_id': file_id,
                'error': f"Audio file {file_id}.wav not found",
                'processing_time': time.time() - start_time,
                'worker_pid': os.getpid()
            }
        
        # Load audio file
        audio_data, sample_rate = load_audio_file(file_path)
        
        # Extract features
        features = extract_mfcc_features(audio_data, sample_rate)
        
        # Simulate processing time based on audio length
        processing_time = min(3.0, len(audio_data) / sample_rate * 0.5)
        time.sleep(processing_time)
        
        # Generate transcription
        transcription = simple_speech_to_text(features, file_id)
        
        total_time = time.time() - start_time
        
        return {
            'task_id': task_id,
            'file_id': file_id,
            'transcription': transcription,
            'duration_seconds': len(audio_data) / sample_rate,
            'sample_rate': sample_rate,
            'num_features': len(features),
            'processing_time': processing_time,
            'total_time': total_time,
            'worker_pid': os.getpid()
        }
        
    except Exception as e:
        return {
            'task_id': task_id,
            'file_id': file_id,
            'error': str(e),
            'processing_time': time.time() - start_time,
            'worker_pid': os.getpid()
        }

class MultiprocessingQueue:
    """Multiprocessing-based queue implementation for comparison with RabbitMQ."""
    
    def __init__(self, num_workers=4):
        self.num_workers = num_workers
        self.pool = None
        self.results = []
        
    def start_workers(self):
        """Start worker pool."""
        self.pool = mp.Pool(processes=self.num_workers)
        print(f"Started {self.num_workers} worker processes")
        
    def submit_tasks(self, file_ids):
        """Submit tasks to worker pool."""
        if not self.pool:
            self.start_workers()
            
        tasks = []
        start_time = time.time()
        
        for i, file_id in enumerate(file_ids):
            task_data = (f"task_{i}", file_id, time.time())
            tasks.append(task_data)
        
        print(f"Submitting {len(tasks)} tasks to worker pool...")
        
        # Submit all tasks
        async_results = []
        for task_data in tasks:
            async_result = self.pool.apply_async(process_speech_task_mp, (task_data,))
            async_results.append(async_result)
            
        return async_results, start_time
        
    def collect_results(self, async_results, timeout=60):
        """Collect results from worker pool."""
        results = []
        completed = 0
        total = len(async_results)
        
        for async_result in async_results:
            try:
                result = async_result.get(timeout=timeout)
                results.append(result)
                completed += 1
                
                if completed % 5 == 0 or completed == total:
                    print(f"Completed: {completed}/{total}")
                    
            except mp.TimeoutError:
                results.append({
                    'error': 'Task timeout',
                    'processing_time': timeout
                })
                
        return results
        
    def shutdown(self):
        """Shutdown worker pool."""
        if self.pool:
            self.pool.close()
            self.pool.join()
            print("Worker pool shutdown complete")

def run_multiprocessing_benchmark(file_ids, num_workers=4):
    """Run benchmark using multiprocessing queue."""
    print(f"\n{'='*60}")
    print(f"MULTIPROCESSING BENCHMARK - {num_workers} workers")
    print(f"{'='*60}")
    
    mp_queue = MultiprocessingQueue(num_workers=num_workers)
    
    try:
        # Submit tasks
        async_results, start_time = mp_queue.submit_tasks(file_ids)
        
        # Collect results
        results = mp_queue.collect_results(async_results)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate metrics
        successful_tasks = [r for r in results if 'error' not in r]
        failed_tasks = [r for r in results if 'error' in r]
        
        if successful_tasks:
            avg_processing_time = sum(r['processing_time'] for r in successful_tasks) / len(successful_tasks)
            avg_audio_duration = sum(r['duration_seconds'] for r in successful_tasks) / len(successful_tasks)
        else:
            avg_processing_time = 0
            avg_audio_duration = 0
            
        print(f"\nMULTIPROCESSING RESULTS:")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Tasks completed: {len(successful_tasks)}/{len(file_ids)}")
        print(f"Tasks failed: {len(failed_tasks)}")
        print(f"Average processing time per task: {avg_processing_time:.2f}s")
        print(f"Average audio duration: {avg_audio_duration:.2f}s")
        print(f"Throughput: {len(successful_tasks)/total_time:.2f} tasks/second")
        
        if failed_tasks:
            print(f"\nFailed tasks:")
            for task in failed_tasks:
                print(f"  - {task.get('file_id', 'unknown')}: {task['error']}")
                
        return {
            'method': 'multiprocessing',
            'num_workers': num_workers,
            'total_time': total_time,
            'successful_tasks': len(successful_tasks),
            'failed_tasks': len(failed_tasks),
            'avg_processing_time': avg_processing_time,
            'avg_audio_duration': avg_audio_duration,
            'throughput': len(successful_tasks)/total_time,
            'results': results
        }
        
    finally:
        mp_queue.shutdown()

if __name__ == "__main__":
    # Test with a subset of TIMIT files
    test_files = [str(i) for i in range(20)]  # Test first 20 files
    
    # Test different worker counts
    for workers in [1, 2, 4, 8]:
        run_multiprocessing_benchmark(test_files, num_workers=workers)
        time.sleep(2)  # Brief pause between tests