import pika
import json
import time
import os
import multiprocessing as mp
from queue import Queue, Empty
import threading
from audio_utils import get_timit_file_path, model_based_speech_to_text

# Pool initializer: load model once per worker process and set thread counts
def _pool_init(threads_per_model: int):
    try:
        import os as _os
        _os.environ["OMP_NUM_THREADS"] = str(threads_per_model)
        _os.environ["MKL_NUM_THREADS"] = str(threads_per_model)
    except Exception:
        pass
    try:
        import torch as _torch
        try:
            _torch.set_num_threads(int(threads_per_model))
            _torch.set_num_interop_threads(int(threads_per_model))
        except Exception:
            pass
    except Exception:
        pass

    try:
        from model_loader import load_model_if_needed
        load_model_if_needed(0)
    except Exception:
        pass

# Standalone worker function for multiprocessing (must be at module level)
def process_single_task_worker(task_data):
    """Process a single speech recognition task (standalone function for multiprocessing)."""
    # Extract only the serializable data
    task_dict, task_id, file_id, temp_path, start_time, process_id = task_data
    
    try:
        # Use temp_path if provided, otherwise use TIMIT directory
        if temp_path:
            file_path = temp_path
        else:
            file_path = get_timit_file_path(file_id, base_path="../timit_eval")
            
        if not file_path or not os.path.exists(file_path):
            return {
                'task_id': task_id,
                'file_id': file_id,
                'success': False,
                'error': f"Audio file not found: {temp_path if temp_path else f'{file_id}.wav'}",
                'processing_time': time.time() - start_time,
                'worker_pid': os.getpid()
            }
        
        # Generate transcription using Whisper model with specific process ID
        transcription_result = model_based_speech_to_text(file_path, file_id, use_model=True, process_id=process_id)
        
        # Basic metadata without extra feature extraction
        actual_inference_time = transcription_result.get('inference_time', 0.0)
        
        result_data = {
            'file_id': file_id,
            'transcription': transcription_result['transcription'],
            'confidence': transcription_result.get('confidence', 0.0),
            'model_used': transcription_result.get('model_used', 'unknown'),
            'processing_time': actual_inference_time,
            'inference_time': actual_inference_time,
            'worker_pid': os.getpid()
        }
        
        # Clean up temp file if it was used
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                print(f"Warning: Could not delete temp file {temp_path}: {e}")
        
        return {
            'task_dict': task_dict,  # Return original task dict
            'task_id': task_id,
            'success': True,
            'result': result_data,
            'total_time': time.time() - start_time,
            'worker_pid': os.getpid()
        }
        
    except Exception as e:
        # Clean up temp file even on failure
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as cleanup_error:
                print(f"Warning: Could not delete temp file {temp_path}: {cleanup_error}")
        
        return {
            'task_dict': task_dict,
            'task_id': task_id,
            'file_id': file_id,
            'success': False,
            'error': str(e),
            'processing_time': time.time() - start_time,
            'worker_pid': os.getpid()
        }

RABBITMQ_HOST = 'localhost'
RABBITMQ_PORT = 5672
QUEUE_TASK = 'audio_processing_queue'

class MultiprocessingConsumerWorker:
    """RabbitMQ consumer with multiprocessing pool for concurrent task processing."""
    
    def __init__(self, worker_pool_size=4, batch_size=5, batch_timeout=2.0, threads_per_model: int = 4):
        self.worker_pool_size = worker_pool_size
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.threads_per_model = int(threads_per_model)
        
        self.connection = None
        self.channel = None
        self.pool = None
        
        # Task batching
        self.pending_tasks = Queue()
        self.result_actions = Queue()
        self.batch_processor_thread = None
        self.shutdown_event = threading.Event()
        
        # Statistics
        self.stats = {
            'tasks_received': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'batches_processed': 0,
            'start_time': time.time(),
            'worker_pid': os.getpid()
        }
        
    def start_worker_pool(self):
        """Start multiprocessing pool and initialize Whisper models."""
        # Create the pool; each worker loads its model once in initializer
        self.pool = mp.Pool(
            processes=self.worker_pool_size,
            initializer=_pool_init,
            initargs=(self.threads_per_model,)
        )
        print(f"Started multiprocessing pool with {self.worker_pool_size} workers")
        print("Each worker process will load its model once")
        
    def stop_worker_pool(self):
        """Stop multiprocessing pool."""
        if self.pool:
            print("Shutting down worker pool...")
            self.pool.close()
            self.pool.join()
            print("Worker pool shutdown complete")
    
    
    def batch_processor(self):
        """Background thread that processes tasks in batches."""
        batch = []
        last_batch_time = time.time()
        
        while not self.shutdown_event.is_set():
            try:
                # Try to get a task (with timeout)
                task_data = self.pending_tasks.get(timeout=0.5)
                batch.append(task_data)
                
                # Process batch if it's full or timeout reached
                should_process = (
                    len(batch) >= self.batch_size or
                    (batch and time.time() - last_batch_time >= self.batch_timeout)
                )
                
                if should_process:
                    self.process_batch(batch)
                    batch = []
                    last_batch_time = time.time()
                    
            except Empty:
                # No tasks available, process any pending batch
                if batch and time.time() - last_batch_time >= self.batch_timeout:
                    self.process_batch(batch)
                    batch = []
                    last_batch_time = time.time()
            
            except Exception as e:
                print(f"Batch processor error: {e}")
        
        # Process remaining batch on shutdown
        if batch:
            self.process_batch(batch)
    
    def process_batch(self, batch):
        """Process a batch of tasks using multiprocessing pool."""
        if not batch:
            return
            
        print(f"Processing batch of {len(batch)} tasks...")
        batch_start = time.time()
        
        # Prepare serializable data for multiprocessing
        mp_task_data = []
        task_metadata = []
        
        for i, task_data in enumerate(batch):
            task, delivery_tag, start_time = task_data
            # Extract serializable parts
            task_dict = {
                'task_id': task['task_id'],
                'client_id': task.get('client_id'),
                'timestamp': task.get('timestamp'),
                'file_id': task.get('file_id'),
                'temp_path': task.get('temp_path')
            }
            
            # Use a fixed process_id per worker (0) so each worker keeps one model
            file_id = task.get('file_id')
            temp_path = task.get('temp_path')
            serializable_data = (task_dict, task['task_id'], file_id, temp_path, start_time, 0)
            mp_task_data.append(serializable_data)
            
            # Store metadata we need for result handling
            task_metadata.append({
                'original_task': task,
                'delivery_tag': delivery_tag,
                'task_id': task['task_id']
            })
        
        # Submit all tasks to pool
        async_results = []
        for task_data in mp_task_data:
            async_result = self.pool.apply_async(process_single_task_worker, (task_data,))
            async_results.append(async_result)
        
        # Collect results
        for i, async_result in enumerate(async_results):
            try:
                result = async_result.get(timeout=30)  # 30 second timeout per task
                
                # Combine result with original metadata
                metadata = task_metadata[i]
                combined_result = {
                    'task': metadata['original_task'],
                    'delivery_tag': metadata['delivery_tag'],
                    'success': result['success'],
                    'worker_pid': result['worker_pid']
                }
                
                if result['success']:
                    combined_result['result'] = result['result']
                    combined_result['total_time'] = result['total_time']
                else:
                    combined_result['error'] = result['error']
                    combined_result['processing_time'] = result['processing_time']
                
                self.handle_task_result(combined_result)
                
            except mp.TimeoutError:
                print(f"Task timed out in multiprocessing")
                self.stats['tasks_failed'] += 1
                # Acknowledge the failed task
                if i < len(task_metadata):
                    # Defer RabbitMQ operation to main thread
                    self.result_actions.put({
                        'type': 'nack',
                        'delivery_tag': task_metadata[i]['delivery_tag'],
                        'requeue': True
                    })
            except Exception as e:
                print(f"Task processing error: {e}")
                self.stats['tasks_failed'] += 1
                # Acknowledge the failed task
                if i < len(task_metadata):
                    # Defer RabbitMQ operation to main thread
                    self.result_actions.put({
                        'type': 'nack',
                        'delivery_tag': task_metadata[i]['delivery_tag'],
                        'requeue': True
                    })
        
        self.stats['batches_processed'] += 1
        batch_time = time.time() - batch_start
        print(f"Batch completed in {batch_time:.2f}s (avg: {batch_time/len(batch):.2f}s per task)")
    
    def handle_task_result(self, result):
        """Handle completed task result."""
        task = result['task']
        delivery_tag = result['delivery_tag']
        
        # All RabbitMQ operations must be performed on the main thread.
        # Queue a single action for the main loop to execute.
        action = {
            'type': 'complete',
            'delivery_tag': delivery_tag,
            'success': result['success'],
            'task_id': task.get('task_id'),
            'worker_pid': result.get('worker_pid')
        }
        if result['success']:
            action['client_id'] = task.get('client_id')
            action['payload'] = {
                'task_id': task.get('task_id'),
                'result': result['result']
            }
        else:
            action['error'] = result.get('error', 'Unknown error')
        self.result_actions.put(action)

    def _drain_result_actions(self):
        """Execute queued RabbitMQ actions from the main thread only."""
        try:
            while True:
                action = self.result_actions.get_nowait()
                try:
                    if action['type'] == 'nack':
                        self._safe_basic_nack(action['delivery_tag'], action.get('requeue', True))
                    elif action['type'] == 'complete':
                        if action['success']:
                            # Publish result if client_id is available
                            client_id = action.get('client_id')
                            if client_id:
                                result_channel = f"result_{client_id}"
                                self._safe_basic_publish('', result_channel, json.dumps(action['payload']))
                            self.stats['tasks_completed'] += 1
                        else:
                            print(f"Task {action['task_id']} failed: {action.get('error')}")
                            self.stats['tasks_failed'] += 1
                        # Ack after handling
                        self._safe_basic_ack(action['delivery_tag'])
                finally:
                    # Mark task done for internal queue
                    pass
        except Empty:
            return

    def _safe_basic_publish(self, exchange, routing_key, body):
        try:
            if self.channel and getattr(self.channel, 'is_open', False):
                self.channel.basic_publish(exchange=exchange, routing_key=routing_key, body=body)
        except Exception as e:
            print(f"Publish failed: {e}")

    def _safe_basic_ack(self, delivery_tag):
        try:
            if self.channel and getattr(self.channel, 'is_open', False):
                self.channel.basic_ack(delivery_tag=delivery_tag)
        except Exception as e:
            print(f"Ack failed: {e}")

    def _safe_basic_nack(self, delivery_tag, requeue=True):
        try:
            if self.channel and getattr(self.channel, 'is_open', False):
                self.channel.basic_nack(delivery_tag=delivery_tag, requeue=requeue)
        except Exception as e:
            print(f"Nack failed: {e}")
    
    def message_callback(self, ch, method, properties, body):
        """RabbitMQ message callback - adds tasks to processing queue."""
        try:
            task = json.loads(body.decode('utf-8'))
            self.stats['tasks_received'] += 1
            
            # Add to pending tasks queue for batch processing
            task_data = (task, method.delivery_tag, time.time())
            self.pending_tasks.put(task_data)
            
        except Exception as e:
            print(f"Error processing message: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
    
    def print_stats(self):
        """Print current worker statistics."""
        runtime = time.time() - self.stats['start_time']
        throughput = self.stats['tasks_completed'] / runtime if runtime > 0 else 0
        
        print(f"\nWORKER STATS (PID: {self.stats['worker_pid']}):")
        print(f"   Runtime: {runtime:.1f}s")
        print(f"   Received: {self.stats['tasks_received']}")
        print(f"   Completed: {self.stats['tasks_completed']}")
        print(f"   Failed: {self.stats['tasks_failed']}")
        print(f"   Batches: {self.stats['batches_processed']}")
        print(f"   Throughput: {throughput:.2f} tasks/sec")
        print(f"   Pool size: {self.worker_pool_size}")
    
    def run(self):
        """Main worker loop."""
        try:
            # Connect to RabbitMQ
            self.connection = pika.BlockingConnection(
                pika.ConnectionParameters(host=RABBITMQ_HOST, port=RABBITMQ_PORT)
            )
            self.channel = self.connection.channel()
            
            # Declare queue
            self.channel.queue_declare(queue=QUEUE_TASK, durable=True)
            self.channel.basic_qos(prefetch_count=self.batch_size * 2)  # Prefetch for batching
            
            # Start multiprocessing pool
            self.start_worker_pool()
            
            # Start batch processor thread
            self.batch_processor_thread = threading.Thread(target=self.batch_processor, daemon=True)
            self.batch_processor_thread.start()
            
            # Set up RabbitMQ consumer
            self.channel.basic_consume(queue=QUEUE_TASK, on_message_callback=self.message_callback)
            
            print("Multiprocessing consumer worker started")
            print(f"   Pool size: {self.worker_pool_size}")
            print(f"   Batch size: {self.batch_size}")
            print(f"   Batch timeout: {self.batch_timeout}s")
            print(f"   Worker PID: {os.getpid()}")
            print("   Waiting for tasks... (Ctrl+C to stop)")
            
            # Start consuming - print stats every 30 seconds
            last_stats_time = time.time()
            
            while True:
                self.connection.process_data_events(time_limit=1)
                # Drain any queued result actions (publish/ack/nack) from batch thread
                self._drain_result_actions()
                
                if time.time() - last_stats_time >= 30:
                    self.print_stats()
                    last_stats_time = time.time()
                    
        except KeyboardInterrupt:
            print("\nShutting down worker...")
        except Exception as e:
            print(f"Worker error: {e}")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Clean shutdown of worker."""
        print("Shutting down multiprocessing consumer worker...")
        
        # Stop batch processor
        self.shutdown_event.set()
        if self.batch_processor_thread:
            self.batch_processor_thread.join(timeout=5)
        
        # Stop worker pool
        self.stop_worker_pool()
        
        # Close RabbitMQ connection
        try:
            if self.channel:
                self.channel.stop_consuming()
            if self.connection:
                self.connection.close()
        except:
            pass
        
        # Final stats
        self.print_stats()
        print("Worker shutdown complete")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Multiprocessing RabbitMQ Consumer Worker')
    parser.add_argument('--pool-size', type=int, default=4, help='Multiprocessing pool size')
    parser.add_argument('--batch-size', type=int, default=5, help='Batch size for processing')
    parser.add_argument('--batch-timeout', type=float, default=2.0, help='Batch timeout in seconds')
    parser.add_argument('--threads-per-model', type=int, default=4, help='CPU threads used by each model/worker')
    
    args = parser.parse_args()
    
    worker = MultiprocessingConsumerWorker(
        worker_pool_size=args.pool_size,
        batch_size=args.batch_size,
        batch_timeout=args.batch_timeout,
        threads_per_model=args.threads_per_model
    )
    
    worker.run()
