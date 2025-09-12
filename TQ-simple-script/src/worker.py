import pika
import json
import time
import os
from audio_utils import load_audio_file, extract_mfcc_features, get_timit_file_path, model_based_speech_to_text

RABBITMQ_HOST = 'localhost'
RABBITMQ_PORT = 5672
QUEUE_TASK = 'audio_processing_queue'

def process_speech_recognition_task(task):
    """Process speech recognition task with real TIMIT audio files."""
    task_id = task['task_id']
    file_id = task['data']
    
    print(f"Processing speech recognition task ID: {task_id} for file: {file_id}")
    
    try:
        # Get the file path
        file_path = get_timit_file_path(file_id, base_path="../timit_eval")
        if not file_path:
            return f"ERROR: Audio file {file_id}.wav not found"
        
        # Generate transcription using Whisper model on audio file
        print(f"Processing audio file with Whisper: {file_path}")
        transcription_result = model_based_speech_to_text(file_path, file_id, use_model=True)
        
        # Load audio file for additional info
        audio_data, sample_rate = load_audio_file(file_path)
        print(f"Audio duration: {len(audio_data)/sample_rate:.2f}s")
        
        # Extract features for metadata
        features = extract_mfcc_features(audio_data, sample_rate)
        
        # Simulate additional processing time if needed
        base_processing_time = min(3.0, len(audio_data) / sample_rate * 0.5)
        actual_inference_time = transcription_result.get('inference_time', 0.1)
        
        # Add some delay to simulate realistic processing if inference was too fast
        if actual_inference_time < base_processing_time * 0.3:
            additional_delay = (base_processing_time * 0.3) - actual_inference_time
            time.sleep(additional_delay)
            total_processing_time = base_processing_time * 0.3
        else:
            total_processing_time = actual_inference_time
        
        result = {
            'file_id': file_id,
            'transcription': transcription_result['transcription'],
            'confidence': transcription_result.get('confidence', 0.0),
            'model_used': transcription_result.get('model_used', 'unknown'),
            'duration_seconds': len(audio_data) / sample_rate,
            'sample_rate': sample_rate,
            'num_features': len(features),
            'processing_time': total_processing_time,
            'inference_time': actual_inference_time
        }
        
        print(f"Task ID: {task_id} processed successfully.")
        print(f"  Model: {result['model_used']}, Confidence: {result['confidence']:.3f}")
        print(f"  Transcription: {result['transcription']}")
        return result
        
    except Exception as e:
        error_msg = f"ERROR processing {file_id}: {str(e)}"
        print(error_msg)
        return {'error': error_msg, 'file_id': file_id}


def callback(ch, method, properties, body):
    try:
        task = json.loads(body.decode('utf-8'))
        result = process_speech_recognition_task(task)
        client_id = task.get('client_id')
        
        if client_id:
            result_channel = f"result_{client_id}"
            result_data = {
                'task_id': task.get('task_id'),
                'result': result
            }
            
            ch.basic_publish(
                exchange='',
                routing_key=result_channel,
                body=json.dumps(result_data)
            )
            print(f"Published result for task ID: {task['task_id']} to queue: {result_channel}")
        
        ch.basic_ack(delivery_tag=method.delivery_tag)
        
    except Exception as e:
        print(f"An error occurred processing task: {e}")
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

def main():
    try:
        # Initialize Whisper model for this worker
        from model_loader import load_model_if_needed
        
        print("🔄 Initializing Whisper model...")
        model_loaded = load_model_if_needed(process_id=0)  # Single worker uses process_id=0
        
        if model_loaded:
            print("✅ Whisper model loaded successfully")
        else:
            print("⚠️  Failed to load Whisper model - will use fallback transcription")
        
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=RABBITMQ_HOST, port=RABBITMQ_PORT)
        )
        channel = connection.channel()
        
        channel.queue_declare(queue=QUEUE_TASK, durable=True)
        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(queue=QUEUE_TASK, on_message_callback=callback)
        
        print("Successfully connected to RabbitMQ. Waiting for tasks...")
        print("To exit press CTRL+C")
        
        channel.start_consuming()
        
    except pika.exceptions.AMQPConnectionError as e:
        print(f"Could not connect to RabbitMQ: {e}")
    except KeyboardInterrupt:
        print("\nShutting down worker.")
        try:
            channel.stop_consuming()
            connection.close()
        except:
            pass
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()