import asyncio
import websockets
import pika
import json
import uuid
from datetime import datetime, timezone
from functools import partial
import threading

RABBITMQ_HOST = 'localhost'
RABBITMQ_PORT = 5672
QUEUE_NAME = 'audio_processing_queue'

def rabbitmq_consumer_thread(client_id, websocket, stop_event, loop):
    try:
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=RABBITMQ_HOST, port=RABBITMQ_PORT)
        )
        channel = connection.channel()
        
        result_queue = f"result_{client_id}"
        channel.queue_declare(queue=result_queue, durable=False, auto_delete=True)
        
        def callback(ch, method, properties, body):
            try:
                data = json.loads(body.decode('utf-8'))
                message = f"Result for task {data['task_id']}: {data['result']}"
                
                # Schedule the coroutine in the main event loop
                future = asyncio.run_coroutine_threadsafe(websocket.send(message), loop)
                # Wait for the send to complete (with timeout)
                future.result(timeout=5.0)
                
                ch.basic_ack(delivery_tag=method.delivery_tag)
            except asyncio.TimeoutError:
                print(f"Timeout sending WebSocket message for task {data.get('task_id', 'unknown')}")
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
            except Exception as e:
                print(f"Error processing result: {e}")
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
        
        channel.basic_consume(queue=result_queue, on_message_callback=callback)
        
        while not stop_event.is_set():
            connection.process_data_events(time_limit=1)
            
    except Exception as e:
        print(f"RabbitMQ consumer error: {e}")
    finally:
        try:
            connection.close()
        except:
            pass

async def handler(websocket, rabbitmq_channel, client_id):
    try:
        async for message in websocket:
            task_id = str(uuid.uuid4())
            print(f"Received message: {message}, assigned task ID: {task_id}")
            task = {
                'task_id': task_id,
                'client_id': client_id,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'data': message
            }
            
            rabbitmq_channel.basic_publish(
                exchange='',
                routing_key=QUEUE_NAME,
                body=json.dumps(task),
                properties=pika.BasicProperties(delivery_mode=2)
            )
            print(f"Task {task_id} enqueued successfully.")

            await websocket.send(f"Task {task_id} received and enqueued.")
    
    except websockets.exceptions.ConnectionClosed:
        print(f"Client disconnected: {websocket.remote_address}")
    except Exception as e:
        print(f"An error occurred: {e}")

async def connection_manager(websocket, rabbitmq_channel):
    client_id = str(uuid.uuid4())
    stop_event = threading.Event()
    
    # Get the current event loop to pass to the thread
    loop = asyncio.get_event_loop()
    
    consumer_thread = threading.Thread(
        target=rabbitmq_consumer_thread,
        args=(client_id, websocket, stop_event, loop)
    )
    consumer_thread.start()

    try:
        await handler(websocket, rabbitmq_channel, client_id)
    finally:
        stop_event.set()
        consumer_thread.join(timeout=5)
        print(f"Connection with client {client_id} closed.")

async def main():
    try:
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=RABBITMQ_HOST, port=RABBITMQ_PORT)
        )
        channel = connection.channel()
        
        channel.queue_declare(queue=QUEUE_NAME, durable=True)
        print("Connected to RabbitMQ successfully.")
        
    except Exception as e:
        print(f"Could not connect to RabbitMQ: {e}")
        return

    handler_with_rabbitmq = partial(connection_manager, rabbitmq_channel=channel)

    async with websockets.serve(handler_with_rabbitmq, "localhost", 8765) as server:
        print("WebSocket server started on ws://localhost:8765")
        await asyncio.Future() 


if __name__ == "__main__":
    asyncio.run(main())
