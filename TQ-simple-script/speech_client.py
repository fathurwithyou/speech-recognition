import asyncio
import os

from numpy import test
import websockets
import json
import random
import time

async def test_speech_recognition():
    """Test speech recognition with TIMIT audio files."""
    uri = "ws://localhost:8765"
    
    test_files = os.listdir("../timit_eval")
    test_files = [f.split(".")[0] for f in test_files if f.endswith(".wav")]    
    test_files = test_files[:20]  
    results_received = 0
    total_tasks = len(test_files)
    start_time = time.perf_counter()

    try:
        async with websockets.connect(uri) as websocket:
            print(f"Connected to speech recognition server at {uri}")
            print("=" * 60)

            # Send audio file processing requests
            for file_id in test_files:
                print(f"> Requesting transcription for TIMIT file: {file_id}.wav")
                await websocket.send(file_id)
                response = await websocket.recv()
                print(f"< Server response: {response}")
                await asyncio.sleep(0.5)
            
            print("\nFinished sending requests. Waiting for transcription results...")
            print("=" * 60)

            # Receive transcription results
            while results_received < total_tasks:
                result = await websocket.recv()
                print(f"< Transcription result: {result}")
                
                # Try to parse as JSON for better formatting
                try:
                    result_data = json.loads(result.split(": ", 1)[1])
                    if isinstance(result_data, dict) and 'transcription' in result_data:
                        print(f"  File: {result_data['file_id']}.wav")
                        print(f"  Duration: {result_data['duration_seconds']:.2f}s")
                        print(f"  Sample Rate: {result_data['sample_rate']} Hz")
                        print(f"  Features: {result_data['num_features']} frames")
                        print(f"  Processing Time: {result_data['processing_time']:.2f}s")
                        print(f"  Transcription: {result_data['transcription']}")
                        print("-" * 40)
                except:
                    pass  # If parsing fails, just show the raw result
                
                results_received += 1

            print("All transcriptions completed!")

    except ConnectionRefusedError:
        print("Connection refused. Is the speech recognition server running?")
        print("Start it with: python src/producer.py")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        print(f"Total time for {total_tasks} tasks: {elapsed:.2f} seconds")
        if total_tasks > 0:
            print(f"Average time per task: {elapsed / total_tasks:.2f} seconds")

async def batch_transcription_test():
    """Test batch transcription of multiple random TIMIT files."""
    uri = "ws://localhost:8765"
    
    # Test with random files from the TIMIT dataset (0-199)
    num_files = 10
    random_files = [str(random.randint(0, 199)) for _ in range(num_files)]
    
    print(f"Testing batch transcription of {num_files} random TIMIT files:")
    print(f"Files: {', '.join(f'{f}.wav' for f in random_files)}")
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"\nConnected to speech recognition server at {uri}")
            
            # Send all requests quickly
            for file_id in random_files:
                await websocket.send(file_id)
                response = await websocket.recv()
                print(f"Queued: {file_id}.wav")
            
            print("\nWaiting for batch transcription results...")
            print("=" * 60)
            
            # Collect all results
            results = []
            for i in range(num_files):
                result = await websocket.recv()
                results.append(result)
                print(f"Result {i+1}/{num_files}: {result}")
            
            print(f"\nBatch transcription of {num_files} files completed!")
            
    except Exception as e:
        print(f"Batch test error: {e}")

if __name__ == "__main__":
    print("TIMIT Speech Recognition Test Client")
    print("====================================")
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        asyncio.run(batch_transcription_test())
    else:
        asyncio.run(test_speech_recognition())