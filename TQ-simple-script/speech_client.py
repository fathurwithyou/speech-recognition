import asyncio
import os
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

    # Timing data collection for benchmarking
    processing_times = []
    total_times = []

    try:
        async with websockets.connect(uri) as websocket:
            print(f"Connected to speech recognition server at {uri}")
            print("=" * 60)

            # Send audio file processing requests
            for file_id in test_files:
                wav_path = f"../timit_eval/{file_id}.wav"
                if os.path.exists(wav_path):
                    print(f"> Sending WAV file: {file_id}.wav")
                    with open(wav_path, "rb") as f:
                        wav_data = f.read()
                    
                    # Send file metadata and data
                    message = {
                        "file_id": file_id,
                        "data": wav_data.hex()
                    }
                    await websocket.send(json.dumps(message))
                    response = await websocket.recv()
                    print(f"< Server response: {response}")
                    await asyncio.sleep(0.5)
                else:
                    print(f"Warning: File {wav_path} not found")
            
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
                        # Collect timing data for benchmarking
                        if 'processing_time' in result_data:
                            processing_times.append(result_data['processing_time'])
                        if 'inference_time' in result_data:
                            processing_times.append(result_data['inference_time'])

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

        # Calculate and display benchmark statistics
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Total time for {total_tasks} tasks: {elapsed:.2f} seconds")
        if total_tasks > 0:
            print(f"Average time per task: {elapsed / total_tasks:.2f} seconds")

        # Audio processing time statistics
        if processing_times:
            import statistics
            avg_processing = statistics.mean(processing_times)
            std_processing = statistics.stdev(processing_times) if len(processing_times) > 1 else 0.0

            print(f"\nAUDIO PROCESSING TIME STATISTICS:")
            print(f"  Sample Count: {len(processing_times)} files")
            print(f"  Average: {avg_processing:.3f} seconds")
            print(f"  Std Deviation: {std_processing:.3f} seconds")
            print(f"  Min Time: {min(processing_times):.3f} seconds")
            print(f"  Max Time: {max(processing_times):.3f} seconds")
        else:
            print("\nNo processing time data collected")

async def batch_transcription_test():
    """Test batch transcription of multiple random TIMIT files."""
    uri = "ws://localhost:8765"

    # Test with random files from the TIMIT dataset (0-199)
    num_files = 10
    random_files = [str(random.randint(0, 199)) for _ in range(num_files)]

    print(f"Testing batch transcription of {num_files} random TIMIT files:")
    print(f"Files: {', '.join(f'{f}.wav' for f in random_files)}")

    # Timing data collection for benchmarking
    processing_times = []
    batch_start_time = time.perf_counter()
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"\nConnected to speech recognition server at {uri}")
            
            # Send all requests quickly
            for file_id in random_files:
                wav_path = f"../timit_eval/{file_id}.wav"
                if os.path.exists(wav_path):
                    with open(wav_path, "rb") as f:
                        wav_data = f.read()
                    
                    message = {
                        "file_id": file_id,
                        "data": wav_data.hex()
                    }
                    await websocket.send(json.dumps(message))
                    response = await websocket.recv()
                    print(f"Queued: {file_id}.wav")
                else:
                    print(f"Warning: File {wav_path} not found")
            
            print("\nWaiting for batch transcription results...")
            print("=" * 60)
            
            # Collect all results
            results = []
            for i in range(num_files):
                result = await websocket.recv()
                results.append(result)
                print(f"Result {i+1}/{num_files}: {result}")

                # Extract timing data from results
                try:
                    result_data = json.loads(result.split(": ", 1)[1])
                    if isinstance(result_data, dict) and 'processing_time' in result_data:
                        processing_times.append(result_data['processing_time'])
                except:
                    pass

            batch_end_time = time.perf_counter()
            batch_elapsed = batch_end_time - batch_start_time

            print(f"\nBatch transcription of {num_files} files completed!")

            # Display batch benchmark results
            print("\n" + "=" * 60)
            print("BATCH BENCHMARK RESULTS")
            print("=" * 60)
            print(f"Total batch time: {batch_elapsed:.2f} seconds")
            print(f"Average time per file: {batch_elapsed / num_files:.2f} seconds")

            if processing_times:
                import statistics
                avg_processing = statistics.mean(processing_times)
                std_processing = statistics.stdev(processing_times) if len(processing_times) > 1 else 0.0

                print(f"\nAUDIO PROCESSING TIME STATISTICS:")
                print(f"  Sample Count: {len(processing_times)} files")
                print(f"  Average: {avg_processing:.3f} seconds")
                print(f"  Std Deviation: {std_processing:.3f} seconds")
                print(f"  Min Time: {min(processing_times):.3f} seconds")
                print(f"  Max Time: {max(processing_times):.3f} seconds")

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