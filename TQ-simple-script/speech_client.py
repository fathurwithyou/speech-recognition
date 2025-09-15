import asyncio
import os
import websockets
import json
import random
import time
import wave

def get_wav_metadata(wav_path):
    """Extract metadata from WAV file."""
    try:
        with wave.open(wav_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            duration = frames / float(sample_rate)

            # Get file size
            file_size = os.path.getsize(wav_path)

            return {
                'duration': duration,
                'sample_rate': sample_rate,
                'file_size': file_size,
                'frames': frames
            }
    except Exception as e:
        print(f"Error reading WAV metadata from {wav_path}: {e}")
        return None

async def test_speech_recognition():
    """Test speech recognition with TIMIT audio files."""
    uri = "ws://localhost:8765"

    test_files = os.listdir("../timit_eval")
    test_files = [f.split(".")[0] for f in test_files if f.endswith(".wav")]
    test_files = test_files[:20]
    results_received = 0
    total_tasks = len(test_files)
    start_time = time.perf_counter()

    # Audio metadata collection for benchmarking
    audio_durations = []
    audio_file_sizes = []
    audio_sample_rates = []

    try:
        async with websockets.connect(uri) as websocket:
            print(f"Connected to speech recognition server at {uri}")
            print("=" * 60)

            # Send audio file processing requests
            for file_id in test_files:
                wav_path = f"../timit_eval/{file_id}.wav"
                if os.path.exists(wav_path):
                    # Extract WAV metadata for benchmarking
                    metadata = get_wav_metadata(wav_path)
                    if metadata:
                        audio_durations.append(metadata['duration'])
                        audio_file_sizes.append(metadata['file_size'])
                        audio_sample_rates.append(metadata['sample_rate'])

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
                
                print(f"  Raw response: {result}")
                print("-" * 40)
                
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

        # Audio metadata statistics
        if audio_durations:
            import statistics

            # Duration statistics
            avg_duration = statistics.mean(audio_durations)
            std_duration = statistics.stdev(audio_durations) if len(audio_durations) > 1 else 0.0
            total_audio_time = sum(audio_durations)

            # File size statistics
            avg_file_size = statistics.mean(audio_file_sizes)
            std_file_size = statistics.stdev(audio_file_sizes) if len(audio_file_sizes) > 1 else 0.0
            total_file_size = sum(audio_file_sizes)

            # Sample rate statistics
            unique_sample_rates = list(set(audio_sample_rates))

            print(f"\nAUDIO METADATA STATISTICS:")
            print(f"  Sample Count: {len(audio_durations)} files")
            print(f"  Total Audio Time: {total_audio_time:.2f} seconds ({total_audio_time/60:.1f} minutes)")
            print(f"  Total File Size: {total_file_size/1024:.1f} KB ({total_file_size/(1024*1024):.2f} MB)")

            print(f"\nAUDIO DURATION STATS:")
            print(f"  Average: {avg_duration:.3f} seconds")
            print(f"  Std Deviation: {std_duration:.3f} seconds")
            print(f"  Min Duration: {min(audio_durations):.3f} seconds")
            print(f"  Max Duration: {max(audio_durations):.3f} seconds")

            print(f"\nFILE SIZE STATS:")
            print(f"  Average: {avg_file_size/1024:.1f} KB")
            print(f"  Std Deviation: {std_file_size/1024:.1f} KB")
            print(f"  Min Size: {min(audio_file_sizes)/1024:.1f} KB")
            print(f"  Max Size: {max(audio_file_sizes)/1024:.1f} KB")

            print(f"\nSAMPLE RATE INFO:")
            print(f"  Sample Rates Found: {unique_sample_rates} Hz")

            # Processing rate calculation
            processing_rate = total_audio_time / elapsed if elapsed > 0 else 0
            print(f"\nPROCESSING EFFICIENCY:")
            print(f"  Audio-to-Processing Ratio: {processing_rate:.2f}x")
            print(f"  (1.0x = real-time, >1.0x = faster than real-time)")
        else:
            print("\nNo audio metadata collected")

async def batch_transcription_test():
    """Test batch transcription of multiple random TIMIT files."""
    uri = "ws://localhost:8765"

    # Test with random files from the TIMIT dataset (0-199)
    num_files = 10
    random_files = [str(random.randint(0, 199)) for _ in range(num_files)]

    print(f"Testing batch transcription of {num_files} random TIMIT files:")
    print(f"Files: {', '.join(f'{f}.wav' for f in random_files)}")

    # Audio metadata collection for benchmarking
    audio_durations = []
    audio_file_sizes = []
    audio_sample_rates = []
    batch_start_time = time.perf_counter()
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"\nConnected to speech recognition server at {uri}")
            
            # Send all requests quickly
            for file_id in random_files:
                wav_path = f"../timit_eval/{file_id}.wav"
                if os.path.exists(wav_path):
                    # Extract WAV metadata for benchmarking
                    metadata = get_wav_metadata(wav_path)
                    if metadata:
                        audio_durations.append(metadata['duration'])
                        audio_file_sizes.append(metadata['file_size'])
                        audio_sample_rates.append(metadata['sample_rate'])

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

            batch_end_time = time.perf_counter()
            batch_elapsed = batch_end_time - batch_start_time

            print(f"\nBatch transcription of {num_files} files completed!")

            # Display batch benchmark results
            print("\n" + "=" * 60)
            print("BATCH BENCHMARK RESULTS")
            print("=" * 60)
            print(f"Total batch time: {batch_elapsed:.2f} seconds")
            print(f"Average time per file: {batch_elapsed / num_files:.2f} seconds")

            # Audio metadata statistics
            if audio_durations:
                import statistics

                # Duration statistics
                avg_duration = statistics.mean(audio_durations)
                std_duration = statistics.stdev(audio_durations) if len(audio_durations) > 1 else 0.0
                total_audio_time = sum(audio_durations)

                # File size statistics
                avg_file_size = statistics.mean(audio_file_sizes)
                std_file_size = statistics.stdev(audio_file_sizes) if len(audio_file_sizes) > 1 else 0.0
                total_file_size = sum(audio_file_sizes)

                # Sample rate statistics
                unique_sample_rates = list(set(audio_sample_rates))

                print(f"\nAUDIO METADATA STATISTICS:")
                print(f"  Sample Count: {len(audio_durations)} files")
                print(f"  Total Audio Time: {total_audio_time:.2f} seconds ({total_audio_time/60:.1f} minutes)")
                print(f"  Total File Size: {total_file_size/1024:.1f} KB ({total_file_size/(1024*1024):.2f} MB)")

                print(f"\nAUDIO DURATION STATS:")
                print(f"  Average: {avg_duration:.3f} seconds")
                print(f"  Std Deviation: {std_duration:.3f} seconds")
                print(f"  Min Duration: {min(audio_durations):.3f} seconds")
                print(f"  Max Duration: {max(audio_durations):.3f} seconds")

                print(f"\nFILE SIZE STATS:")
                print(f"  Average: {avg_file_size/1024:.1f} KB")
                print(f"  Std Deviation: {std_file_size/1024:.1f} KB")
                print(f"  Min Size: {min(audio_file_sizes)/1024:.1f} KB")
                print(f"  Max Size: {max(audio_file_sizes)/1024:.1f} KB")

                print(f"\nSAMPLE RATE INFO:")
                print(f"  Sample Rates Found: {unique_sample_rates} Hz")

                # Processing rate calculation
                processing_rate = total_audio_time / batch_elapsed if batch_elapsed > 0 else 0
                print(f"\nPROCESSING EFFICIENCY:")
                print(f"  Audio-to-Processing Ratio: {processing_rate:.2f}x")
                print(f"  (1.0x = real-time, >1.0x = faster than real-time)")
            else:
                print("\nNo audio metadata collected")

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