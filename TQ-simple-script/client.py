import asyncio
import websockets

async def send_and_receive():
    uri = "ws://localhost:8765"
    messages_to_send = ["audio_chunk_1", "audio_chunk_2", "audio_chunk_3"]
    results_received = 0
    total_tasks = len(messages_to_send)

    try:
        async with websockets.connect(uri) as websocket:
            print(f"Connected to {uri}")

            for msg in messages_to_send:
                print(f"> Sending: {msg}")
                await websocket.send(msg)
                response = await websocket.recv()
                print(f"< Received: {response}")
                await asyncio.sleep(0.5)
            print("Finished sending messages. Waiting for results...")

            while results_received < total_tasks:
                result = await websocket.recv()
                print(f"< Result: {result}")
                results_received += 1

    except ConnectionRefusedError:
        print("Connection refused. Is the server running?")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(send_and_receive())