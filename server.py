# server.py
import asyncio
import websockets
from custom_model import CustomLanguageModel

# Initialize custom language model
custom_model = CustomLanguageModel()

async def handle_client(websocket, path):
    print("Client connected")
    try:
        async for message in websocket:
            print(f"Received message: {message}")

            # Process the message and send response back to the client
            response = custom_model.generate_response(message)
            await websocket.send(response)
    except websockets.exceptions.ConnectionClosedError:
        print("Client disconnected")

start_server = websockets.serve(handle_client, "localhost", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
