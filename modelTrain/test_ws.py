import asyncio, websockets, json

async def test():
    async with websockets.connect("ws://localhost:3000/ws") as ws:
        # Calibrate
        await ws.send(json.dumps({"type": "calibrate"}))
        print("Sent: calibrate")

        # Send angles
        await ws.send(json.dumps({"type": "angles", "x": 12.5, "y": -3.2}))
        print("Sent: angles x=12.5 y=-3.2")

asyncio.run(test())