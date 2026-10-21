"""Type a message to the running mirror, as if you had said it out loud.

The UI's text box is not wired up in frontend/script.js, so this is the way to
talk to the mirror in --demo mode (no microphone) or in an interview.

    .venv311\Scripts\python.exe say.py "I drank two glasses of water"
"""
import asyncio
import json
import sys

import websockets


async def main(text: str) -> None:
    async with websockets.connect("ws://localhost:8765") as ws:
        await ws.send(json.dumps({"type": "say", "text": text}))
        await asyncio.sleep(0.5)  # let the frame flush before closing
    print("sent:", text)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('usage: python say.py "your message"')
    asyncio.run(main(" ".join(sys.argv[1:])))
