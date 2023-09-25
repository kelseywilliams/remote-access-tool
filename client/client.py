import asyncio
from websockets import client as ws
from websockets.client import WebSocketClientProtocol
from websockets.exceptions import ConnectionClosed
import mss
from screeninfo import get_monitors
import numpy as np
import sys
import logging
import zlib
import cv2
from PIL import Image
import json

np.set_printoptions(threshold=sys.maxsize)

# TODO Set timeouts properly and attempt reconnect.  Sometimes random huge pieces of data jam the connection
class Client:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.uri = f"ws://{self.host}:{self.port}"
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.logger = create_logger()
        self._server_timeout = 5
        self._wake_up_task = None
    def start(self):
        #self.logger.info(f"Connected to {self.host}:{self.port}")
        self.loop.run_until_complete(self.connect())
        self._wake_up_task = asyncio.ensure_future(_wake_up(), loop=self.loop)

        try:
            self.loop.run_forever()
        except KeyboardInterrupt:
            self.logger.debug("caught keyboard interrupt")
            self.exit()

    async def connect(self):
        ws_client = await ws.connect(self.uri)
        self.logger.info(f"Connected to {self.host}:{self.port}")
        keep_alive_task = asyncio.ensure_future(self.keep_alive(ws_client))
        try:
            await self.handler(ws_client)
        except ConnectionClosed:
            keep_alive_task.cancel()
            await self.disconnect(ws_client)
    
    async def handler(self, server: WebSocketClientProtocol):
        monitors = get_monitors()
        monitor = monitors[0]
        bounding_box = {"top":0, "left":0, "width":monitor.width, "height":monitor.height}
        print(f"{monitor.width}x{monitor.height}")
        sct = mss.mss()
        while True:
            # Send resolution
            res = f"{monitor.width}x{monitor.height}"
            await server.send(res)
             # Grab screen image
            frame = sct.grab(bounding_box)
            frame_raw = frame.raw
            message = zlib.compress(frame_raw)
            await server.send(message)
            self.logger.info(f"Sent {sys.getsizeof(message)} to {server.remote_address}")

    
    async def disconnect(self, server):
        await server.close()
        self.logger.debug(f"Disconnected from server at {self.host}:{self.port}:Code {server.close_code}", stack_info=False)
        
    async def keep_alive(self, server: WebSocketClientProtocol):
        while True:
            await asyncio.sleep(self._server_timeout)
            try:
                self.logger.debug(f"pinging {self.host}:{self.port}")
                await asyncio.wait_for(server.ping(), self._server_timeout)
            except asyncio.TimeoutError:
                self.logger.info(f"Server {self.host}:{self.port} timed out")
    
    def exit(self):
        self.logger.info("exiting")
        self._wake_up_task.cancel()
        try:
            self.loop.run_until_complete(self._wake_up_task)
        except asyncio.CancelledError:
            self.loop.close()

def create_logger():
    fmt="%(pathname)s:%(funcName)s():%(lineno)i: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)
    logger = logging.getLogger("remote-access-tool.client")
    logger.setLevel(logging.INFO)
    return logger

async def _wake_up():
    while True:
        await asyncio.sleep(1)

def main():
    host = input("host:")
    port = int(input("port:"))
    client = Client(host, port)
    client.start()

if __name__ == "__main__":
    main()

'''
async def main():
    reader, writer = await asyncio.open_connection("127.0.0.1", 8080)
    monitors = get_monitors()
    monitor = monitors[0]
    bounding_box = {"top":0, "left":0, "width":monitor.width, "height":monitor.height}
    sct = mss.mss()

    while True:
        # Grab screen image
        sct_img = sct.grab(bounding_box)
        sct_img = mss.tools.to_png(sct_img.rgb, sct_img.size)
        #sct_img = zlib.compress(sct_img)
        # Print size of image
        img_size = sys.getsizeof(sct_img)
        img_size_str = str(img_size)

        # Send the image size to the server
        writer.write(img_size_str.encode())
        writer.write(b"\n")
        await writer.drain()
        
        # Break up image into 64kb chunks
        # Start by turning the image array into a numpy array
        img_array = np.frombuffer(sct_img, dtype=np.uint8)
        # The image will be broken up into 64kb chunks.
        chunks = np.array_split(img_array, np.arange(CHUNK_SIZE, len(img_array), CHUNK_SIZE))
        # ***** TODO Remove this test code *****
        total_mem = 0
        for chunk in chunks:
            print(chunk.nbytes)
            total_mem += chunk.nbytes
        print(f"Len of chunks: {len(chunks)}, Size of chunks: {total_mem}")

        chunks_length = len(chunks)
        for i in range(chunks_length):
            writer.writelines(chunks[i])
            await writer.drain()
            '''
