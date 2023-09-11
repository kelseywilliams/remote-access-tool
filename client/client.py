import asyncio
import websockets
from websockets import WebSocketClientProtocol
import mss
from screeninfo import get_monitors
import numpy as np
import sys
import logging
from time import time
np.set_printoptions(threshold=sys.maxsize)

class Client:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.uri = f"ws://{self.host}:{self.port}"
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.logger = _create_logger()
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
        ws_client = await websockets.connect(self.uri)
        self.logger.info(f"Connected to {self.host}:{self.port}")
        keep_alive_task = asyncio.ensure_future(self.keep_alive(ws_client))
        try:
            await self.handler(ws_client)
        except websockets.ConnectionClosed:
            keep_alive_task.cancel()
            await self.disconnect(ws_client)
    
    async def handler(self, server: WebSocketClientProtocol):
        while True:
            message = str(time())
            await server.send(message)
            self.logger.info(f"Sent {message} to {server.remote_address}")
    
    async def disconnect(self, server):
        await server.close()
        self.logger.info(f"Disconnected from server at {self.host}:{self.port}")
        
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

def _create_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("remote-access-tool.server")
    logger.setLevel(logging.INFO)
    # ws_logger is never used
    ws_logger = logging.getLogger('websockets.server')
    ws_logger.setLevel(logging.ERROR)
    ws_logger.addHandler(logging.StreamHandler())
    return logger

async def _wake_up():
    while True:
        await asyncio.sleep(1)

def main():
    client = Client("localhost", 80)
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
