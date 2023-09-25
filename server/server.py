# Base code written by stefanotorresi
# https://gist.github.com/stefanotorresi/bfb9716f73f172ac4439436456a96d6f

from websockets import server as ws
from websockets.server import WebSocketServerProtocol
from websockets.exceptions import ConnectionClosed
import asyncio
import logging
import sys
import cv2
import zlib
import numpy as np
from PIL import Image
from time import process_time_ns

class Server:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.clients = set()
        # Loop handling set to satisfy new py 3.11 asyncio reqs https://stackoverflow.com/questions/73361664
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.logger = _create_logger()
        self._client_timeout = 5
        self._wake_up_task = None
        self.process_time = []

    def listen(self): 
        self.logger.info(f"listening on {self.host}:{self.port}")
        ws_server = ws.serve(self.connect_client, self.host, self.port)
        
        self.loop.run_until_complete(ws_server)
        self._wake_up_task = asyncio.ensure_future(_wake_up(), loop=self.loop)

        # TODO Implement shell mode and silent mode
        try:
            self.loop.run_forever()
        except KeyboardInterrupt:
            self.logger.debug('caught keyboard interrupt')
            self.exit()

    async def connect_client(self, client: WebSocketServerProtocol, path):
        self.clients.add(client)
        self.logger.info('new client connected from {}:{}'.format(*client.remote_address))
        keep_alive_task = asyncio.ensure_future(self.keep_alive(client))
        try:
            await self.handle_messages(client)
        except ConnectionClosed or KeyboardInterrupt:
            keep_alive_task.cancel()
            await self.disconnect_client(client)

    async def handle_messages(self, client):
        windowName = str(client.remote_address)
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        while True:
            sw = process_time_ns()
            res = await client.recv()
            res = res.split("x")
            width = int(res[0])
            height = int(res[1])
            raw = await client.recv()
            cv2.resizeWindow(windowName, width, height)
            message = zlib.decompress(raw)
            try:
                frame = Image.frombytes('RGBA', (width, height), message)
            except Exception as e:
                print(f"Exception on line 61: {e}")

            frame = np.array(frame)
            text = str(self.get_fps()) + " fps"
            coordinates = (10,25)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (0,0,0)
            thickness = 2
            frame = cv2.putText(frame, text, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)
            self.logger.info(f"recieved message from {windowName}: {sys.getsizeof(message)}")
            cv2.imshow(windowName, frame)
            if len(self.process_time) > 1000:
                self.process_time.pop(0)
            self.process_time.append(process_time_ns() - sw)
            cv2.waitKey(1)
            if cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) <1:
                cv2.destroyAllWindows()
                await self.disconnect_client(client)
                break
        await asyncio.wait([client.send(message) for client in self.clients])

    async def disconnect_client(self, client):
        await client.close()
        self.clients.remove(client)
        self.logger.info('client {}:{} disconnected'.format(*client.remote_address))

    # TODO Determine: is this a necessary function? Does it simply poll client?
    async def keep_alive(self, client: WebSocketServerProtocol):
        while True:
            await asyncio.sleep(self._client_timeout)
            try:
                self.logger.debug('pinging {}:{}'.format(*client.remote_address))
                await asyncio.wait_for(client.ping(), self._client_timeout)
            except asyncio.TimeoutError:
                self.logger.info('client {}:{} timed out'.format(*client.remote_address))
                await self.disconnect_client(client)

    def exit(self):
        self.logger.info("exiting")
        self._wake_up_task.cancel()
        try:
            self.loop.run_until_complete(self._wake_up_task)
        except asyncio.CancelledError:
            self.loop.close()

    def get_fps(self):
        time = 0
        for pt in self.process_time:
            time = time + pt
        # TODO clean this time code up.  what is this
        if len(self.process_time) == 0:
            return 0
        return int(1000000000/ (time / len(self.process_time)))

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
    server = Server("127.0.0.1", 80)
    server.listen()

if __name__ == "__main__":
    main()