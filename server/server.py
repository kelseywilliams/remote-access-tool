# Base code written by stefanotorresi
# https://gist.github.com/stefanotorresi/bfb9716f73f172ac4439436456a96d6f

import websockets
from websockets import WebSocketServerProtocol
import asyncio
import logging


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

    def listen(self): 
        self.logger.info(f"listening on {self.host}:{self.port}")
        ws_server = websockets.serve(self.connect_client, self.host, self.port)
        
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
        except websockets.ConnectionClosed:
            keep_alive_task.cancel()
            await self.disconnect_client(client)

    async def handle_messages(self, client):
        while True:
            message = await client.recv()
            self.logger.info(f"recieved message from {client.remote_address}: {message}")
            #await asyncio.wait([client.send(message) for client in self.clients])

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
    server = Server("localhost", 80)
    server.listen()

if __name__ == "__main__":
    main()