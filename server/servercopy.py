import sys
import socket
import selectors
import types
import asyncio
import cv2
import numpy as np

host = "127.0.0.1"
port = 8888

async def handler(reader, writer):
    data = await reader.read(100)
    addr = writer.get_extra_info("peername")
    print(f"{addr}:{data}")
    
    cv2.imshow("screen", np.array(data))
    if(cv2.waitKey(1) & 0xFF) == ord('q'):
		cv2.destroyAllWindows()
		break
    loop = asyncio.get_running_loop()
    rsock, wsock = socket.socketpair()
    reader, write = await asyncio.open_connection(sock = rsock)
    loop.call_soon(wsock.send, "abc".encode())
    data = await reader.read(100)
    
    print("Received:", data)

async def main():
    server = await asyncio.start_server(handler, host, port)
    addrs = ', '.join(str(sock.getsockname()) for sock in server.sockets)
    print(f"Serving on {addrs}")
    
    async with server:
         await server.serve_forever
asyncio.run(main())