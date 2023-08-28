import socket
import asyncio
import cv2
import numpy as np
from typing import Deque, DefaultDict, Dict
import mss
from screeninfo import get_monitors
host = "127.0.0.1"
port = 8080

def show_image(data):
    # read image as an numpy array
    image = np.asarray(bytearray(data), dtype="uint8")
        
    # use imdecode function
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    cv2.imshow("screen", image)
    if(cv2.waitKey(1) & 0xFF) == ord("q"):
        cv2.destroyAllWindows()

async def client_connected(reader, writer):
    addr = writer.get_extra_info("peername")
    print(f"{addr} has connected.")

    while True:
        recvd = b""
        while data := await reader.readline():
            recvd = recvd + data 
            print(data)
        recvd = recvd[:-1]
        print(f"Total:{recvd.__sizeof__()}")

        show_image(data) 
        
async def main():
    server = await asyncio.start_server(client_connected, host, port)
    addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets)
    print(f"Serving on {addrs}")

    async with server:
        await server.serve_forever()

asyncio.run(main())