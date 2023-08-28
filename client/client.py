import asyncio
import mss
from screeninfo import get_monitors
import time

async def main():
    reader, writer = await asyncio.open_connection("127.0.0.1", 8080)
    monitors = get_monitors()
    monitor = monitors[0]
    bounding_box = {"top":0, "left":0, "width":monitor.width, "height":monitor.height}
    sct = mss.mss()
    while True:
        sct_img = sct.grab(bounding_box)
        sct_img = mss.tools.to_png(sct_img.rgb, sct_img.size)
        img_size = sct_img.__sizeof__()
        print(img_size)
        writer.write(sct_img)
        writer.write("\n".encode())
        await writer.drain()
        time.sleep(15)

asyncio.run(main())

