import numpy as np
import cv2
from mss import mss
from PIL import Image
from screeninfo import get_monitors
import socket

# Establish connection to server
host = input("Host:")
port = input("Port:")

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
    stream.bind((Host, Port))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")

monitors = get_monitors()
monitor = 0

if len(monitors) > 1:
    monitor = input(f"{len(monitors)} screens have been detected. {monitors}\n Please choose a screen number.")

width = monitors[monitor].width
height = monitors[monitor].height

bounding_box = {'top': 0,'left': 0, 'width': width, 'height': height}

sct = mss()

while True:
    sct_img = sct.grab(bounding_box)
    cv2.imshow('screen', np.array(sct_img))

    if(cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break