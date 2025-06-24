#!/usr/bin/env python3

import cv2
import struct
import numpy as np
#import time
#from pathlib import Path
import matplotlib.pyplot as plt


framex = 256
framey = 192
metadatay = 4
metadatax = framex
imgy = framey - metadatay
imgx = framex
#framex = 640
#framey = 360
fps = 25
#fps = 30
#fourcc = cv2.VideoWriter.fourcc("Y","U","Y","V")
fourcc = cv2.VideoWriter.fourcc("N","V","1","2")
autofocus = 0
pixel_format = -1

dev = "/dev/video0"
api = cv2.CAP_V4L2

open_params = ()
open_params += (cv2.CAP_PROP_FRAME_WIDTH, framex)
open_params += (cv2.CAP_PROP_FRAME_HEIGHT, framey)
open_params += (cv2.CAP_PROP_FPS, fps)
open_params += (cv2.CAP_PROP_FOURCC, fourcc)
open_params += (cv2.CAP_PROP_CONVERT_RGB, 0)
#open_params += (cv2.CAP_PROP_ZOOM, 0x8004)
#open_params += (cv2.CAP_PROP_FORMAT, -1)


def process_camdat(frame):
    raw = bytes(frame)
    metadata_len = 2*metadatax*metadatay
    metadata = raw[-metadata_len:]
    imgraw = raw[:-metadata_len]
    img_16bit = [x[0] for x in struct.iter_unpack('<h', imgraw)]
    imgdata = np.array(img_16bit, np.uint16).reshape(imgy, imgx)
    return metadata, imgdata


def decode_fourcc(v):
    v = int(v)
    return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])


cam = cv2.VideoCapture()

#while not Path(dev).is_char_device():
#    pass
frame = None
try:
    is_open = cam.open(dev, api, open_params)
    cam.set(cv2.CAP_PROP_ZOOM, 0x8000)  # raw mode

    if is_open:
        print("OPEN!")
        print(f"{cam.get(cv2.CAP_PROP_FRAME_WIDTH)=}")
        print(f"{cam.get(cv2.CAP_PROP_FRAME_HEIGHT)=}")
        print(f"{cam.get(cv2.CAP_PROP_FPS)=}")
        print(f"{cam.get(cv2.CAP_PROP_FOURCC)=}")
        print(f"wantfourcc={fourcc}")
        grabbed = cam.grab()
        if grabbed:
            retval, frame = cam.retrieve()
            print(f"{retval=}")

    else:
        print("Not open :-(")
except Exception as e:
    print(f"FAILFAIL: {e}")
finally:
    cam.release()

#if frame is not None:
if frame is not None:
    frame = frame.reshape(-1,framex)
    #metadata, imgdata = process_camdat(frame)
    #imgdata = (imgdata/(2**15)*2**8).round().astype(np.uint8)
    #metadata = proper_shape[-4:, :]
    #imgdata = proper_shape[:-4,:]
    #img_data = frame[4:,:]
    #img = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)
    #img = cv2.cvtColor(imgdata, cv2.COLOR_RGBA2BGRA)
    #cvuint8 = cv2.convertScaleAbs(img)
    #img = cv2.cvtColor(imgdata, cv2.COLOR_BGR2YUV_YV12)
    #img = cv2.imdecode(imgdata, cv2.IMREAD_GRAYSCALE)
    #img = cv2.IMREAD_GRAYSCALE
    #window_name = "Video"
    #cv2.namedWindow(window_name)
    #cv2.imshow(window_name, img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #x = range(imgx)
    #y = range(imgy)
    #X, Y = np.meshgrid(x, y)
    #plt.pcolor(X, Y, imgdata)
    #plt.show()
    #z = np.array

    convert_options = [v for v in dir(cv2) if v.startswith("COLOR_")]
    decode_options = [v for v in dir(cv2) if v.startswith("IMREAD_")]

    print(f"Started with {len(list(convert_options))}")
    success = []
    for opt in convert_options:
        try:
            img = cv2.cvtColor(frame, getattr(cv2, opt))
            success.append(opt)
            cv2.imwrite(f"/tmp/imgs/{opt}.png", img)
        except:
            pass
    for opt in decode_options:
        try:
            img = cv2.decode_options(frame, getattr(cv2, opt))
            success.append(opt)
            cv2.imdecode(f"/tmp/imgs/{opt}.png", img)
        except:
            pass
    
    print(f"Ended with {len(success)}")