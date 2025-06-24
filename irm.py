#!/usr/bin/env python3

import cv2
import struct
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from itertools import batched
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable


class IR:
    raw_x = 256

    raw_y = 192
    img_y_start = 0
    img_y_end = 192
    fourcc = "NV12"
    Bpp = 1

    #raw_y = 192
    #img_y_start = 0
    #img_y_end = 192
    #fourcc = "YUYV"
    #Bpp = 2


    fps = 25
    dev = "/dev/video0"
    api = cv2.CAP_V4L2

    open_params = ()
    cam = cv2.VideoCapture
    debug = True

    def __init__(self):
        self.img_x = self.raw_x
        self.img_y = self.img_y_end - self.img_y_start

        self.metadata_x = self.raw_x
        self.metadata_y = self.raw_y - self.img_y


        self.fourcc_code = cv2.VideoWriter.fourcc(*tuple(self.fourcc))
        #autofocus = 0
        #pixel_format = -1


        self.open_params = ()
        self.open_params += (cv2.CAP_PROP_FRAME_WIDTH, self.raw_x)
        self.open_params += (cv2.CAP_PROP_FRAME_HEIGHT, self.raw_y)
        self.open_params += (cv2.CAP_PROP_FPS, self.fps)
        self.open_params += (cv2.CAP_PROP_FOURCC, self.fourcc_code)
        self.open_params += (cv2.CAP_PROP_CONVERT_RGB, 0)
        #self.open_params += (cv2.CAP_PROP_ZOOM, 0x8004)
        #self.open_params += (cv2.CAP_PROP_FORMAT, -1)
        self.cam = cv2.VideoCapture()


    def __enter__(self):
        try:
            if self.cam.open(self.dev, self.api, self.open_params):
                self.cam.set(cv2.CAP_PROP_ZOOM, 0x8004)  # raw mode
                if self.debug:
                    print(f"{self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)=}")
                    print(f"{self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)=}")
                    print(f"{self.cam.get(cv2.CAP_PROP_FPS)=}")
                    print(f"{self.cam.get(cv2.CAP_PROP_FOURCC)=}")
                    print(f"wantfourcc={self.fourcc_code}")
        except Exception as e:
            print(f"Could not open camera: {e}")
            self.cam.release()

        return self
    
    def __exit__(self, type, value, traceback):
        self.cam.release()
        if self.debug:
            print("Released.")
        return None

    def process_camdat(self, frame):
        # correct is:
        raw = frame.reshape(-1,self.raw_x)
        answer = cv2.cvtColor(raw, cv2.COLOR_YUV2GRAY_NV12)
        #raw = bytes(frame)
        raw = frame.reshape(-1,self.raw_x)
        img_start = self.img_y_start*self.raw_x*self.Bpp
        img_end = self.img_y_end*self.raw_x*self.Bpp
        imgraw = raw[img_start:img_end]
        metadata = raw[0:img_start] + raw[img_end:-1]
        metadata_rows = [bytes(x) for x in batched(metadata, 2*self.metadata_x)]

        # metadata_len = 2*self.metadata_x*self.metadata_y
        # if metadata_len > 0:
        #     metadata = raw[-metadata_len:]
        #     split_metadata = batched(metadata, 2*self.metadata_x)
        #     meta_ret = [bytes(x) for x in split_metadata]
        #     imgraw = raw[:-metadata_len]
        # else:
        #     meta_ret = None
        #     imgraw = raw
        img_16bit = [x[0] for x in struct.iter_unpack('<H', imgraw)]
        imgdata = np.array(img_16bit, np.uint16).reshape(self.img_y, self.img_x)
        return metadata_rows, imgdata


    def frame_please(self):
        grabbed = False
        while not grabbed:
            grabbed = self.cam.grab()
        success, frame = self.cam.retrieve()
        return frame

    def framegen(self):
        while True:
            yield self.process_camdat(self.frame_please())[1]

    def video(self):
        fig = plt.figure()
        ax = plt.gca()
        imgdata = self.process_camdat(self.frame_please())[1]
        im = ax.imshow(imgdata)
        ax.figure.colorbar(im, ax=ax)

        def animate(imgdata):
            im.set_data(imgdata)
            im.set_clim(np.min(imgdata),np.max(imgdata))

            return ()
        
        anim = animation.FuncAnimation(fig, animate, frames=self.framegen, interval=0, cache_frame_data=False, blit=True)
        plt.show()



if __name__ == "__main__":
    with IR() as ir:
        i = 0
        ir.cam.set(cv2.CAP_PROP_ZOOM, 0x8020)  # raw mode
        while i<-1:
            metadata, data = ir.process_camdat(ir.frame_please())
            if metadata:
                md = metadata[0][0:]+ b'0'
                if i%500 == 0 and i != 0:
                    time.sleep(3)
                    ir.cam.set(cv2.CAP_PROP_ZOOM, 0x8020)  # raw mode
                    print("ZOOM")
                    time.sleep(3)
                data = [x[0] for x in struct.iter_unpack('<f', md)]
                print([f"{x:+012g}" for x in data])
                #print([f"{x:+015g}" for x in data])
        
            i +=1
        ir.video()
