#!/usr/bin/env python3

import struct
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from itertools import batched
#import time
from functools import partial
from linuxpy.video.device import Device, VideoCapture
#import skimage
#import cv2


class IR:
    dev = "/dev/video0"
    stream_mode_index = 0

    img_y_start = 0
    img_y_end = 192
    pixel_data_type = "<H"


    plot_xmin = -1
    plot_xmax = 1
    plot_ymin = -1
    plot_ymax = 1
    #plot_colormap = "viridis"
    plot_colormap = "Grays"

    debug = False
    enable_colorbar = True  # could limit fps when true
    t_last = 0
    fps = 0
    fps_warning_limit = 0

    def __init__(self):
        self.cam = Device(self.dev)
        self.img_y = self.img_y_end - self.img_y_start

    def __enter__(self):
        self.cam = self.cam.__enter__()

        return self
    
    def __exit__(self, type, value, traceback):
        return self.cam.__exit__(type, value, traceback)

    def setup_capture(self) -> VideoCapture:
        capture = VideoCapture(self.cam)
        frame_sizes = self.cam.info.frame_sizes
        if not frame_sizes:
            raise RuntimeError(f"{self.dev} seems to have no capture modes")
        if self.debug:
            for i, fs in enumerate(frame_sizes):
                print(f"{i}: {fs}")
            print(f"Using {self.stream_mode_index}")
        self.cap_format = frame_sizes[self.stream_mode_index]
        self.fps_warning_limit = self.cap_format.min_fps.numerator/self.cap_format.min_fps.denominator * 0.9

        self.raw_x = self.cap_format.width
        self.img_x = self.cap_format.width
        self.metadata_x = self.cap_format.width
        self.raw_y = self.cap_format.height
        capture.set_format(self.cap_format.width, self.cap_format.height, self.cap_format.pixel_format.human_str())
        print(f"Starting capture: {capture.get_format().pixel_format.human_str()}@{capture.get_format().width}x{capture.get_format().height}")
        self.npxformat = self.structtype2numpytype(self.pixel_data_type)
        return capture

    def structtype2numpytype(self, strformat):
        if "b" in strformat:
            npdtype = np.byte
        elif "B" in strformat:
            npdtype = np.ubyte
        elif "h" in strformat:
            npdtype = np.short
        elif "H" in strformat:
            npdtype = np.ushort
        elif "i" in strformat:
            npdtype = np.intc
        elif "I" in strformat:
            npdtype = np.uintc
        elif "l" in strformat:
            npdtype = np.long
        elif "L" in strformat:
            npdtype = np.ulong
        elif "q" in strformat:
            npdtype = np.longlong
        elif "Q" in strformat:
            npdtype = np.ulonglong
        elif "e" in strformat:
            npdtype = np.half
        elif "f" in strformat:
            npdtype = np.single
        elif "d" in strformat:
            npdtype = np.double
        else:
            npdtype = np.uint8
        return npdtype

    def process_camdat(self, frame):
        t = frame.timestamp
        if self.t_last:
            self.fps = 1/(t - self.t_last)
            if self.debug:
                if self.fps < self.fps_warning_limit:
                    print(f"Frame missed! FPS={self.fps}")
        self.t_last = t
        with open(f"/tmp/IR_raw_{frame.frame_nb}.bin", "wb") as dump:
            dump.write(frame.data)

        row_len = len(frame.data) / frame.height
        if row_len.is_integer():
            row_len = int(row_len)
        else:
            raise ValueError("Frame data does not divide evently into image widths")
        img_start = int(self.img_y_start*row_len)
        img_end = int(self.img_y_end*row_len)
        imgraw = frame.data[img_start:img_end]
        metadata = frame.data[0:img_start] + frame.data[img_end:-1]
        metadata_rows = [bytes(x) for x in batched(metadata, row_len)]

        img_unpacked = [x[0] for x in struct.iter_unpack(self.pixel_data_type, imgraw)]
        imgdata = np.array(img_unpacked, self.npxformat).reshape(self.img_y, -1)
        imgdata_flipped = np.flipud(imgdata)

        with open(f"/tmp/IR_img_{frame.frame_nb}.bin", "wb") as dump:
            dump.write(imgraw)

        with open(f"/tmp/IR_meta_{frame.frame_nb}.bin", "wb") as dump:
            dump.write(metadata)


        meta000 = metadata_rows[0][0:]+b''
        meta148 = metadata_rows[148]
        unpack000 = [x[0] for x in struct.iter_unpack('<f', meta000)]
        unpack148 = [x[0] for x in struct.iter_unpack('<H', meta148)]
        strd000 = [f"{x:+08.2e}" for x in unpack000]
        strd148 = [f"{x:05.0f}" for x in unpack148]
        #print(strd000)
        #print(strd148[0:17])
        #0 =FFFF
        #1 61166
        #2
        #3
        #4
        #5
        #6
        #7
        #8 avg
        #9 center
        #10 counter
        #11 255
        #12 255
        #13
        #14
        #15 max
        #16 min
        
        

        return metadata_rows, imgdata_flipped


    def frame_please(self, capture):
        return next(capture.__iter__())

    def framegen(self, capture):
        while True:
            yield self.process_camdat(self.frame_please(capture))[1]

    def video(self, capture):
        fig = plt.figure()
        ax = plt.gca()
        imgdata = self.process_camdat(self.frame_please(capture))[1]
        x = np.linspace(self.plot_xmin, self.plot_xmax, imgdata.shape[1])
        y = np.linspace(self.plot_ymin, self.plot_ymax, imgdata.shape[0])
        X, Y = np.meshgrid(x, y)
        pcm = ax.pcolormesh(X,Y,imgdata, cmap=self.plot_colormap, aa=True)
        #im = ax.imshow(imgdata)
        if self.enable_colorbar:
            # recomputing the colorbar seems to slow down the rendering a lot
            ax.figure.colorbar(pcm, ax=ax)

        def animate(imgdata):
            pcm.set_array(imgdata)
            #im.set_data(imgdata)
            pcm.set_clim(np.min(imgdata),np.max(imgdata))

            return ()
        
        frames = partial(self.framegen, capture)
        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=0, cache_frame_data=False, blit=True)
        plt.show()



if __name__ == "__main__":
    with IR() as ir:
        with ir.setup_capture() as cap:
            #while True:
                # meta, img = ir.process_camdat(ir.frame_please(cap))
                # meta000 = meta[0]
                # meta148 = meta[148]
                # unpack000 = [x[0] for x in struct.iter_unpack('B', meta000)]
                # unpack148 = [x[0] for x in struct.iter_unpack('<H', meta148)]
                # strd000 = [f"{x:03.0f}" for x in unpack000]
                # strd148 = [f"{x:05.0f}" for x in unpack148]
                # #print(strd000)
                # print(strd148[0:17])

            ir.video(cap)
        # i = 0
        # ir.cam.set(cv2.CAP_PROP_ZOOM, 0x8020)  # raw mode
        # while i<-1:
        #     metadata, data = ir.process_camdat(ir.frame_please())
        #     if metadata:
        #         md = metadata[0][0:]+ b'0'
        #         if i%500 == 0 and i != 0:
        #             time.sleep(3)
        #             ir.cam.set(cv2.CAP_PROP_ZOOM, 0x8020)  # raw mode
        #             print("ZOOM")
        #             time.sleep(3)
        #         data = [x[0] for x in struct.iter_unpack('<f', md)]
        #         print([f"{x:+012g}" for x in data])
        #         #print([f"{x:+015g}" for x in data])
        
        #     i +=1
        # ir.video()
