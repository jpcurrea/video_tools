#!/usr/bin/env python

import io
import os
import time
import subprocess
from datetime import datetime
import picamera
import picamera.array
import numpy as np
from PIL import Image, ImageDraw


# FILE_PATTERN = './motion%02d.h264' # the file pattern in which to record videos
# FILE_BUFFER = 1048576            # the size of the file buffer (bytes)

REC_RESOLUTION = (1280, 720) # the recording resolution
REC_FRAMERATE = 60           # the recording framerate
MAX_TIME = 60*30             # maximum time of recording for this session in seconds
BUFFER_SECONDS = 3           # number of seconds to store in ring buffer
END_PAD = 3                  # number of seconds to wait after motion stops
REC_BITRATE = 1000000        # bitrate for H.264 encoder

MOTION_MAGNITUDE = 30        # the magnitude of vectors required for motion
MOTION_VECTORS = 5           # the number of vectors required to detect motion


class MotionDetector(picamera.array.PiMotionAnalysis):
    def __init__(self, camera, size=None):
        super(MotionDetector, self).__init__(camera, size)
        self.vector_count = 0
        self.detected = 0

    def analyse(self, a):
        a = np.sqrt(
            np.square(a['x'].astype(np.float)) +
            np.square(a['y'].astype(np.float))
            ).clip(0, 255).astype(np.uint8)
        vector_count = (a > MOTION_MAGNITUDE).sum()  # count of vectors exceeding threshold
        if vector_count > MOTION_VECTORS:
            self.detected = time.time()
            self.vector_count = vector_count


def merge_before_after(folder="./"):
    fns = os.listdir(folder)
    fns = [fn for fn in fns if fn.endswith(".h264")]
    befores = [fn for fn in fns if "before" in fn]
    afters = [fn for fn in fns if "after" in fn]
    timestamps = []
    for fn in fns:
        flist = fn.split("_")
        timestamps += ["_".join(flist[2:])]
    timestamps = set(timestamps)
    for ts in tuple(timestamps):
        before = [val for val in befores if ts in val]
        after = [val for val in afters if ts in val]
        if all([len(before) > 0, len(after) > 0]):
            before = before[0]
            after = after[0]
            with open("./temp_list.txt", 'w') as fname:
                fname.write("file \'{}\'\nfile \'{}\'".format(before, after))
            subprocess.call(
                ["ffmpeg", "-f", "concat", "-safe", "0",
                 "-i", "./temp_list.txt",
                 "-c", "copy",
                 # "-r", "60",
                 # "-filter:v", "setpts=0.5*PTS",
                 # ts])
                 ts])
            os.remove("./temp_list.txt")
    for fn in befores + afters:
        os.remove(fn)

def create_recording_overlay(camera):
    # Make a recording symbol (red circle) overlay. This isn't perfect as
    # overlays don't support alpha transparency (so there'll be black corners
    # around the red circle) but oh well, it's only a demo!
    img = Image.new('RGB', (64, 64))
    d = ImageDraw.Draw(img)
    d.ellipse([(0, 0), (63, 63)], fill='red')
    o = camera.add_overlay(img.tostring(), size=img.size)
    o.alpha = 128
    o.layer = 1
    o.fullscreen = False
    o.window = (32, 32, 96, 96)
    return o


def main():
    with picamera.PiCamera() as camera:
        camera.resolution = REC_RESOLUTION
        camera.framerate = REC_FRAMERATE
        # Let the camera settle for a bit. This avoids detecting motion when
        # it's just the white balance and exposure settling.
        time.sleep(4)

        camera.start_preview()
        recording_overlay = create_recording_overlay(camera)
        ring_buffer = picamera.PiCameraCircularIO(
            camera, seconds=BUFFER_SECONDS)
        file_number = 1
        # file_output = io.open(
        #     FILE_PATTERN % file_number, 'wb', buffering=FILE_BUFFER)
        motion_detector = MotionDetector(camera)

        # Start recording data to the ring buffer and the motion detector
        # at the specified bitrates
        camera.start_recording(
            ring_buffer, format='h264', intra_period=REC_FRAMERATE,
            motion_output=motion_detector)

        # import pdb; pdb.set_trace()
        time_left = MAX_TIME
        try:
            while True:
                # wait around for the motion detector to notice something.
                buffer_start = time.time()
                print('Waiting for motion')
                while motion_detector.detected < time.time() - 1:
                    camera.wait_recording(1)
                print('Motion detected (%d vectors)' % motion_detector.vector_count)
                buffer_stop = time.time()
                
                # split recording to new file for video after movement
                time_string = datetime.fromtimestamp(buffer_start).timetuple()
                time_string = list(time_string[:-3])
                time_string = [str(val) for val in time_string]
                time_string = "_".join(time_string)
                camera.split_recording(
                    'after_{}_{}.h264'.format(file_number, time_string))
                buffer_duration = buffer_stop - buffer_start
                time_left -= min(buffer_duration, BUFFER_SECONDS)
                vid_start = time.time()

                # turn on recording indicator
                recording_overlay.layer = 3

                # save buffer of video from before motion 
                ring_buffer.copy_to(
                    'before_{}_{}.h264'.format(file_number, time_string),
                    seconds=BUFFER_SECONDS)
                # then clear buffer
                ring_buffer.clear()

                # wait until motion stops for END_PAD amount of time
                # or until vid length reaches MAX_TIME
                vid_stop = time.time()
                while (motion_detector.detected > time.time() - END_PAD
                       and vid_stop - vid_start < time_left):
                    camera.wait_recording(1)
                    vid_stop = time.time()
                time_left -= vid_stop - vid_start

                camera.split_recording(ring_buffer)
                recording_overlay.layer = 1
                if time_left <= 0:
                    print("{} seconds of motion recorded. DONE!".format(MAX_TIME))
                    break

                # split video again, back to ring buffer
                file_number += 1
        finally:
            camera.stop_preview()
            camera.stop_recording()
            merge_before_after()

if __name__ == '__main__':
    main()
