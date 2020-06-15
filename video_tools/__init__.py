import os
import numpy as np
import subprocess
from scipy import ndimage
import skimage
from skvideo import io
from skvideo.io import ffprobe
from fly_eye import *
from bird_call import *

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def print_progress(part, whole):
    import sys
    prop = float(part)/float(whole)
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ("="*int(20*prop), 100*prop))
    sys.stdout.flush()

def smooth(arr, sigma=5):
    """A 2d smoothing filter for the heights array"""
    # arr = arr.astype("float32")
    fft2 = np.fft.fft2(arr)
    ndimage.fourier_gaussian(fft2, sigma=sigma, output=fft2)
    positive = np.fft.ifft2(fft2).real
    return positive

def center_of_mass(arr):
    blur = smooth(arr, 20)
    center = skimage.feature.peak_local_max(blur, num_peaks=1)
    if len(center) > 0:
        center = center[0]
    else:
        center = np.empty(2, dtype=float)
        center.fill(np.nan)
    return center

class Video1(Stack):
    """Takes a stack of images, or a video that is converted to a stack of images,
    and uses common functions to track motion or color."""

    def __init__(self, filename, fps=30, f_type=".jpg"):
        self.vid_formats = [
            '.mov',
            '.mp4',
            '.mpg',
            '.avi']
        self.filename = filename
        self.f_type = f_type
        self.track = None
        self.colors = None
        if ((os.path.isfile(self.filename) and
             self.filename.lower()[-4:] in self.vid_formats)):
            # if file is a video, use ffmpeg to generate a jpeg stack
            self.dirname = self.filename[:-4]
            self.fps = subprocess.check_output([
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=avg_frame_rate", "-of",
                "default=noprint_wrappers=1:nokey=1",
                self.filename])
            self.fps = int(str(fps))
            # self.fps = int(self.fps.split("/"))
            # self.fps = float(self.fps[0])/float(self.fps[1])
            if os.path.isdir(self.dirname) is False:
                os.mkdir(self.dirname)
            try:
                if len(os.listdir(self.dirname)) == 0:
                    failed = subprocess.check_output(
                        ["ffmpeg", "-i", self.filename,
                         "-vf", "scale=720:-1",
                         "./{}/frame%05d{}".format(self.dirname, self.f_type)])
            except subprocess.CalledProcessError:
                print("failed to parse video into {}\
                stack!".format(self.f_type))
            try:
                self.audio_fname = "{}.wav".format(self.filename[:-4])
                if os.path.exists(self.audio_fname):
                    resp = None
                    while resp not in ['0', '1']:
                        resp = input("Audio file found. Press <1> to load this file or <0> to extract the files again.")
                    if resp == '0':
                        failed = subprocess.check_output(
                            ["ffmpeg", "-i", self.filename, "-f", "wav",
                             "-ar", "44100",
                             "-ab", "128",
                             "-vn", self.audio_fname, '-y'])
                    self.audio = Recording(self.audio_fname, trim=False)
            except subprocess.CalledProcessError:
                print("failed to get audio from video!")

        elif os.path.isdir(self.filename):
            self.dirname = self.filename
            self.fps = fps
        Stack.__init__(self, self.dirname, f_type=self.f_type)

    def select_color_range(self, samples=5):
        color_range = []
        intervals = int(round(len(self.layers)/samples))
        for l in self.layers[::intervals]:
            l.select_color()
            color_range += [l.cs.colors]
            l.image = None
        color_range = np.array(color_range)
        self.colors = np.array([color_range.min((0, 1)),
                                color_range.max((0, 1))])

    def track_foreground(self, diff_threshold=None, frames_avg=50,
                         smooth_std=3):
        """Simple motion tracking using an average of the whole video as the
        background and the absolut difference between each frame and the
        background as the foreground.
        """
        avg = self.get_average(frames_avg)
        self.track = []
        self.diffs = []
        for ind, layer in enumerate(self.layers):
            diff = abs(layer.load_image() - avg)
            diff = colors.rgb_to_hsv(diff)[..., 2]
            layer.image = None
            diff = gaussian_filter(diff, smooth_std)
            layer.diff = diff
            if diff_threshold is None:
                xs, ys = local_maxima(diff, disp=False, p=95)
                if len(xs) > 0:
                    self.track += [(xs[0], ys[0])]
                else:
                    self.track += [(np.nan, np.nan)]
            else:
                xs, ys = local_maxima(diff, disp=False,
                                      min_diff=diff_threshold)
                if len(xs) > 0:
                    self.track += [(xs, ys)]
                else:
                    self.track += [(np.nan, np.nan)]
            # self.diffs += [diff]
            # self.track += [(np.argmax(diff.mean(0)),
            #                 np.argmax(diff.mean(1)))]
            print_progress(ind, len(self.layers))

    def color_key(self, samples=5, display=True):
        """Grab unmoving 'background' of the stack by averaging over
        a sample of layers. The default is 50 samples.
        """
        if self.colors is None:
            self.select_color_range(samples=samples)
        if self.track is None:
            print("tracking color range")
            self.track = []
            progress = 0
            for l in self.layers:
                img = l.load_image()
                hsv = colors.rgb_to_hsv(img)
                low_hue = self.colors[:, 0].min()
                hi_hue = self.colors[:, 0].max()
                if low_hue < 0:
                    hues = np.logical_or(
                        hsv[:, :, 0] > 1 + low_hue,
                        hsv[:, :, 0] < hi_hue)
                else:
                    hues = np.logical_and(
                        hsv[:, :, 0] > low_hue,
                        hsv[:, :, 0] < hi_hue)
                sats = np.logical_and(
                    hsv[:, :, 1] > self.colors[:, 1].min(),
                    hsv[:, :, 1] < self.colors[:, 1].max())
                vals = np.logical_and(
                    hsv[:, :, 2] > self.colors[:, 2].min(),
                    hsv[:, :, 2] < self.colors[:, 2].max())
                mask = np.logical_and(hues, sats, vals)
                track = center_of_mass(mask)
                self.track += [(track[1], track[0])]
                # l.image = None
                progress += 1
                print_progress(progress, len(self.layers))
        if display:
            # plt.ion()
            first = True
            for l, (x, y) in zip(self.layers, self.track):
                # l.load_image()
                if first:
                    self.image_fig = plt.imshow(l.image)
                    dot = plt.plot(x, y, 'o')
                    plt.show()
                else:
                    self.image_fig.set_data(l.image)
                    dot[0].set_data(x, y)
                    plt.draw()
                    plt.pause(.001)
                l.image = None
                if first:
                    first = False


class Video2():
    """A wrapper for grabbing frames from a video using ffmpeg. This is
    inspired in large part by the skvideo io library.
    """
    def __init__(self, path, bw=False):
        self.path = os.path.realpath(path)
        self.probedata = ffprobe(self.path)
        self.num_frames = int(self.probedata['video']['@duration_ts'])
        fr = self.probedata['video']['@avg_frame_rate'].split("/")
        self.fps = float(fr[0])/float(fr[1])
        self.duration = float(self.num_frames)/self.fps
        self.width = int(self.probedata['video']['@width'])
        self.height = int(self.probedata['video']['@height'])
        if bw:
            self.depth = 1
        else:
            self.depth = 3
        self.framesize = self.width*self.height*self.depth
        self.pix_fmt = self.probedata['video']['@pix_fmt']

    def grab_frame(self, index, num=1):
        """Use ffmpeg to get data of a single frame using the frame's index.
        """
        # remember to set -vsync 0
        seconds = float(index)/self.fps
        cmd = ['ffmpeg',
               "-nostats",
               "-loglevel", "0",
               '-ss', str(seconds),
               '-i',  self.path,
               # "-vf", "select=gte(n\, {})".format(index),
               '-f', 'image2pipe',
               '-pix_fmt', 'rgb24',
               '-vcodec', 'rawvideo',
               '-vsync', '0',
               '-']
        self._proc = subprocess.Popen(cmd,
                                      stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE,
                                      # stderr=None)
                                      stderr=subprocess.PIPE)
        arr = np.frombuffer(self._proc.stdout.read(self.framesize),
                            dtype=np.uint8)
        return arr.reshape((self.height, self.width, self.depth))

    def get_average(self, samples=50):
        spacing = self.num_frames/samples
        inds = range(0, self.num_frames, spacing)
        avg = np.zeros([self.height, self.width, self.depth], dtype='int64')
        for ind in inds:
            avg += self.grab_frame(ind)
        avg = avg.astype(float)/float(len(inds))
        avg = avg.astype('uint8')
        return avg

    def back_subtraction(self, background_samples=50, smooth_param=3,
                     edges=False, output_samples=None):
        avg = self.get_average(samples=background_samples)
        if output_samples is not None:
            spacing = self.num_frames/output_samples
        else:
            spacing = 1
        inds = range(0, self.num_frames, spacing)
        track = np.empty((len(inds), self.height, self.width, self.depth),
                              dtype='uint8')
        for num, ind in enumerate(inds):
            frame = self.grab_frame(ind)
            diff = abs(frame.astype(float) - avg.astype(float))
            diff = ndimage.gaussian_filter(diff, smooth_param)
            if edges:
                sx = ndimage.filters.sobel(diff, axis=0, mode='constant')
                sy = ndimage.filters.sobel(diff, axis=1, mode='constant')
                diff = np.hypot(sx, sy).astype('uint8')
            track[num] = diff
            print_progress(num, len(inds))
        return track
