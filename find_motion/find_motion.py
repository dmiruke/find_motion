#!/usr/bin/env python3

# pylint: disable=line-too-long,logging-format-interpolation,no-member

"""
Motion detection with OpenCV

With much help from https://www.pyimagesearch.com/

Caches images for a few frames before and after it detects movement
"""

"""
# TODO: fix --ignore-drive, it's broken

# TODO: have args as provided by argparse take priority over those in the config (currently it is vv)

# TODO: think about r/g/b channel motion detection, instead of just grayscale, or some kind of colour-change detection instead of just tone change - measure rgb on a linear scale, detect change of 'high' amount

# TODO: scale box sizes by location in frame - gradient, or custom matrix

# TODO: look at more OpenCV functions, e.g.
    https://docs.opencv.org/3.2.0/d7/de9/group__video.html

# TODO: add other output streams - not just to files, to cloud, sFTP server or email
"""

import sys
import os

import signal
import time
from datetime import datetime
import math

from argparse import ArgumentParser, Namespace
from configparser import ConfigParser
from ast import literal_eval
import json
from jsonschema import validate
from time import strptime

import typing

from collections import deque

from functools import partial
from multiprocessing import Pool, Event
from pynput import keyboard

import logging
import progressbar

from mem_top import mem_top
from orderedset import OrderedSet

from numpy import array as np_array
from numpy import int32 as np_int32
from numpy import ndarray as np_ndarray

import cv2
import imutils


# pylint: disable=invalid-name
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
# pylint: enable=invalid-name

LINE_BUFFERED: int = 1

# Color constants
BLACK = (0, 0, 0)
RED = (0, 0, 255)
GREEN = (0, 255, 0)

MASK_SCHEMA = {
    "type": "array",
    "items": {
        "type": "array",
        "minItems": 2,
        "items": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": {
                "type": "integer"
            }
        }
    }
}


unpaused = Event()


def init_worker(event) -> None:
    """
    Supress signal handling in the worker processes so that they don't capture SIGINT (ctrl-c)

    Set an event so we can pause workers
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global unpaused
    unpaused = event


HELD: set = set()


def on_press(key: keyboard.Key) -> bool:
    """
    Listen for spacebar keypresses in controller process, set event in worker processes to pause them

    This listens to keypress on all programs, so we pick a strange and unusual key combination - shift-pause
    """
    log.debug(str(key) + ' down')
    log.debug(str(HELD))
    if key == keyboard.Key.esc:
        # Stop listener
        return False
    if key == keyboard.Key.pause and keyboard.Key.pause not in HELD and keyboard.Key.shift in HELD and keyboard.Key.alt_l in HELD:
        if unpaused.is_set():
            unpaused.clear()
            log.debug('Pausing')
        else:
            unpaused.set()
            log.debug('Resuming')
    HELD.add(key)
    return True


def on_release(key: keyboard.Key) -> bool:
    log.debug(str(key) + ' up')
    HELD.remove(key)
    if key == keyboard.Key.esc:
        # Stop listener
        return False
    return True


class VideoError(Exception):
    """
    An error when processing a video
    """
    def __init__(self, msg):
        super()


class VideoInfo(object):
    """
    Class to read in a video, and get metadata out
    """
    def __init__(self, filename: str=None, log_level=logging.INFO) -> None:
        self.filename: str = filename
        self.cap: cv2.VideoCapture = None
        self.amount_of_frames: int = 0
        self.frame_width: int = 0
        self.frame_height: int = 0

        self.log = logging.getLogger('find_motion.VideoInfo')
        self.log.setLevel(log_level)

        self.loaded = self._load_video()


    def __str__(self) -> str:
        return "File: {}; frames: {}; {}x{}px".format(self.filename, self.amount_of_frames, self.frame_width, self.frame_height)


    def _load_video(self) -> bool:
        """
        Open the input video file, get the video info
        """
        self.cap = cv2.VideoCapture(self.filename)
        try:
            self._get_video_info()
        except VideoError as e:
            self.log.error(str(e))
            return False
        return True


    def _get_video_info(self) -> None:
        """
        Set some metrics from the video
        """
        self.log.debug(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.amount_of_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.amount_of_frames == 0 or self.frame_width == 0 or self.frame_height == 0:
            broken = 'frames' if self.amount_of_frames == 0 else 'height/width'
            raise VideoError("Video info malformed - {} is 0: {}".format(broken, self.filename))
        return


class VideoFrame(object):
    """
    encapsulate frame stuff here, out of main video class
    """
    def __init__(self, frame, show=False) -> None:
        self.raw: np_ndarray = frame
        self.frame: np_ndarray = self.raw.copy()    # TODO: work out how to remove this if show is False and still have things work
        self.in_cache: bool = False
        self.contours = None
        self.frame_delta: np_ndarray = None
        self.thresh: np_ndarray = None
        self.blur: np_ndarray = None


    def diff(self, ref_frame) -> None:
        """
        Find the diff between this frame and the reference frame
        """
        self.frame_delta = cv2.absdiff(self.blur, cv2.convertScaleAbs(ref_frame))


    def threshold(self, thresh) -> None:
        """
        Find the threshold of the diff
        """
        self.thresh = cv2.threshold(self.frame_delta, thresh, maxval=255, type=cv2.THRESH_BINARY)[1]


    def find_contours(self) -> None:
        """
        Find edges of the shapes in the thresholded image
        """
        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        self.thresh = cv2.dilate(self.thresh, kernel=None, iterations=2)
        cnts = cv2.findContours(self.thresh.copy(), mode=cv2.RETR_EXTERNAL,
                                method=cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        self.contours = cnts


    def cleanup(self) -> None:
        """
        Actively destroy the frame
        """
        log.debug('Cleanup frame')
        for attr in ('frame', 'thresh', 'contours', 'frame_delta', 'blur'):
            if hasattr(self, attr):
                delattr(self, attr)
        if self.in_cache:
            log.debug('Frame in cache, not cleaning up up raw frame or deleting frame object')
            return
        if hasattr(self, 'raw'):
            del self.raw
        del self


class VideoMotion(object):
    """
    Class to read in a video, detect motion in it, and write out just the motion to a new file
    """
    # pylint: disable=too-many-instance-attributes,too-many-arguments
    def __init__(self, filename: typing.Union[str, int]=None, outdir: str='', fps: int=30,
                 box_size: int=100, min_box_scale: int=50, cache_time: float=2.0, min_time: float=0.5,
                 threshold: int=7, avg: float=0.1, blur_scale: int=20,
                 mask_areas: list=None, show: bool=False,
                 codec: str='MJPG', log_level: int=logging.INFO,
                 mem: bool=False, cleanup: bool=False, multiprocess: bool=False) -> None:
        self.filename = filename

        if self.filename is None:
            raise Exception('Filename required')

        self.log = logging.getLogger('find_motion.VideoMotion')
        self.log.setLevel(log_level)

        self.log.debug("Reading from {}".format(self.filename))

        self.multiprocess = multiprocess

        if not self.multiprocess:
            log.debug('Single process')

        self.outfile: cv2.VideoWriter = None    # type: ignore
        self.outfiles: int = 0
        self.outfile_name: str = ''
        self.outdir: str = os.path.normpath(outdir)

        self.fps: int = fps
        self.box_size: int = box_size
        self.min_box_scale: int = min_box_scale
        self.min_area: int = -1
        self.max_area: int = -1
        self.gaussian_scale: int = blur_scale
        self.cache_frames = int(cache_time * fps)
        self.min_movement_frames: int = int(min_time * fps)
        self.delta_thresh: int = threshold
        self.avg: float = avg
        self.mask_areas: typing.List[typing.Any] = mask_areas if mask_areas is not None else []
        self.show: bool = show

        self.log.debug('Caching {} frames, min motion {} frames'.format(self.cache_frames, self.min_movement_frames))

        self.codec: str = codec
        self.debug: bool = log_level == logging.DEBUG
        self.mem: bool = mem
        self.cleanup_flag: bool = cleanup

        # TODO: make path relative to this file
        self.cascade = cv2.CascadeClassifier("./find_motion/haarcascade_frontalface_default.xml")

        self.log.debug(self.codec)

        # initialised in _load_video
        self.amount_of_frames: int = -1
        self.frame_width: int = -1
        self.frame_height: int = -1
        self.scale: float = -1.0

        self.current_frame: VideoFrame
        self.ref_frame: VideoFrame
        self.frame_cache: typing.Deque[VideoFrame]

        self.wrote_frames: bool = False
        self.err_msg: str = ''

        self.movement: bool = False
        self.movement_decay: int = 0
        self.movement_counter: int = 0

        # Initialisation functions
        self._calc_min_area()
        self._make_gaussian()
        self.loaded = self._load_video()


    def _calc_min_area(self) -> None:
        """
        Set the minimum motion area based on the box size
        """
        self.min_area = int(math.pow(self.box_size / self.min_box_scale, 2))


    def _load_video(self) -> bool:
        """
        Open the input video file, set up the ref frame and frame cache, get the video info and set the scale
        """
        self.cap = cv2.VideoCapture(self.filename)
        self.ref_frame = None
        self.frame_cache = deque(maxlen=self.cache_frames)

        try:
            self._get_video_info()
        except VideoError as e:
            self.log.error(str(e))
            return False
        self.scale = self.box_size / self.frame_width
        self.max_area = int((self.frame_width * self.frame_height) / 2 * self.scale)
        return True


    def _get_video_info(self) -> None:
        """
        Set some metrics from the video
        """
        self.log.debug(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.amount_of_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.frame_width == 0 or self.frame_height == 0:
            broken = 'width' if self.frame_width == 0 else 'height'
            raise VideoError("Video info malformed - {} is 0: {}".format(broken, self.filename))
        if self.amount_of_frames == 0:
            log.warning('Video info malformed - frames reported as 0')
        elif self.amount_of_frames == -1:
            log.debug('Streaming - frames reported as -1')
        return


    def _make_outfile(self) -> None:
        """
        Create an output file based on the input filename and the output directory
        """
        self.outfiles += 1

        if self.outfiles > 1 and self.outfile is not None:
            self.outfile.release()

        outname = str(self.filename) + '_' + str(self.outfiles)

        if self.outdir == '':
            self.outfile_name = outname + '_motion.avi'
        else:
            self.outfile_name = os.path.join(self.outdir,
                                             os.path.basename(outname)) + '_motion.avi'

        if self.debug:
            self.log.debug("Writing to {}".format(self.outfile_name))

        try:
            self.outfile = cv2.VideoWriter(self.outfile_name,
                                        cv2.VideoWriter_fourcc(*self.codec),
                                        self.fps, (self.frame_width, self.frame_height))
        except Exception as e:
            self.log.error('Failed to create output file: {}'.format(e))
            raise e

        self.log.debug('Made output file')


    def _make_gaussian(self) -> None:
        """
        Make a gaussian for the blur using the box size as a guide
        """
        gaussian_size = int(self.box_size / self.gaussian_scale)
        gaussian_size = gaussian_size + 1 if gaussian_size % 2 == 0 else gaussian_size
        self.gaussian = (gaussian_size, gaussian_size)


    def blur_frame(self, frame=None) -> None:
        """
        Shrink, grayscale and blur the frame
        """
        frame = self.current_frame if frame is None else frame
        small = imutils.resize(frame.raw, width=self.box_size)
        frame.gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        frame.blur = cv2.GaussianBlur(frame.gray, self.gaussian, 0)


    def read(self) -> bool:
        """
        Read a frame from the capture member
        """
        (ret, frame) = self.cap.read()
        if not ret:
            return False

        self.current_frame = VideoFrame(frame)
        return True


    def output_frame(self, frame: VideoFrame=None) -> None:
        """
        Put a frame out to screen (if required) and file

        Initialise the output file if necessary
        """
        frame = self.current_frame if frame is None else frame

        if self.show:
            self.log.debug('Showing frame on output')
            cv2.imshow('frame', frame.frame)

        if not self.wrote_frames:
            self._make_outfile()
            self.wrote_frames = True

        try:
            self.outfile.write(frame.raw)
        except Exception as e:
            self.log.warning('Having to create output file due to exception: {}'.format(e))
            self._make_outfile()
            self.outfile.write(frame.raw)


    def output_raw_frame(self, frame: np_ndarray=None) -> None:
        """
        Output a raw frame, not a VideoFrame
        """
        if not self.wrote_frames:
            self._make_outfile()
            self.wrote_frames = True

        try:
            self.outfile.write(frame)
        except Exception as e:
            self.log.warning('Having to create output file due to exception: {}'.format(e))
            self._make_outfile()
            self.outfile.write(frame)


    def decide_output(self) -> None:
        """
        Decide if we are going to put out this frame
        """
        self.log.debug('Deciding output')

        if (self.movement_counter >= self.min_movement_frames) or (self.movement_decay > 0):
            self.log.debug('There is movement')
            # show cached frames
            if self.movement:
                self.movement_decay = self.cache_frames

                for frame in self.frame_cache:
                    if frame is not None:
                        self.log.debug('Outputting cached raw frame')
                        self.output_raw_frame(frame.raw)
                        if self.cleanup_flag:
                            frame.in_cache = False
                            frame.cleanup()
                            del frame

                self.frame_cache.clear()
            # draw the text and identify objects
            if self.show:
                self.draw_text()
                self.find_objects()

            self.log.debug('Outputting frame')

            self.output_frame()
        else:
            self.log.debug('No movement, putting in cache')
            if self.cleanup_flag:
                self.cleanup_cache()
            self.frame_cache.append(self.current_frame)
            self.current_frame.in_cache = True


    def cleanup_cache(self) -> None:
        cache_size = len(self.frame_cache)
        self.log.debug(str(cache_size))
        if cache_size == self.cache_frames:
            self.log.debug('Clearing first cache entry')
            delete_frame = self.frame_cache.popleft()
            if delete_frame is not None:
                delete_frame.in_cache = False
                delete_frame.cleanup()
                del delete_frame


    def is_open(self) -> bool:
        """
        Return if the capture member is open
        """
        return self.cap.isOpened()


    @staticmethod
    def scale_area(area: list, scale: float) -> list:
        """
        Scale the area by the scale factor
        """
        return [(int(a[0] * scale), int(a[1] * scale)) for a in area]


    def mask_off_areas(self, frame: VideoFrame=None):
        """
        Draw black polygons over the masked off areas
        """
        frame = self.current_frame if frame is None else frame
        for area in self.mask_areas:
            scaled_area = VideoMotion.scale_area(area, self.scale)
            dim = len(scaled_area)
            if dim == 2:
                cv2.rectangle(frame.blur,
                              *scaled_area,
                              BLACK, cv2.FILLED)
            else:
                pts = np_array(scaled_area, np_int32)
                cv2.fillConvexPoly(frame.blur,
                                   pts,
                                   BLACK)


    def find_diff(self, frame: VideoFrame=None) -> None:
        """
        Find the difference between this frame and the moving average

        Locate the contours around the thresholded difference

        Update the moving average
        """
        frame = self.current_frame if frame is None else frame
        if self.ref_frame is None:
            self.ref_frame = frame.blur.copy().astype("float")

        # compute the absolute difference between the current frame and ref frame
        frame.diff(self.ref_frame)
        frame.threshold(self.delta_thresh)

        # update reference frame using weighted average
        cv2.accumulateWeighted(frame.blur, self.ref_frame, self.avg)

        # find contours from the diff data
        frame.find_contours()


    def find_movement(self, frame: VideoFrame=None) -> None:
        """
        Locate contours that are big enough to count as movement
        """
        frame = self.current_frame if frame is None else frame

        self.movement = False
        self.movement_decay -= 1 if self.movement_decay > 0 else 0

        if frame.contours:
            # loop over the contours
            for contour in frame.contours:
                # if the contour is too small, ignore it
                if self.max_area < cv2.contourArea(contour) < self.min_area:
                    continue

                # compute the bounding box for the contour, draw it on the frame,
                # and update the text

                if self.show:
                    box = self.make_box(contour, frame)
                    self.draw_box(box, frame)

                self.movement_counter += 1
                self.movement = True

        if not self.movement:
            self.movement_counter = 0

        return


    # TODO: add multiple classes of object detection
    def find_objects(self, frame: VideoFrame=None) -> None:
        frame = self.current_frame if frame is None else frame

        objects = self.cascade.detectMultiScale(frame.gray, scaleFactor=1.1, minNeighbors=5)  # TODO: take scaleFactor and minNeighbours as parameters

        for rect in objects:

            area = VideoMotion.make_area(rect)

            if self.show:
                cv2.rectangle(frame.frame, *self.scale_area(area, 1 / self.scale), RED, 3)
            
            self.log.debug('Object found!')


    def draw_text(self, frame: VideoFrame=None) -> None:
        """
        Put the status text on the frame
        """
        frame = self.current_frame if frame is None else frame
        # draw the text
        cv2.putText(frame.frame,
                    "Status: {}".format('motion' if self.movement else 'quiet'),
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, RED, 2)
        return


    def make_box(self, contour, frame: VideoFrame=None) -> typing.Tuple[typing.Tuple[int, int], typing.Tuple[int, int]]:
        """
        Draw a green bounding box on the frame
        """
        frame = self.current_frame if frame is None else frame
        return VideoMotion.make_area(cv2.boundingRect(contour))


    @staticmethod
    def make_area(object_tuple: typing.Tuple[int, int, int, int]) -> typing.Tuple[typing.Tuple[int, int], typing.Tuple[int, int]]:
        # pylint: disable=invalid-name
        (x, y, w, h) = object_tuple
        area = ((x, y), (x + w, y + h))
        return area


    def draw_box(self, area, frame: VideoFrame=None) -> None:
        frame = self.current_frame if frame is None else frame
        cv2.rectangle(frame.frame, *self.scale_area(area, 1 / self.scale), GREEN, 2)


    @staticmethod
    def key_pressed(key: str) -> bool:
        """
        Say if we pressed the key we asked for
        """
        return cv2.waitKey(1) & 0xFF == ord(key)


    def cleanup(self) -> None:
        """
        Close the input file, output file, and get rid of OpenCV windows
        """
        if self.cap is not None:
            self.cap.release()

        if self.outfile is not None:
            self.outfile.release()

        if self.cleanup_flag:
            self.current_frame.in_cache = False
            self.current_frame.cleanup()
            del self.current_frame

            del self.ref_frame

            for frame in self.frame_cache:
                if frame is not None:
                    frame.in_cache = False
                    frame.cleanup()
                    del frame
            self.frame_cache.clear()
            del self.frame_cache

        return


    def find_motion(self) -> tuple:
        """
        Main loop. Find motion in frames.
        """
        while self.is_open():

            if self.multiprocess:
                self.log.debug('Waiting...')
                unpaused.wait()

            if not self.read():
                self.log.debug('Reading did not succeed')
                break

            self.blur_frame()
            self.mask_off_areas()
            self.find_diff()

            self.log.debug('Blurred frame, masked off, and diff made')

            # draw contours and set movement
            try:
                self.find_movement()
            except Exception as e:
                self.log.error('find_movement: {}'.format(e))

            self.log.debug('Searched for movement')

            if self.mem:
                self.log.info(mem_top())

            self.decide_output()

            self.log.debug('Decided output')

            if self.show:
                if self.frame_height > 0 and self.frame_width > 0:
                    try:
                        cf: VideoFrame = self.current_frame
                        if cf.thresh is not None:
                            self.log.debug('Showing threshold frame')
                            cv2.imshow('thresh', cf.thresh)
                        if cf.gray is not None:
                            self.log.debug('Showing gray frame')
                            cv2.imshow('gray', cf.gray)
                        if cf.blur is not None:
                            self.log.debug('Showing blur frame')
                            cv2.imshow('blur', cf.blur)
                        if cf.raw is not None:
                            self.log.debug('Showing raw frame')
                            cv2.imshow('raw', cf.raw)
                    except Exception as e:
                        self.log.error('Oops: {}'.format(e))
                else:
                    self.log.warning('Not showing frames, height or width is 0')

            self.log.debug('Decided to show frames or not')

            self.current_frame.cleanup()

            if VideoMotion.key_pressed('q'):
                self.wrote_frames = None
                self.err_msg = 'Closing video at user request'
                break

        self.log.debug('Cleaning up video')

        self.cleanup()
#        self.frame_cache.clear()

        return self.wrote_frames, self.err_msg


def find_files(directory: str) -> typing.List[str]:
    """
    Finds files in the directory, recursively, sorts them by last modified time
    """
    return [os.path.normpath(os.path.abspath(os.path.join(dirpath, f))) for dirpath, dnames, fnames in os.walk(directory) for f in fnames if f != 'progress.log'] if directory is not None else []


def verify_files(file_list: typing.List[str]) -> typing.List[str]:
    """
    Locates files with given file names, returns in a list of tuples with their last modified time
    """
    return [os.path.normpath(os.path.abspath(f)) for f in file_list if os.path.isfile(f)]


def sort_files_by_time(file_list: typing.List[str], priority_intervals: typing.List[typing.Tuple[time.struct_time, time.struct_time]]) -> OrderedSet:
    """
    Sort files by modified time.

    Take into account if a file falls in one of the priority periods, do it first.

    Method: find files in the intervals, starting with the highest priority one first.

    Go through the files, putting any matches at the start of the OrderedSet.

    Repeat until we get to the end of the intervals, then put any remaining files onto the set in order.
    """
    sorted_files: typing.List[typing.Tuple[str, float]] = [f for f in sorted([(f, os.path.getmtime(f)) for f in file_list], key=lambda f: f[1])]

    file_set: OrderedSet = OrderedSet()

    for time_interval in priority_intervals:
        for vid_file in sorted_files:
            if in_interval(vid_file, time_interval) and vid_file not in file_set:
                file_set.add(vid_file)

    for vid_file in sorted_files:
        if vid_file not in file_set:
            file_set.add(vid_file)

    if log.getEffectiveLevel() == logging.DEBUG:
        for f in file_set:
            log.debug(datetime.fromtimestamp(f[1]).strftime("%H:%M:%S"))

    return file_set


def in_interval(vid_file: typing.Tuple[str, float], time_interval: typing.Tuple[time.struct_time, time.struct_time]) -> bool:
    """
    Check if the mtime (epoch) of the file is in the time interval given
    """
    file_time = ClockTime(time.gmtime(vid_file[1]))
    interval = (ClockTime(time_interval[0]), ClockTime(time_interval[1]))
    return interval[0] <= file_time and file_time < interval[1]


class ClockTime(object):
    def __init__(self, struct_time: time.struct_time) -> None:
        self.hour: int = struct_time.tm_hour
        self.min: int = struct_time.tm_min
        self.sec: int = struct_time.tm_sec

    def __str__(self) -> str:
        return "{:02d}:{:02d}:{:02d}".format(self.hour, self.min, self.sec)

    # typhints left out on 'other' since 'ClockTime' fails and using 'object' provides little benefit
    def __lt__(self, other) -> bool:
        return self.hour < other.hour \
            or self.hour == other.hour and self.min < other.min \
            or self.hour == other.hour and self.min == other.min and self.sec < other.sec

    def __le__(self, other) -> bool:
        return self.__eq__(other) or \
            self.__lt__(other)

    def __eq__(self, other) -> bool:
        return self.hour == other.hour and \
            self.min == other.min and \
            self.sec == other.sec

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __gt__(self, other) -> bool:
        return self.hour > other.hour \
            or self.hour == other.hour and self.min > other.min \
            or self.hour == other.hour and self.min == other.min and self.sec > other.sec

    def __ge__(self, other) -> bool:
        return self.__eq__(other) or \
            self.__gt__(other)


def run_vid(filename: typing.Union[str, int], **kwargs) -> tuple:
    """
    Video creation and runner function to pass to multiprocessing pool
    """
    try:
        vid = VideoMotion(filename=filename, **kwargs)
        if vid.loaded:
            log.debug('Video loaded')
            wrote_frames, err_msg = vid.find_motion()
        else:
            wrote_frames = None
            err_msg = 'Video did not load successfully'
    except Exception as e:
        err_msg = 'Error processing video {}: {}'.format(filename, e)
        wrote_frames = None
    return (wrote_frames, filename, err_msg)


class DummyProgressBar(object):
    """
    A pretend progress bar
    """
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __exit__(self, *args, **kwargs) -> None:
        pass

    def __enter__(self, *args, **kwargs) -> object:
        return self

    def update(self, *args, **kwargs) -> None:
        pass


def get_progress(log_file: str) -> set:
    """
    Load the progress log file, get the list of files
    """
    try:
        with open(log_file, 'r') as progress_log:
            done_files = {f.strip() for f in progress_log.readlines()}
            return done_files
    except FileNotFoundError:
        return set()


def run_pool(job: typing.Callable[..., typing.Any], processes: int, files: typing.Iterable[str]=None, pbar: typing.Union[progressbar.ProgressBar, DummyProgressBar]=DummyProgressBar(), progress_log: typing.TextIO=None) -> None:
    """
    Create and run a pool of workers

    Allows pausing by pressing the spacebar
    """
    if not files:
        raise ValueError('More than 0 files needed')

    num_files: int = len(list(files))
    done: int = 0
    files_written: typing.Set = set()
    results: list = []

    global unpaused

    try:
        pool = Pool(processes=processes, initializer=partial(init_worker, unpaused))
        unpaused.set()

        for filename in files:
            results.append(pool.apply_async(job, (filename,)))

        num_err = 0
        num_wrote = 0

        while True:
            # Collect keyboard events until released
            with keyboard.Listener(
                on_press=on_press,
                on_release=on_release
            ) as listener:
                if unpaused.is_set():
                    files_done = {res.get() for res in results if res.ready()}
                    num_done = len(files_done)

                    if num_done > done:
                        done = num_done

                    if done > 0:
                        new = files_done.difference(files_written)
                        files_written.update(new)

                        for wrote_frames, filename, err_msg in new:
                            log.debug('Done {}{}'.format(filename, '' if wrote_frames else ' (no output)'))

                            if err_msg:
                                log.error('Error processing {}: {}'.format(filename, err_msg))
                                num_err += 1
                            else:
                                if progress_log is not None:
                                    print(filename, file=progress_log)

                            if wrote_frames:
                                num_wrote += 1

                    pbar.update(done)

                    if num_done == num_files:
                        log.debug("All processes completed. {} errors, wrote {} files".format(num_err, num_wrote))
                        break
                time.sleep(1)
    except KeyboardInterrupt:
        log.warning('Ending processing at user request')

    unpaused.clear()
    listener.stop()
    pool.terminate()


def run_map(job: typing.Callable, files: typing.Iterable[str], pbar, progress_log: typing.TextIO) -> None:
    if not files:
        raise ValueError('More than 0 files needed')

    log.debug('Processing each file one-by-one')

    files_processed: typing.Iterable[str] = map(job, files)
    done: int = 0

    try:
        # Collect keyboard events until released
        with keyboard.Listener(
            on_press=on_press,
            on_release=on_release
        ) as listener:
            for status, filename, err in files_processed:
                if pbar is not None:
                    done += 1
                    pbar.update(done)

                log.debug('Done {}{}'.format(filename, '' if status else ' (no output)'))

                if status is not None:
                    print(filename, file=progress_log)
    except KeyboardInterrupt:
        log.warning('Ending processing at user request')
    listener.stop()


def run_stream(job: typing.Callable, processes: int, cameras: typing.Iterable[int], progress_log: typing.TextIO) -> None:
    if not cameras:
        raise ValueError('More than 0 cameras needed')

    log.debug(cameras)

    num_cameras: int = len(list(cameras))
    done: int = 0
    files_written: typing.Set = set()
    results: list = []

    if num_cameras == 1 and processes == 1:
        log.debug('Single process, single camera stream')
        job(cameras[0])
    else:
        try:
            pool = Pool(processes=processes, initializer=partial(init_worker, unpaused))

            for camera in cameras:
                results.append(pool.apply_async(job, (camera,)))

            log.debug('Running jobs')

            while True:
                files_done = {res.get() for res in results if res.ready()}
                num_done = len(files_done)

                if num_done > done:
                    done = num_done
                
                if done > 0:
                    new = files_done.difference(files_written)
                    files_written.update(new)
                
                    for status, stream, err_msg in new:
                        log.debug('Done {}{}'.format(stream, '' if status else ' (no output)'))
                
                        if err_msg:
                            log.error('Ended processing camera {}: {}'.format(stream, err_msg))
                
                        print('Finished streaming from camera {}'.format(stream), file=progress_log)
                
                if num_done == num_cameras:
                    log.debug("All processes completed")
                    break
                
                time.sleep(1)
        except KeyboardInterrupt:
            log.warning('Ending processing at user request')

        pool.terminate()


def test_files(files):
    for f in files:
        log.debug("{}: {}".format(f[0], datetime.fromtimestamp(f[1]).isoformat()))
        try:
            log.debug(str(VideoInfo(f[0], log_level=logging.DEBUG)))
        except Exception as e:
            log.error("File {}: {}".format(f[0], e))


def test_stream(cameras) -> None:
    for camera in cameras:
        try:
            log.debug("Camera stream: " + str(VideoInfo(camera, log_level=logging.DEBUG)))
        except Exception as e:
            log.error("Failed to open camera {}: {}".format(camera, e))


def main():
    """
    Main app entry point
    """
    parser: ArgumentParser = ArgumentParser()
    get_args(parser)
    args: Namespace = parser.parse_args()

    logging.basicConfig()
    if args.debug:
        log.setLevel(logging.DEBUG)

    run(args, parser.print_help)


def make_pbar_widgets(num_files: int) -> list:
    """
    Create progressbar widgets
    """
    return [
        progressbar.Counter(), '/', str(num_files), ' ',
        progressbar.Percentage(), ' ',
        progressbar.Bar(), ' ',
        progressbar.Timer(), ' ',
        progressbar.ETA(),
    ]


def make_progressbar(progress: bool=False, num_files: int=0) -> progressbar.ProgressBar:
    """
    Create progressbar
    """
    return progressbar.ProgressBar(max_value=num_files,
                                   redirect_stdout=True,
                                   redirect_stderr=True,
                                   widgets=make_pbar_widgets(num_files)
                                   ) if progress else DummyProgressBar()


def read_masks(masks_file: str) -> list:
    try:
        with open(masks_file, 'r') as mf:
            masks = json.load(mf)
            validate(masks, MASK_SCHEMA)
            out_masks = []

            for mask in masks:
                log.debug('Mask area: {}'.format(mask))
                out_masks.append(tuple([tuple(coord) for coord in mask]))

            return out_masks

    except Exception as e:
        log.error('Masks file not read ({}): {}'.format(masks_file, e))
        return []


def set_log_file(input_dir: str=None, output_dir: str=None) -> str:
    return os.path.normpath(os.path.join(output_dir if output_dir is not None else input_dir if input_dir is not None else '.', 'progress.log'))


def run(args: Namespace, print_help: typing.Callable=lambda x: None) -> None:
    """
    Secondary entry point to allow running from a different app using an argparse Namespace
    """
    if args.config:
        log.debug('Processing config: {}'.format(args.config))
        process_config(args.config, args)

    if args.debug or args.test:
        log.setLevel(logging.DEBUG)

    if args.progress:
        progressbar.streams.wrap_stderr()

    if not args.files and not args.input_dir and not args.cameras:
        # no input: help message, exit
        print_help()
        sys.exit(2)

    if args.output_dir and not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
        log.warning('Created output directory {}'.format(args.output_dir))

    masks: list = args.masks if args.masks else []

    if args.masks_file:
        masks.extend(read_masks(args.masks_file))

    log.debug(str(masks))

    log_file: str = set_log_file(args.input_dir, args.output_dir)

    job = partial(run_vid,
                  outdir=args.output_dir, mask_areas=masks,
                  show=args.show, codec=args.codec,
                  log_level=logging.DEBUG if args.debug else logging.INFO,
                  mem=args.mem, cleanup=args.cleanup,
                  blur_scale=args.blur_scale, box_size=args.box_size, min_box_scale=args.min_box_scale,
                  threshold=args.threshold, avg=args.avg,
                  fps=args.fps, min_time=args.mintime, cache_time=args.cachetime, multiprocess=args.processes > 1)

    try:
        if args.cameras:
            if args.test:
                test_stream(args.cameras)
                sys.exit(0)

            # processing camera streams
            with open(log_file, 'a+', LINE_BUFFERED) as progress_log:
                run_stream(job, args.processes, args.cameras, progress_log)
        else:
            # processing input files

            # sort out time ordering priority
            time_order = process_times(args.time_order)
            log.debug(str(time_order))

            # find files on disk
            in_files: typing.List[str] = verify_files(args.files)
            in_files.extend(find_files(args.input_dir))

            # sort them
            files: OrderedSet = sort_files_by_time(in_files, time_order)

            found_files_num = len(files)

            log.debug("{} files found".format(str(found_files_num)))

            if not args.ignore_progress:
                files = process_progress(files, log_file, args.ignore_drive)
            else:
                log.debug('Ignoring previous progress, processing all found files')

            num_files: int = len(files)

            log.debug('Processing {} files'.format(num_files))
            log.debug(str(files))

            if args.test:
                test_files(files)
                sys.exit(0)

            do_files: typing.List[str] = [f[0] for f in files]

            with make_progressbar(args.progress, num_files) as pbar:
                pbar.update(0)
                with open(log_file, 'a+', LINE_BUFFERED) as progress_log:
                    if args.processes > 1:
                        run_pool(job, args.processes, do_files, pbar, progress_log)
                    else:
                        run_map(job, do_files, pbar, progress_log)
    except ValueError as e:
        log.error(str(e))
        sys.exit(1)


# TODO: fix ignore_drive
def process_progress(files: OrderedSet, log_file: str, ignore_drive: bool=False):
    found_files_num = len(files)
    done_files = get_progress(log_file)
    log.debug("{} done files".format(str(len(done_files))))
    if not ignore_drive:
        files = OrderedSet([f for f in files if f[0] not in done_files])
    else:
        files = OrderedSet([f for f in files if not map(lambda x: os.path.splitdrive(f[0])[1] == os.path.splitdrive(x), done_files)])
    log.debug("{} files removed".format(str(found_files_num - len(files))))
    return files


def process_times(time_order: typing.List[str]) -> typing.List[typing.Tuple[time.struct_time, time.struct_time]]:
    times: typing.List[typing.Tuple[time.struct_time, time.struct_time]] = []

    if time_order is None:
        return times

    for time_slot in time_order:
        try:
            res: typing.List[time.struct_time] = list(map(lambda t: strptime(t, '%H:%M'), time_slot.split('-')))
            interval: typing.Tuple[time.struct_time, time.struct_time] = (res[0], res[1])
        except ValueError as e:
            log.error('Time interval {} misparsed: {}'.format(time_slot, e))
            continue
        times.append(interval)
    return times


def process_config(config_file: str, args: Namespace) -> Namespace:
    """
    Read an INI style config

    TODO: apply argparse validation to the config values

    TODO: replace any - with an _
    """
    config: ConfigParser = ConfigParser()
    config.read(config_file)
    for setting, value in config['settings'].items():
        use_value: typing.Any = value
        if setting in ('processes', 'blur_scale', 'min_box_scale', 'threshold', 'fps', 'box_size'):
            use_value = int(value)
        if setting in ('mintime', 'cachetime', 'avg'):
            use_value = float(value)
        if setting in ('mem', 'progress', 'debug', 'show', 'ignore_progress', 'ignore_drive'):
            if value == 'True':
                use_value = True
            elif value == 'False':
                use_value = False
            else:
                raise ValueError('{} must be True or False'.format(setting))
        if setting in ('masks', 'cameras', 'time_order'):
            use_value = literal_eval(value)
            # TODO: validate that this is a list of tuples of int (masks) or a list of ints (cameras) or a list of strings (time_order)
        args.__setattr__(setting, use_value)
    log.debug(str(vars(args)))
    return args


def get_args(parser: ArgumentParser) -> None:
    """
    Set how to process command line arguments
    """
    parser.add_argument('files', nargs='*', help='Video files to find motion in')

    parser.add_argument('--config', '-c', help='Config in INI format')

    parser.add_argument('--cameras', nargs='*', type=int, help='0-indexed number of camera to stream from')
    parser.add_argument('--input-dir', '-i', help='Input directory to process')
    parser.add_argument('--output-dir', '-o', default='', help='Output directory for processed files')
    parser.add_argument('--ignore-progress', '-I', action='store_true', default=False, help='Ignore progress log')
    parser.add_argument('--ignore-drive', '-D', action='store_true', default=False, help='Ignore drive letter in progress log')

    parser.add_argument('--codec', '-k', default='MP42', help='Codec to write files with')
    parser.add_argument('--fps', '-f', type=int, default=30, help='Frames per second of input files')

    parser.add_argument('--time_order', '-to', nargs='*', help='Time ranges in priority order for processing. Express as "HH:MM-HH:MM"')

    parser.add_argument('--masks', '-m', nargs='*', type=literal_eval, help='Areas to mask off in video')
    parser.add_argument('--masks_file', help='File holding mask coordinates (JSON)')

    parser.add_argument('--blur-scale', '-b', type=int, default=20, help='Scale of gaussian blur size compared to video width (used as 1/blur_scale)')
    parser.add_argument('--box-size', '-B', type=int, default=100, help='Pixel size to scale the video to for processing')
    parser.add_argument('--min-box-scale', '-mbs', type=int, default=50, help='Scale of minimum motion compared to video width (used as 1/min_box_scale')
    parser.add_argument('--threshold', '-t', type=int, default=12, help='Threshold for change in grayscale')
    parser.add_argument('--mintime', '-M', type=float, default=0.5, help='Minimum time for motion, in seconds')
    parser.add_argument('--cachetime', '-C', type=float, default=1.0, help='How long to cache, in seconds')
    parser.add_argument('--avg', '-a', type=float, default=0.1, help='How much to weight the most recent frame in the running average')

    parser.add_argument('--processes', '-J', default=1, type=int, help='Number of processors to use')

    parser.add_argument('--progress', '-p', action='store_true', help='Show progress bar')
    parser.add_argument('--show', '-s', action='store_true', default=False, help='Show video processing')

    parser.add_argument('--cleanup', '-cu', action='store_true', help='Cleanup used frames (do not wait for garbage collection)')
    parser.add_argument('--mem', '-u', action='store_true', help='Run memory usage')
    parser.add_argument('--debug', '-d', action='store_true', help='Debug')
    parser.add_argument('--test', '-T', action='store_true', help='Test which files or camera streams would be processed')


if __name__ == '__main__':
    main()
