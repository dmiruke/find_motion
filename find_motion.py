#!/usr/bin/env python3

# pylint: disable=line-too-long,logging-format-interpolation,no-member

"""
Motion detection with OpenCV

With much help from...
https://www.pyimagesearch.com/

Caches images for a few frames before and after we detect movement

TODO: deal with 2GB max file output size :-( in OpenCV
        catch Exception and open another file with a 1-up number
        mostly dealt with using compression, but should still handle this

DONE: spec an output folder, and an input folder

DONE: compress output somehow!

DONE: remove any 0 frame ouput files - better, only create output file once we have our first output

DONE: minimum length of motion before we think it is motion

DONE: multiprocessing with a Pool

DONE: deal well with KeyboardInterrupt when multiprocess.
    Wherever we are, fail in a way that kills the pool as a whole
    tried using a metaclass to decorate each method of the class, but that's not working
    managed to block SIGINT in workers and use apply_async and a loop to check if workers
    were complete, which allows neat ctrl-c behaviour at the expense of ugly code.

DONE: keep track of which files have processed in a dot file in the directory, allow resuming, skipping those
    already doing log.debug(), start to write to a file, and then read that file on resume

DONE: progress bar with ProgressBar2 module - one for the whole run

DONE: sort input files by date, process in time order. Change sets to orderedset.

TODO: nicer way to specify filtered areas, using literal_eval - XXX

TODO: return flag to say if the file completed without error, or was user interrupted
    (q or ctrl-c), or had an exception of some kind

TODO: logging in the workers - they should pass messages to the master on completion
    - override log and write output to master in a list

DONE: imap_async, so that master gets results as they arrive, not once the whole job has completed,
    so that progress.log gets written to

TODO: allow pausing: https://stackoverflow.com/questions/23449792/how-to-pause-multiprocessing-pool-from-execution

DONE: remove 'frame' processing - drawing box, adding text, etc. if we are not showing it - waste of time and memory. Just del self.frame etc. done with it.

DONE: fix motion detection so we stop writing frames after movement goes away (seems broken at the moment)
    not sure what was wrong, seems to be working OK now

DONE: fix OutOfMemory problems? Can I catch that error and return that it happened, at least?
    I now aggressively del objects once they are done with, especially in the frame_cache - seems to have solved the memory leak
DONE: now leaking VideoFrame objects somewhere - fix!!!
    I'd initialised frame_cache to a list *after* it was initialised in _load_video() to a deque

DONE: improve progress output so that filenames are being written as they are processed - keep track of which have been done in a set, and only write out if done

TODO: add config using ConfigParser

TODO: think about r/g/b channel motion detection, instead of just grayscale, or some kind of colour-change detection instead of just tone change - measure rgb on a linear scale, detect change of 'high' amount

TODO: make frame_cache its own class so we can do cleanup etc. more neatly

DONE: why it is so slooow now? Seems to have got slower than first version. Why? Is it the constant object deletion to avoid the memory leak?
    it was running mem_top even when not in debug. Put boolean check round it.

DONE: why is it not processing videos correctly now? It seems to just skip to the end.
    Fiddled until it wasn't broken

DONE: progress bar updates eta very strangely, seems to be sensitive to immediate rate - change to overall rate, not immediate
    Tried using a different ETA widget. Let's see how this one behaves.

DONE: max box size for movement, so that sun changes that create "movement" over the whole screen don't get registered as motion

TODO: scale box sizes by location in frame - gradient, or custom matrix

DONE: increase blur dimension - allow that as an arg

DONE: add option to have 3 coordinate areas to mask (with a triangle)
    Can have arbitrary convex polygons

TODO: look at more OpenCV functions, e.g.
    https://docs.opencv.org/3.2.0/d7/df6/classcv_1_1BackgroundSubtractor.html
    https://docs.opencv.org/3.2.0/dd/d73/classcv_1_1bioinspired_1_1RetinaFastToneMapping.html
    https://docs.opencv.org/3.2.0/d9/d7a/classcv_1_1xphoto_1_1WhiteBalancer.html
"""

import os

import signal
import time
import math

from argparse import ArgumentParser
from ast import literal_eval

from collections import deque

from functools import partial
from multiprocessing import Pool

import logging
import progressbar

from mem_top import mem_top
from orderedset import OrderedSet
import numpy as np

import cv2
import imutils


# pylint: disable=invalid-name
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
# pylint: enable=invalid-name

LINE_BUFFERED = 1

BLACK = (0, 0, 0)
RED = (0, 0, 255)
GREEN = (0, 255, 0)


def init_worker():
    """
    Supress signal handling in the worker processes so that they don't capture SIGINT (ctrl-c)
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class VideoMotion(object):
    """
    Class to read in a video, detect motion in it, and write out just the motion to a new file
    """
    # pylint: disable=too-many-instance-attributes,too-many-arguments
    def __init__(self, filename=None, outdir=None, fps=30,
                 box_size=100, cache_time=2.0, min_time=0.5,
                 threshold=7, avg=0.1, gaussian_scale=20,
                 mask_areas=None, show=False,
                 codec='MJPG', log_level=logging.INFO, mem=False):
        self.filename = filename

        if self.filename is None:
            raise Exception('Filename required')

        log.debug("Reading from {}".format(self.filename))

        self.outfile = None
        self.outfile_name = None
        self.outdir = outdir

        self.fps = fps
        self.box_size = box_size
        self.min_area = None
        self.max_area = None
        self.gaussian_scale = gaussian_scale
        self.cache_frames = int(cache_time * fps)
        self.min_movement_frames = int(min_time * fps)
        self.delta_thresh = threshold
        self.avg = avg
        self.mask_areas = mask_areas if mask_areas is not None else []
        self.show = show

        log.debug('Caching {} frames, min motion {} frames'.format(self.cache_frames, self.min_movement_frames))

        self.codec = codec
        self.debug = log_level is logging.DEBUG
        self.mem = mem

        log.debug(self.codec)

        # initialised in _load_video
        self.amount_of_frames = None
        self.frame_width = None
        self.frame_height = None
        self.scale = None

        self.current_frame = None
        self.ref_frame = None
        self.frame_cache = None
        self.wrote_frames = False

        self._calc_min_area()
        self._make_gaussian()
        self._load_video()

        self.movement = False
        self.movement_decay = 0
        self.movement_counter = 0


    def _calc_min_area(self):
        """
        Set the minimum motion area based on the box size
        """
        self.min_area = int(math.pow(self.box_size/50, 2))


    def _load_video(self):
        """
        Open the input video file, set up the ref frame and frame cache, get the video info and set the scale
        """
        self.cap = cv2.VideoCapture(self.filename)
        self.ref_frame = None
        self.frame_cache = deque(maxlen=self.cache_frames)

        self._get_video_info()
        self.scale = self.box_size / self.frame_width
        self.max_area = int((self.frame_width * self.frame_height)/2) * self.scale


    def _make_outfile(self):
        """
        Create an output file based on the input filename and the output directory
        """
        if self.outdir is None:
            self.outfile_name = self.filename + '_motion.avi'
        else:
            self.outfile_name = os.path.join(self.outdir, os.path.basename(self.filename)) \
                                + '_motion.avi'

        if self.debug:
            log.debug("Writing to {}".format(self.outfile_name))

        self.outfile = cv2.VideoWriter(self.outfile_name,
                                       cv2.VideoWriter_fourcc(*self.codec),
                                       self.fps, (self.frame_width, self.frame_height))


    def _make_gaussian(self):
        """
        Make a gaussian for the blur using the box size as a guide
        """
        gaussian_size = int(self.box_size/self.gaussian_scale)
        gaussian_size = gaussian_size + 1 if gaussian_size % 2 == 0 else gaussian_size
        self.gaussian = (gaussian_size, gaussian_size)


    def blur_frame(self, frame=None):
        """
        Shrink, grayscale and blur the frame
        """
        frame = self.current_frame if frame is None else frame
        small = imutils.resize(frame.raw, width=self.box_size)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        frame.blur = cv2.GaussianBlur(gray, self.gaussian, 0)
        del small
        del gray


    def read(self):
        """
        Read a frame from the capture member
        """
        (ret, frame) = self.cap.read()
        if not ret:
            return False

        self.current_frame = VideoFrame(frame)
        return True


    def output_frame(self, frame=None):
        """
        Put a frame out to screen (if required) and file

        Initialise the output file if necessary
        """
        frame = self.current_frame if frame is None else frame
        self.wrote_frames = True
        if self.show:
            cv2.imshow('frame', frame.frame) # imutils.resize(frame.frame, width=self.frame_width))
        if self.outfile is None:
            self._make_outfile()
        self.outfile.write(frame.raw)
        #frame.cleanup


    def output_raw_frame(self, frame=None):
        """
        Output a raw frame, not a VideoFrame
        """
        self.wrote_frames = True
        if self.outfile is None:
            self._make_outfile()
        self.outfile.write(frame)


    def decide_output(self):
        """
        Decide if we are going to put out this frame
        """
        log.debug('Deciding output')

        if (self.movement_counter >= self.min_movement_frames) or (self.movement_decay > 0):
            log.debug('There is movement')
            # show cached frames
            if self.movement:
                self.movement_decay = self.cache_frames

                for frame in self.frame_cache:
                    if frame is not None:
                        self.output_raw_frame(frame.raw)
                        frame.in_cache = False
                        frame.cleanup()
                        del frame

                self.frame_cache.clear()
            # draw the text
            if self.show:
                self.draw_text()

            self.output_frame()
        else:
            log.debug('No movement, putting in cache')
            cache_size = len(self.frame_cache)
            log.debug(cache_size)
            if cache_size == self.cache_frames:
                log.debug('Clearing first cache entry')
                delete_frame = self.frame_cache.popleft()
                if delete_frame is not None:
                    delete_frame.in_cache = False
                    delete_frame.cleanup()
                    del delete_frame
            log.debug('Putting frame onto cache')
            self.frame_cache.append(self.current_frame)
            self.current_frame.in_cache = True


    def is_open(self):
        """
        Return if the capture member is open
        """
        return self.cap.isOpened()


    @staticmethod
    def scale_area(area, scale):
        """
        Scale the area by the scale factor
        """
        return [(int(a[0] * scale), int(a[1] * scale)) for a in area]


    def mask_off_areas(self, frame=None):
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
                pts = np.array(scaled_area, np.int32)
                cv2.fillConvexPoly(frame.blur,
                              pts,
                              BLACK)

    def find_diff(self, frame=None):
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


    def find_movement(self, frame=None):
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

        del frame.contours

        if not self.movement:
            self.movement_counter = 0

        return


    def draw_text(self, frame=None):
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


    def make_box(self, contour, frame=None):
        """
        Draw a green bounding box on the frame
        """
        frame = self.current_frame if frame is None else frame
        # pylint: disable=invalid-name
        (x, y, w, h) = cv2.boundingRect(contour)
        area = ((x, y), (x + w, y + h))
        return area


    def draw_box(self, area, frame=None):
        frame = self.current_frame if frame is None else frame
        cv2.rectangle(frame.frame, *self.scale_area(area, 1/self.scale), GREEN, 2)


    @staticmethod
    def key_pressed(key):
        """
        Say if we pressed the key we asked for
        """
        return cv2.waitKey(1) & 0xFF == ord(key)


    def _get_video_info(self):
        """
        Set some metrics from the video
        """
        self.amount_of_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.amount_of_frames == 0 or self.frame_width == 0 or self.frame_height == 0:
            broken = 'frames' if self.amount_of_frames == 0 else 'height/width'
            raise Exception("Video info malformed - {} is 0: {}".format(broken, self.filename))
        return


    def cleanup(self):
        """
        Close the input file, output file, and get rid of OpenCV windows
        """
        if self.cap is not None:
            self.cap.release()

        if self.outfile is not None:
            self.outfile.release()

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

        if self.show:
            cv2.destroyAllWindows()
        return


    def find_motion(self):
        """
        Main loop. Find motion in frames.
        """
        while self.is_open():
            if not self.read():
                break

            #a frame')

            self.blur_frame()
            self.mask_off_areas()
            self.find_diff()

            #log.debug('Found diff')

            # draw contours and set movement
            self.find_movement()

            #log.debug('Found movement')

            if self.mem:
                log.info(mem_top())

            self.decide_output()

            #log.debug('Decided output')

            if self.show:
                cv2.imshow('thresh', self.current_frame.thresh)
                #cv2.imshow('delta', self.current_frame.frame_delta)
                cv2.imshow('blur', self.current_frame.blur)
                cv2.imshow('raw', self.current_frame.raw)

            self.current_frame.cleanup()

            #log.debug('Cleaned up frame')

            if VideoMotion.key_pressed('q'):
                log.debug('Closing video at user request')
                # TODO: set flag saying so
                break

        log.debug('Cleaning up video')

        self.cleanup()

        return self.wrote_frames


class VideoFrame(object):
    """
    encapsulate frame stuff here, out of main video class
    """
    #__metaclass__ = ErrorCatcher

    def __init__(self, frame):
        self.frame = frame
        self.raw = self.frame.copy()
        self.in_cache = False
        self.contours = None
        self.frame_delta = None
        self.thresh = None
        self.blur = None


    def diff(self, ref_frame):
        """
        Find the diff between this frame and the reference frame
        """
        self.frame_delta = cv2.absdiff(self.blur, cv2.convertScaleAbs(ref_frame))


    def threshold(self, thresh):
        """
        Find the threshold of the diff
        """
        self.thresh = cv2.threshold(self.frame_delta, thresh, 255, cv2.THRESH_BINARY)[1]
#        del self.frame_delta


    def find_contours(self):
        """
        Find edges of the shapes in the thresholded image
        """
        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        self.thresh = cv2.dilate(self.thresh, None, iterations=2)
        cnts = cv2.findContours(self.thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        self.contours = cnts


    def cleanup(self):
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


def find_files(directory):
    """
    Finds files in the directory, recursively, sorts them by last modified time
    """
    files = [os.path.join(dirpath, f) for dirpath, dnames, fnames in os.walk(directory) for f in fnames] if directory is not None else []
    return OrderedSet([f[0] for f in sorted([(f, os.path.getmtime(f)) for f in files], key=lambda f: f[1])])


# pylint: disable=too-many-arguments
def run_vid(filename, outdir=None, mask_areas=None, show=None, codec=None, log_level=None, mem=None, blur_scale=None, threshold=None, fps=None, min_time=None, cache_time=None, avg=None):
    """
    Video creation and runner function to pass to multiprocessing pool
    """
    try:
        vid = VideoMotion(filename=filename, outdir=outdir, mask_areas=mask_areas,
                          gaussian_scale=blur_scale, show=show, codec=codec,
                          log_level=log_level, mem=mem, threshold=threshold, avg=avg,
                          fps=fps, min_time=min_time, cache_time=cache_time)
        err = vid.find_motion()
    except Exception as e:
        raise e
        #log.error(e)
        #err = None
    return (err, filename)
# pylint: enable=too-many-arguments


class DummyProgressBar(object):
    # pylint: disable=too-few-public-methods
    """
    A pretend progress bar
    """
    def __init__(self):
        pass

    def __exit__(self, *args):
        pass

    def __enter__(self, *args):
        return self

    def update(self, *args, **kwargs):
        pass


def get_progress(log_file):
    """
    Load the progress log file, get the list of files
    """
    try:
        with open(log_file, 'r') as progress_log:
            done_files = {f.strip() for f in progress_log.readlines()}
            return done_files
    except FileNotFoundError:
        return []


def run_pool(job=None, processes=2, files=None, pbar=None, progress_log=None):
    """
    Create and run a pool of workers
    """
    num_files = len(files)
    done = 0
    files_written = set()
    results = []

    try:
        pool = Pool(processes=processes, initializer=init_worker)
        for filename in files:
            results.append(pool.apply_async(job, (filename,)))
        while True:
            files_done = {res.get() for res in results if res.ready()}
            num_done = len(files_done)
            if num_done > done:
                done = num_done
            if done > 0:
                new = files_done.difference(files_written)
                files_written.update(new)
                for status, filename in new:
                    log.debug('Done {}{}'.format(filename, '' if status else ' (no output)'))
                    if status is not None and progress_log is not None:
                        print(filename, file=progress_log)
            pbar.update(done)
            if num_done == num_files:
                log.debug("All processes completed")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        log.warning('Ending processing at user request')

    pool.terminate()


def run_map(job, files, pbar, progress_log):
    files_processed = map(job, files)

    done = 0

    try:
        for status, filename in files_processed:
            if pbar is not None:
                done += 1
                pbar.update(done)
            log.debug('Done {}{}'.format(filename, '' if status else ' (no output)'))
            if status is not None:
                print(filename, file=progress_log)
    except KeyboardInterrupt:
        log.warning('Ending processing at user request')


def main():
    """
    Main app entry point
    """
    parser = ArgumentParser()
    get_args(parser)

    args = parser.parse_args()

    logging.basicConfig()

    run(args)


def run(args):
    """
    Secondary entry point to allow running from a different app
    """
    if args.progress:
        progressbar.streams.wrap_stderr()

    if args.debug:
        log.setLevel(logging.DEBUG)

    files = OrderedSet(args.files)
    files.update(find_files(args.input_dir))

    log_file = os.path.join(args.output_dir if args.output_dir is not None else '.', 'progress.log')

    done_files = get_progress(log_file)
    files.difference_update(done_files)

    num_files = len(files)

    log.debug('Processing {} files'.format(num_files))

    pbar_widgets = [
        progressbar.Counter(), '/', str(num_files), ' ',
        progressbar.Percentage(), ' ',
        progressbar.Bar(), ' ',
        progressbar.Timer(), ' ',
        progressbar.ETA(),
    ]

    with progressbar.ProgressBar(max_value=len(files), redirect_stdout=True, redirect_stderr=True, widgets=pbar_widgets) if args.progress else DummyProgressBar() as pbar:
        pbar.update(0)
        with open(log_file, 'a+', LINE_BUFFERED) as progress_log:
            # freeze parameters for multiprocessing
            job = partial(run_vid,
                          outdir=args.output_dir, mask_areas=MASK_AREAS,
                          show=args.show, codec=args.codec,
                          log_level=logging.DEBUG if args.debug else logging.INFO,
                          mem=args.mem, blur_scale=args.blur_scale, threshold=args.threshold, avg=args.avg,
                          fps=args.fps, min_time=args.mintime, cache_time=args.cachetime)

            if args.processes > 1:
                run_pool(job, args.processes, files, pbar, progress_log)
            else:
                run_map(job, files, pbar, progress_log)


def get_args(parser):
    """
    Set how to process command line arguments
    """
    parser.add_argument('files', nargs='*', help='Video files to find motion in')

    parser.add_argument('--input-dir', '-i', help='Input directory to process')
    parser.add_argument('--output-dir', '-o', help='Output directory for processed files')

    parser.add_argument('--codec', '-c', default='MP42', help='Codec to write files with')
    parser.add_argument('--masks', '-m', nargs='*', type=literal_eval, help='Areas to mask off in video') # XXX

    parser.add_argument('--blur_scale', '-b', type=int, default=20, help='Scale of gaussian blur size compared to video width')
    parser.add_argument('--threshold', '-t', type=int, default=12, help='Threshold for change in grayscale')
    parser.add_argument('--mintime', type=float, default=0.5, help='Minimum time for motion, in seconds')
    parser.add_argument('--cachetime', type=float, default=1.0, help='How long to cache, in seconds')
    parser.add_argument('--avg', '-a', type=float, default=0.1, help='How much to weight the most recent frame in the running average')
    parser.add_argument('--fps', '-f', type=int, default=30, help='Frames per second of input files')

    parser.add_argument('--processes', '-J', default=1, type=int, help='Number of processors to use')

    parser.add_argument('--progress', '-p', action='store_true', help='Show progress bar')
    parser.add_argument('--show', '-s', action='store_true', default=False, help='Show video processing')

    parser.add_argument('--mem', action='store_true', help='Run memory usage')
    parser.add_argument('--debug', '-d', action='store_true', help='Debug')


MASK_AREAS = [
    ((0, 0), (600, 300)),
    ((0, 0), (1450, 200)),
    ((1053, 370), (1215, 450)),
    ((1900, 0), (1920, 1080)),
    ((1600, 1080), (1920, 850), (1920, 1080))
]


if __name__ == '__main__':
    main()
