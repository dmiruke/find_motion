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

TODO: nicer way to specify filtered areas, using literal_eval

TODO: return flag to say if the file completed without error, or was user interrupted
    (q or ctrl-c), or had an exception of some kind

TODO: logging in the workers - they should pass messages to the master on completion
    - override log and write output to master in a list

DONE: imap_async, so that master gets results as they arrive, not once the whole job has completed,
    so that progress.log gets written to

TODO: allow pausing: https://stackoverflow.com/questions/23449792/how-to-pause-multiprocessing-pool-from-execution

DONE: remove 'frame' processing - drawing box, adding text, etc. if we are not showing it - waste of time and memory. Just del self.frame etc. done with it.

TODO: fix motion detection so we stop writing frames after movement goes away (seems broken at the moment)
TODO: fix OutOfMemory problems? Can I catch that error and return that it happened, at least?
TODO: fix progress output to file - why are filenames not being written as they are processed?
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

from orderedset import OrderedSet

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
                 box_size=100, cache_frames=60, min_movement_frames=5,
                 delta_thresh=7, avg=0.1, mask_areas=None, show=False,
                 codec='MJPG', log_level=logging.INFO):
        self.filename = filename

        if self.filename is None:
            raise Exception('Filename required')

        log.debug("Reading from {}".format(self.filename))

        self.outfile = None
        self.outfile_name = None
        self.outdir = outdir

        self.fps = fps
        self.box_size = box_size
        self.cache_frames = cache_frames
        self.min_movement_frames = min_movement_frames
        self.delta_thresh = delta_thresh
        self.avg = avg
        self.mask_areas = mask_areas if mask_areas is not None else []
        self.show = show

        self.codec = codec

        log.debug(self.codec)

        # initialised in _load_video
        self.amount_of_frames = 0
        self.frame_width = 0
        self.frame_height = 0
        self.scale = None

        self._calc_min_area()
        self._make_gaussian()
        self._load_video()

        self.movement = False
        self.movement_decay = 0
        self.movement_counter = 0

        self.current_frame = None
        self.ref_frame = None
        self.frame_cache = []
        self.wrote_frames = False


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
        self.frame_cache = deque([], self.cache_frames)

        self._get_video_info()
        self.scale = self.box_size / self.frame_width


    def _make_outfile(self):
        """
        Create an output file based on the input filename and the output directory
        """
        if self.outdir is None:
            self.outfile_name = self.filename + '_motion.avi'
        else:
            self.outfile_name = os.path.join(self.outdir, os.path.basename(self.filename)) \
                                + '_motion.avi'

        log.debug("Writing to {}".format(self.outfile_name))

        self.outfile = cv2.VideoWriter(self.outfile_name,
                                       cv2.VideoWriter_fourcc(*self.codec),
                                       self.fps, (self.frame_width, self.frame_height))


    def _make_gaussian(self):
        """
        Make a gaussian for the blur using the box size as a guide
        """
        gaussian_size = int(self.box_size/20)
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
        if not self.show:
            del frame.frame


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
            cv2.imshow('frame', imutils.resize(frame.frame, width=500))
        if self.outfile is None:
            self._make_outfile()
        self.outfile.write(frame.raw)
        #frame.cleanup()


    def decide_output(self):
        """
        Decide if we are going to put out this frame
        """
        if (self.movement_counter >= self.min_movement_frames) or (self.movement_decay > 0):
            # show cached frames
            if self.movement:
                self.movement_decay = self.cache_frames

                for frame in self.frame_cache:
                    if frame is not None:
                        self.output_frame(frame)

                self.frame_cache.clear()
            # draw the text
            if self.show:
                self.draw_text()

            self.output_frame()
        else:
            self.frame_cache.append(VideoFrame(self.current_frame.raw.copy()))


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
        Draw black rectangles over the masked off areas
        """
        frame = self.current_frame if frame is None else frame
        for area in self.mask_areas:
            cv2.rectangle(frame.blur,
                          *VideoMotion.scale_area(area, self.scale),
                          BLACK, cv2.FILLED)


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
                if cv2.contourArea(contour) < self.min_area:
                    continue

                # compute the bounding box for the contour, draw it on the frame,
                # and update the text
                if self.show:
                    self.make_box(contour, frame)
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
        # pylint: disable=invalid-name
        frame = self.current_frame if frame is None else frame
        (x, y, w, h) = cv2.boundingRect(contour)
        area = ((x, y), (x + w, y + h))
        cv2.rectangle(frame.frame, *self.scale_area(area, 1/self.scale), GREEN, 2)
        return


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
            raise Exception("Video info malformed - number of frames, height or width is 0")
        return


    def cleanup(self):
        """
        Close the input file, output file, and get rid of OpenCV windows
        """
        if self.cap is not None:
            self.cap.release()

        if self.outfile is not None:
            self.outfile.release()

        self.current_frame.cleanup()
        del self.ref_frame
        for frame in self.frame_cache:
            frame.cleanup()
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

            self.blur_frame()
            self.mask_off_areas()
            self.find_diff()

            # draw contours and set movement
            self.find_movement()

            self.decide_output()

            if self.show:
                cv2.imshow('thresh', self.current_frame.thresh)
                cv2.imshow('delta', self.current_frame.frame_delta)
                cv2.imshow('raw', imutils.resize(self.current_frame.raw, width=500))

            if VideoMotion.key_pressed('q'):
                break

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
        for attr in ('frame', 'raw', 'thresh', 'contours', 'frame_delta', 'blur'):
            if hasattr(self, attr):
                delattr(self, attr)
        del self


def find_files(directory):
    """
    Finds files in the directory, recursively, sorts them by last modified time
    """
    files = [os.path.join(dirpath, f) for dirpath, dnames, fnames in os.walk(directory) for f in fnames] if directory is not None else []
    return OrderedSet([f[0] for f in sorted([(f, os.path.getmtime(f)) for f in files], key=lambda f: f[1])])


# pylint: disable=too-many-arguments
def run_vid(filename, outdir=None, mask_areas=None, show=None, codec=None, log_level=None):
    """
    Video creater and runner funnction to pass to multiprocessing pool
    """
    vid = VideoMotion(filename=filename, outdir=outdir, mask_areas=mask_areas,
                      show=show, codec=codec, log_level=log_level)
    err = vid.find_motion()
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
        log.debug('Did not find log file at {}'.format(log_file))
        return []

def run_pool(job=None, processes=2, files=None, pbar=None):
    """
    Create and run a pool of workers
    """
    results = []

    num_files = len(files)
    done = 0

    try:
        pool = Pool(processes=processes, initializer=init_worker)
        for filename in files:
            results.append(pool.apply_async(job, (filename,)))
        while True:
            num_done = [res.ready() for res in results].count(True)
            if num_done > done:
                done = num_done
                pbar.update(done)
            if num_done == num_files:
                log.debug("All processes completed")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        log.warning('Ending processing at user request')
        pool.terminate()

    return results


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

    log_file = os.path.join(args.output_dir, 'progress.log')

    done_files = get_progress(log_file)
    files.difference_update(done_files)

    num_files = len(files)

    log.debug('Processing {} files'.format(num_files))

    with progressbar.ProgressBar(max_value=len(files), redirect_stdout=True, redirect_stderr=True) if args.progress else DummyProgressBar() as pbar:
        if pbar is not None:
            pbar.update(0)
        with open(log_file, 'a', LINE_BUFFERED) as progress_log:
            # freeze parameters for multiprocessing
            job = partial(run_vid, outdir=args.output_dir, mask_areas=MASK_AREAS,
                          show=args.show, codec=args.codec,
                          log_level=logging.DEBUG if args.debug else logging.INFO)

            if args.processes > 1:
                results = run_pool(job, args.processes, files, pbar)
                for res in results:
                    if res.ready():
                        status, filename = res.get()
                        log.debug('Done {}{}'.format(filename, '' if status else ' (no output)'))
                        print(filename, file=progress_log)
            else:
                files_processed = map(job, files)

                done = 0

                for status, filename in files_processed:
                    if pbar is not None:
                        done += 1
                        pbar.update(done)
                    log.debug('Done {}{}'.format(filename, '' if status else ' (no output)'))
                    print(filename, file=progress_log)


def get_args(parser):
    """
    Set how to process command line arguments
    """
    parser.add_argument('files', nargs='*', help='Video files to find motion in')
    parser.add_argument('--show', '-s', action='store_true', default=False, help='Show video processing')
    parser.add_argument('--input-dir', '-i', help='Input directory to process')
    parser.add_argument('--output-dir', '-o', help='Output directory for processed files')
    parser.add_argument('--debug', '-d', action='store_true', help='Debug')
    parser.add_argument('--codec', '-c', default='MP42', help='Codec to write files with')
    parser.add_argument('--processes', '-J', default=4, type=int, help='Number of processors to use')
    parser.add_argument('--progress', '-p', action='store_true', help='Show progress bar')
    parser.add_argument('--masks', '-m', nargs='*', type=literal_eval, help='Areas to mask off in video') # XXX


MASK_AREAS = [
    ((0, 0), (600, 300)),
    ((0, 0), (1450, 200)),
]


if __name__ == '__main__':
    main()
