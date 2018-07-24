#!/usr/bin/env python3

"""
Motion detection with OpenCV

With much help from...
https://www.pyimagesearch.com/

Caches images for a few frames before and after we detect movement

TODO: deal with 2GB max file output size :-( in OpenCV
        catch Exception and open another file with a 1-up number

DONE: spec an output folder, and an input folder

DONE: compress output somehow!

DONE: remove any 0 frame ouput files - better, only create output file once we have our first output

DONE: minimum length of motion before we think it is motion

DONE: multiprocessing with a Pool

TODO: deal well with KeyboardInterrupt when multiprocess. Wherever we are, fail in a way that kills the pool as a whole
    tried using a metaclass to decorate each method of the class, but that's not working

DONE: keep track of which files have processed in a dot file in the directory, allow resuming, skipping those
    already doing log.debug(), start to write to a file, and then read that file on resume

TODO: progress bar with ProgressBar2 module - one per file, and one for the whole run

DONE: sort input files by date, process in time order. Change sets to orderedset.

TODO: nicer way to specify filtered areas

TODO: return flag to say if the file completed without error, or was user interrupted (q or ctrl-c), or had an exception of some kind

TODO: logging in the workers - they should each create a logger and pass messages somehow to the master (does multiprocessing disable STDERR?)

DONE: imap_async, so that master gets results as they arrive, not once the whole job has completed, so that progress.log gets written to

TODO: allow pausing... somehow?
"""

import cv2
import imutils
from argparse import ArgumentParser
from collections import deque
from orderedset import OrderedSet
from functools import partial, wraps
import math
from multiprocessing import Pool
import sys
import os
import logging
import progressbar

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

LINE_BUFFERED = 1
BLACK = (0,0,0)
GREEN = (0,255,0)


def catch_exception(f):
    @wraps(f)
    def func(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except KeyboardInterrupt:
            print('KeyboardInterrupt!!')
            self = args[0]
            self.cleanup()
    return func


class ErrorCatcher(type):
    def __new__(cls, name, bases, dct):
        for m in dct:
            if hasattr(dct[m], '__call__'):
                dct[m] = catch_exception(dct[m])
        return type.__new__(cls, name, bases, dct)


class VideoMotion(object):
    __metaclass__ = ErrorCatcher

    def __init__(self, filename=None, outdir=None, fps=30, box_size=100, cache_frames=60, min_movement_frames=5, delta_thresh=7, avg=0.1, mask_areas=None, show=False, codec='MJPG', log_level=logging.INFO):
        self.filename = filename

        self.log = logging.getLogger(__name__)
        self.log.setLevel(log_level)

        if self.filename is None:
            raise Exception('Filename required')

        log.debug("Reading from {}".format(self.filename))

        self.outfile = None
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

        self._calc_min_area()
        self._make_gaussian()    
        self._load_video()

        self.movement = False
        self.movement_decay = 0
        self.movement_counter = 0

        self.current_frame = None
        self.wrote_frames = False


    def _calc_min_area(self):
        self.min_area = int(math.pow(self.box_size/50, 2))


    def _load_video(self):
        self.cap = cv2.VideoCapture(self.filename)
        self.ref_frame = None
        self.frame_cache = deque([], self.cache_frames)

        self._get_video_info()
        self.scale = self.box_size / self.frame_width


    def _make_outfile(self):
        if self.outdir == None:
            self.outfile_name = self.filename + '_motion.avi'
        else:
            self.outfile_name = os.path.join(self.outdir, os.path.basename(self.filename)) + '_motion.avi'

        log.debug("Writing to {}".format(self.outfile_name))

        self.outfile = cv2.VideoWriter(self.outfile_name, cv2.VideoWriter_fourcc(*self.codec), self.fps, (self.frame_width, self.frame_height))


    def _make_gaussian(self):
        gaussian_size = int(self.box_size/20)
        gaussian_size = gaussian_size + 1 if gaussian_size % 2 == 0 else gaussian_size
        self.gaussian = (gaussian_size, gaussian_size)


    def blur_frame(self, frame=None):
        frame = self.current_frame if frame is None else frame
        small = imutils.resize(frame.raw, width=self.box_size)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        frame.blur = cv2.GaussianBlur(gray, self.gaussian, 0)

    
    def read(self):
        (ret, frame) = self.cap.read()
        if not ret:
            return False
        else:
            self.current_frame = VideoFrame(frame)
            return True


    def output_frame(self, frame=None):
        frame = self.current_frame if frame is None else frame
        self.wrote_frames = True
        if self.show:
            cv2.imshow('frame', imutils.resize(frame.frame, width=500))
        if self.outfile is None:
            self._make_outfile()
        self.outfile.write(frame.raw)


    def decide_output(self):
        #log.debug(self.movement_counter)
        if (self.movement_counter >= self.min_movement_frames) or (self.movement_decay > 0):
            # show cached frames
            if self.movement:
                self.movement_decay = self.cache_frames
                #self.frame_cache.reverse()

                for f in self.frame_cache:
                    if f is not None:
                        self.output_frame(f)
                
                self.frame_cache.clear()
            # draw the text
            self.draw_text()

            self.output_frame()
        else:
            self.frame_cache.append(VideoFrame(self.current_frame.raw.copy()))


    def is_open(self):
        return self.cap.isOpened()


    @staticmethod
    def scale_area(area, scale):
        return [(int(a[0] * scale), int(a[1] * scale)) for a in area]


    def mask_off_areas(self, frame=None):
        frame = self.current_frame if frame is None else frame
        for area in self.mask_areas:
            cv2.rectangle(frame.blur, *VideoMotion.scale_area(area, self.scale), BLACK, cv2.FILLED)


    def find_diff(self, frame=None):
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
        frame = self.current_frame if frame is None else frame

        self.movement = False
        self.movement_decay -= 1 if self.movement_decay > 0 else 0    

        if frame.contours:
            # loop over the contours
            for c in frame.contours:
                # if the contour is too small, ignore it
                if cv2.contourArea(c) < self.min_area:
                    continue

                # compute the bounding box for the contour, draw it on the frame,
                # and update the text
                self.make_box(c, frame)
                self.movement_counter += 1
                self.movement = True

        if self.movement == False:
            self.movement_counter = 0

        return


    def draw_text(self, frame=None):
        frame = self.current_frame if frame is None else frame
        # draw the text
        cv2.putText(frame.frame, "Status: {}".format('motion' if self.movement else 'quiet'), (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return


    def make_box(self, c, frame=None):
        frame = self.current_frame if frame is None else frame
        (x, y, w, h) = cv2.boundingRect(c)
        area = ((x, y), (x + w, y + h))
        cv2.rectangle(frame.frame, *self.scale_area(area, 1/self.scale), GREEN, 2)
        return


    @staticmethod
    def key_pressed(key):
        return cv2.waitKey(1) & 0xFF == ord(key)


    def _get_video_info(self):
        self.amount_of_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.amount_of_frames == 0 or self.frame_width == 0 or self.frame_height == 0:
            raise Exception("Video info malformed - number of frames, height or width is 0")
        return


    def cleanup(self):
        if self.cap is not None:
            self.cap.release()

        if self.outfile is not None:
            self.outfile.release()
        
        cv2.destroyAllWindows()
        return


    def find_motion(self):
        while(self.is_open()):
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


class VideoFrame(object):
    """
    encapsulate frame stuff here, out of main video class
    """
    __metaclass__ = ErrorCatcher

    def __init__(self, frame):
        self.frame = frame
        self.raw = self.frame.copy()
        self.tresh = None
        self.contours = None
        self.frame_delta = None
        self.thresh = None


    def diff(self, ref_frame):
        self.frame_delta = cv2.absdiff(self.blur, cv2.convertScaleAbs(ref_frame))


    def threshold(self, thresh):
        self.thresh = cv2.threshold(self.frame_delta, thresh, 255, cv2.THRESH_BINARY)[1]


    def find_contours(self):
        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        self.thresh = cv2.dilate(self.thresh, None, iterations=2)
        cnts = cv2.findContours(self.thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        self.contours = cnts


# TODO: make nice way to pass in - I just hardcode what I need instead
MASK_AREAS = [
    ((0, 0), (600, 300)),
    ((0, 0), (1450, 200)),
]


def find_files(directory):
    """
    Finds files in the directory, recursively, sorts them by last modified time
    """
    files = [os.path.join(dirpath, f) for dirpath, dnames, fnames in os.walk(directory) for f in fnames] if directory is not None else []
    return OrderedSet([f[0] for f in sorted([(f, os.path.getmtime(f)) for f in files], key=lambda f: f[1])])


def run_vid(filename, outdir=None, mask_areas=None, show=None, codec=None, log_level=None):
    """
    Video creater and runner funnction to pass to multiprocessing pool
    """
    vid = VideoMotion(filename=filename, outdir=outdir, mask_areas=mask_areas, show=show, codec=codec, log_level=log_level)
    vid.find_motion()
    return filename


class DummyProgressBar(object):
    def __init__(self):
        pass

    def __exit__(self, *args):
        pass

    def __enter__(self, *args):
        pass


def main(args):
    files = OrderedSet(args.files)
    files.update(find_files(args.input_dir))

    log_file = os.path.join(args.output_dir, 'progress.log')

    with open(log_file, 'r') as progress_log:
        done_files = {f.strip() for f in progress_log.readlines()}
        log.debug('Not repeating {} files'.format(len(done_files)))
        files.difference_update(done_files)

    log.debug('Processing {} files'.format(len(files)))

    done = 0

    with progressbar.ProgressBar(max_value=len(files), redirect_stdout=True, redirect_stderr=True) if args.progress else DummyProgressBar() as bar:
        if bar is not None:
            bar.update(0)
        with open(log_file, 'a', LINE_BUFFERED) as progress_log:
            # freeze parameters for multiprocessing
            run = partial(run_vid, outdir=args.output_dir, mask_areas=MASK_AREAS, show=args.show, codec=args.codec, log_level=logging.DEBUG if args.debug else logging.INFO)

            if args.processes > 1:
                pool = Pool(processes=args.processes)
                files_processed = pool.map_async(run, files).get()
            else:
                files_processed = map(run, files)

            for filename in files_processed:
                if bar is not None:
                    done += 1
                    bar.update(done)
                log.debug('Done {}'.format(filename))
                print(filename, file=progress_log)


def get_args(parser):
    parser.add_argument('files', nargs='*', help='Video files to find motion in')
    parser.add_argument('--show', '-s', action='store_true', default=False, help='Show video processing')
    parser.add_argument('--input-dir', '-i', help='Input directory to process')
    parser.add_argument('--output-dir', '-o', help='Output directory for processed files')
    parser.add_argument('--debug', '-d', action='store_true', help='Debug')
    parser.add_argument('--codec', '-c', default='MP42', help='Codec to write files with')
    parser.add_argument('--processes', '-J', default=4, type=int, help='Number of processors to use')
    parser.add_argument('--progress', '-p', action='store_true', help='Show progress bar')


if __name__ == '__main__':
    parser = ArgumentParser()
    get_args(parser)

    args = parser.parse_args()

    if args.progress:
        progressbar.streams.wrap_stderr()

    logging.basicConfig()

    if args.debug:
        log.setLevel(logging.DEBUG)

    main(args)
