# find_motion - Motion detection with OpenCV

Video processing script to detect motion, with tunable parameters.

Caches images for a few frames before and after it detects movement.

With much help from https://www.pyimagesearch.com/

To reduce false motion from shadows, wind movement, noise, sun glare and so on, you can control:
* Threshold

  How much difference in gray level between frames counts as a difference
* Minimum time

  How long a difference in the background must be seen for to count as motion
* Cache time

  How long to cache frames before motion is detected and how long to keep showing them after motion stops
* Averaging

  What proportion of the 'background' is provided by the last seen frame (0 to 1)
* Blur scale

  How large the blurring of the image is during processing (larger sizes are slower, but less sensitive to tiny motions)
* Masking

  Areas to ignore. They are defined as tuples of coordinates. They can be provided as literals at the commandline or in a JSON file
  * square: `((0, 0), (100, 100))`
  * triangle: `((0, 0), (0, 100), (100, 0))`
  * JSON:
    ```
    [
        [[0, 0], [100, 100]],
        [[0, 0], [0, 100], [100, 0]]
    ]
    ```

# Usage

```
usage: find_motion.py [-h] [--input-dir INPUT_DIR] [--output-dir OUTPUT_DIR]
                      [--config CONFIG] [--codec CODEC]
                      [--masks [MASKS [MASKS ...]]] [--masks_file MASKS_FILE]
                      [--blur_scale BLUR_SCALE]
                      [--threshold THRESHOLD] [--mintime MINTIME]
                      [--cachetime CACHETIME] [--avg AVG] [--fps FPS]
                      [--processes PROCESSES] [--progress] [--show] [--mem]
                      [--debug]
                      [files [files ...]]

positional arguments:
  files                 Video files to find motion in

optional arguments:
  -h, --help            show this help message and exit
  --input-dir INPUT_DIR, -i INPUT_DIR
                        Input directory to process
  --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                        Output directory for processed files
  --config CONFIG       Config in INI format
  --codec CODEC, -c CODEC
                        Codec to write files with
  --masks [MASKS [MASKS ...]], -m [MASKS [MASKS ...]]
                        Areas to mask off in video
  --masks_file MASKS_FILE
                        File holding mask coordinates (JSON)
  --blur_scale BLUR_SCALE, -b BLUR_SCALE
                        Scale of gaussian blur size compared to video width
  --threshold THRESHOLD, -t THRESHOLD
                        Threshold for change in grayscale
  --mintime MINTIME     Minimum time for motion, in seconds
  --cachetime CACHETIME
                        How long to cache, in seconds
  --avg AVG, -a AVG     How much to weight the most recent frame in the
                        running average
  --fps FPS, -f FPS     Frames per second of input files
  --processes PROCESSES, -J PROCESSES
                        Number of processors to use
  --progress, -p        Show progress bar
  --show, -s            Show video processing
  --mem, -u             Run memory usage
  --debug, -d           Debug
```

If you use a config file with `--config` then it takes an INI file format with a single `[settings]` section.

Use the same names as the optional arguments given above for the fields of the settings file.

# Functions

    find_files(directory)
        Finds files in the directory, recursively, sorts them by last modified time
    
    get_args(parser)
        Set how to process command line arguments
    
    get_progress(log_file)
        Load the progress log file, get the list of files
    
    init_worker()
        Supress signal handling in the worker processes so that they don't capture SIGINT (ctrl-c)
    
    main()
        Main app entry point
    
    make_pbar_widgets(num_files)
        Create progressbar widgets
    
    read_masks(masks_file)
    
    run(args)
        Secondary entry point to allow running from a different app using an argparse Namespace
    
    run_map(job, files, pbar, progress_log)
    
    run_pool(job=None, processes=2, files=None, pbar=None, progress_log=None)
        Create and run a pool of workers
    
    run_vid(filename, **kwargs)
        Video creation and runner function to pass to multiprocessing pool

# Classes

## VideoMotion

    class VideoMotion(builtins.object)
     |  Class to read in a video, detect motion in it, and write out just the motion to a new file
     |  
     |  Methods defined here:
     |  
     |  __init__(self, filename=None, outdir=None, fps=30, box_size=100, cache_time=2.0, min_time=0.5, threshold=7, avg=0.1, blur_scale=20, mask_areas=None, show=False, codec='MJPG', log_level=20, mem=False)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  blur_frame(self, frame=None)
     |      Shrink, grayscale and blur the frame
     |  
     |  cleanup(self)
     |      Close the input file, output file, and get rid of OpenCV windows
     |  
     |  decide_output(self)
     |      Decide if we are going to put out this frame
     |  
     |  draw_box(self, area, frame=None)
     |  
     |  draw_text(self, frame=None)
     |      Put the status text on the frame
     |  
     |  find_diff(self, frame=None)
     |      Find the difference between this frame and the moving average
     |      
     |      Locate the contours around the thresholded difference
     |      
     |      Update the moving average
     |  
     |  find_motion(self)
     |      Main loop. Find motion in frames.
     |  
     |  find_movement(self, frame=None)
     |      Locate contours that are big enough to count as movement
     |  
     |  is_open(self)
     |      Return if the capture member is open
     |  
     |  make_box(self, contour, frame=None)
     |      Draw a green bounding box on the frame
     |  
     |  mask_off_areas(self, frame=None)
     |      Draw black polygons over the masked off areas
     |  
     |  output_frame(self, frame=None)
     |      Put a frame out to screen (if required) and file
     |      
     |      Initialise the output file if necessary
     |  
     |  output_raw_frame(self, frame=None)
     |      Output a raw frame, not a VideoFrame
     |  
     |  read(self)
     |      Read a frame from the capture member
     |  
     |  ----------------------------------------------------------------------
     |  Static methods defined here:
     |  
     |  key_pressed(key)
     |      Say if we pressed the key we asked for
     |  
     |  scale_area(area, scale)
     |      Scale the area by the scale factor

## VideoFrame  
    class VideoFrame(builtins.object)
     |  encapsulate frame stuff here, out of main video class
     |  
     |  Methods defined here:
     |  
     |  __init__(self, frame)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  cleanup(self)
     |      Actively destroy the frame
     |  
     |  diff(self, ref_frame)
     |      Find the diff between this frame and the reference frame
     |  
     |  find_contours(self)
     |      Find edges of the shapes in the thresholded image
     |  
     |  threshold(self, thresh)
     |      Find the threshold of the diff