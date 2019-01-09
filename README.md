# find_motion - Motion detection with OpenCV

Video processing script to detect motion, with tunable parameters.

Caches images for a few frames before and after it detects movement.

With much help from https://www.pyimagesearch.com/

## Usage

### Tuning

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

### Video sources

Input can come from a list of files, an input folder, or one or more cameras.

### Output

Files are written to an output directory in 2GB sections.

### Config file

If you use a config file with `--config` then it takes an INI file format with a single `[settings]` section.

Use the same names as the optional arguments given in the help for the fields of the settings file.

### Detailed usage

Use ```find_motion.py -h``` and ```pydoc find_motion.py``` to get more detailed documentation.

## Installing

You can clone the repo and call ```find_motion.py``` directly. Use ```pip3 install -r requirements.txt``` to install dependencies.

To install as a module do ```pip3 install .``` in the repo directory.