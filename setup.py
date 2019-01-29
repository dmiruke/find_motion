import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = [
    'progressbar2',
    'mem_top',
    'orderedset',
    'opencv-python',
    'imutils',
    'numpy',
    'pynput',
]

setuptools.setup(
    name="find_motion",
    version="0.0.2",
    author="Aegilops",
    author_email="41705651+aegilops@users.noreply.github.com",
    description="Processes video to detect motion, with tunable parameters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aegilops/find_motion",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
