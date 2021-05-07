# fly2p

Tools for analyzing two-photon (2p) imaging data collected with [Vidrio Scanimage software](https://vidriotechnologies.com/scanimage/). Relies on [scanimageReader](https://pypi.org/project/scanimage-tiff-reader/), which can be installed via 'pip install scanimage-tiff-reader'. Other dependencies are tracked using poetry.

I'm also working on expanding this codebased to work with imaging data collected with micromanger. 

### Organization:
The fly2p package contains the following submodules:
* **preproc**: Some file-format specific functions that extract metadata and load the imaging data. imgPreproc.py defines a data object to hold metadata and imaging data as well as basic proporcessing functions.
* **viz**: A collection of utility functions related to plotting flourescence traces and images.

In addition, the **scripts** folder contains notebooks that illustrate how to use functions in this module based on example files in **sample** (sample files are not currently pushed to repo).
