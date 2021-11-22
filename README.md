# IRE_major_project

## Installation 

Install Anaconda or Miniconda Package Manager from here.
Create a new virtual environment and install packages.
-- conda create -n ire python pandas tqdm
-- conda activate ire

Using a CUDA capable GPU is recommended. To install Pytorch with CUDA support:

-- conda install pytorch>=1.6 cudatoolkit=11.0 -c pytorch

CPU only:


-- conda install pytorch cpuonly -c pytorch

Install simpletransformers. pip install simpletransformers

For running demo 

Install streamlit 

pip3 install streamlit

## How to run 

Naviagate to src folder.

To run a model

-- python srcfilename.py

To run demo

-- python demo.py
