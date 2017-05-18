#!/bin/bash

conda env create -f env_py35_tf101.yml;
conda env create -f env_py36_tf101.yml;

conda env create -f env_py35_tf11_np111.yml; 
source activate env_py35_tf11_np111;
pip install tensorflow_gpu-1.1.0-cp35-cp35m-linux_x86_64.whl;
source deactivate;


conda env create -f env_py35_tf11_np112.yml; 
source activate env_py35_tf11_np112;
pip install tensorflow_gpu-1.1.0-cp35-cp35m-linux_x86_64.whl;
source deactivate;

