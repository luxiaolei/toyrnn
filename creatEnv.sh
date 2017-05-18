#!/bin/bash

conda env create -f env_py35_tf101.yml;
conda env create -f env_py36_tf101.yml;

conda env create -f env_py35_tf11_np111.yml; 

conda env create -f env_py35_tf11_np112.yml;

conda env create -f env_py36_tf11_np112.yml

