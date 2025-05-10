#!/bin/bash

pip install gdown

gdown --id 1Km5IRryvsgCMTy9XCjZzC-fRlsDiIKZm --output best_model.pth
gdown --id 1VdPhCp9d4RNLbnTjayNb3k4x-vIaxwi0 --output best_model1.pth

gunicorn app:app --bind 0.0.0.0:$PORT
