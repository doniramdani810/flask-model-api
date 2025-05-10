#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Install gdown for downloading models from Google Drive
pip install gdown

# Download model files from Google Drive
# Model A
gdown --id 1Km5IRryvsgCMTy9XCjZzC-fRlsDiIKZm --output best_model.pth
# Model B
gdown --id 1VdPhCp9d4RNLbnTjayNb3k4x-vIaxwi0 --output best_model1.pth

# Run Flask app using Gunicorn
exec gunicorn app:app --bind 0.0.0.0:$PORT
