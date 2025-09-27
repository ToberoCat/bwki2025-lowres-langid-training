# This file is supposed for the docker image to build and train the model

uv run create_dataset_fasttext.py
uv run train.py --epochs 5