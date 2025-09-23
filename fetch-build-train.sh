# This file is supposed for the docker image to auto fetch, build and train the model

uv run tools/download_dataset.py
uv run create_dataset_fasttext.py
uv run train.py --epochs 5