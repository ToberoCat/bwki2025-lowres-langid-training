# BWKI 2025: Sprachidentifikation seltener Sprachen - Training

This repository provides everything you need to download the dataset, prepare it, and train the FastText-based language identification models.
You can run the pipeline locally **or** inside a Docker container.

## 📦 Setup

Install dependencies:

```bash
uv sync
```

## 📥 Download the Dataset

The raw dataset (\~26 GB) will be stored in the `artifacts/` folder.
Run:

```bash
uv run tools/download_dataset.py
```

* Use `--small` for a smaller test dataset, around 500MB in size (`uv run tools/download_dataset.py --small`)
* If the download was interrupted or you want to switch between `--small` and full mode, **delete the entire `artifacts/` folder first**.

## 🛠️ Prepare the Dataset

Clean, split, and prepare the data:

```bash
uv run create_dataset_fasttext.py
```

* This will create the processed dataset under `artifacts/splitParquet/`.
* **Note:** Running again requires deleting the `artifacts/` folder, as files are not overwritten.

## 🧑‍🏫 Train the Model

> ⏱️ Training is heavy. On the full dataset, 5 demo epochs take **\~0.5 day**; full runs take several days depending on hardware.


Once the dataset is prepared:

```bash
uv run train.py
```


This will:

* Train FastText experts for each writing system.
* Quantize models to `.ftz` format for smaller size and more efficient handling.
* Evaluate models and save predictions.

Output is written to `data/fasttext_experts/`, for example:

```
data/fasttext_experts/
└── Arab/
    ├── langclf/model.bin         # Original FastText model (before quantization; Most likely gone after quantization)
    ├── langclf_quant/model.ftz   # Quantized model (final)
    ├── langclf/preds_test.tsv    # Predictions on the test set
    └── langclf/preds_valid.tsv   # Predictions on the validation set
```


## 🐳 Run Everything in Docker

> ⏱️ Training is heavy. On the full dataset, 5 demo epochs take **\~0.5 day**; full runs take several days depending on hardware.

To build and run in a container:

```bash
docker compose up
```

The container will:

1. Download the dataset (`artifacts/`).
2. Prepare the dataset (`artifacts/splitParquet/`).
3. Train the model (`data/fasttext_experts/`).


## 🚀 Using the Models

The trained models can now be placed in the backend's `models/` folder to be served by the API.

## 👥 Team

Tobias Madlberger, Florian Schwanzer, Fabian Popov; HTL St. Pölten
Entwickelt für den Bundesweiten Informatikwettbewerb (BWKI) 2025.
In kooperation mit [DIGILINGDIV](https://digiling.univie.ac.at/digilingdiv/)
