# Spatial generalisability of power estimation for new PV plants

Code for the leave-one-site-out (LOSO) experiments in:

> *Yeni Fotovoltaik Santrallerde Güç Kestiriminin Uzamsal Genellenebilirliği: Makine Öğrenmesi ve Fiziksel Referansın Karşılaştırılması* (citation to be added)

Six algorithms (MLR, KNN, SVR, Random Forest, XGBoost, LSTM) are compared with a one-parameter physical reference (k·POA) on seven PV stations in seven countries. Each station is used once as an unseen test site.

## Repository structure

```
solar_ml_core.py              shared configuration, data loading, models and evaluation
notebooks/
  0_preprocessing/            one notebook per station + merge and LOSO folds
  1_training/                 one notebook per algorithm
  2_results/                  tables of the paper
data/
  raw/                        raw files (download, see below)
  processed/                  station files and LOSO folds (generated)
results/                      predictions and results (generated)
```

## Data

The raw data are not redistributed. Download them from the sources below and place them under `data/raw/`.

| Station | Source | Link | Licence | Expected file(s) |
|---|---|---|---|---|
| Atacama | ATAMOSTEC | https://doi.org/10.5281/zenodo.18317131 | CC BY 4.0 | `ATAMOS/Fixed_PERC/Fixed_PERC.csv` |
| Bolzano | Eurac Research | https://doi.org/10.5281/zenodo.18316996 | CC BY 4.0 | `EURAC/Trina/Trina.csv` |
| Elazığ | MİRBEY Enerji | https://doi.org/10.5281/zenodo.21405290 | Apache 2.0 | `Solar Power Plant Production Data.xlsx` |
| Helsinki | FMI | https://doi.org/10.57707/fmi-b2share.c7b3ec9f19324a6497639d6d8c0cde84 | CC BY | a `.csv` file with `Helsinki` in its name |
| Islamabad | Figshare | https://doi.org/10.6084/m9.figshare.22579948 | CC BY 4.0 | `weather_data.csv`, `solar_generation_1kW.csv`, `Building_1.csv` |
| Ningxia | ScienceDB | https://doi.org/10.57760/sciencedb.26284 | CC0 1.0 | `Photovoltaic power data.xlsx` |
| Yulara | DKASC | https://dkasolarcentre.com.au/source/yulara/yulara-1-fixed | DKASC terms | `5-Site_DG-PV1-DB-DG-M1A.csv` |

Processed hourly files for six stations are included in `data/processed/`. The Yulara file is not included because redistribution requires consent under the DKASC terms of use; generate it with `07_Yulara.ipynb` from the raw DKASC file. With these seven files, the experiments can be run from step 2 below.

Relative humidity for Bolzano, Elazığ and Yulara is retrieved from NASA POWER, and wind speed for Islamabad from the Open-Meteo archive, during preprocessing (internet access required).

## Running

Tested with Python 3.13 on Google Colab. Install the dependencies:

```
pip install -r requirements.txt
```

Run each notebook from its own folder, in this order:

1. `notebooks/0_preprocessing/01_…` to `07_…` write one file per station to `data/processed/`.
2. `notebooks/0_preprocessing/08_merge_and_LOSO_folds.ipynb` merges the stations and writes the LOSO folds to `data/processed/LOSO_Folds/`.
3. `notebooks/1_training/train_*.ipynb` train each algorithm and save predictions to `results/main/`.
4. `notebooks/2_results/01_main_results.ipynb` produces Tables 4–7.
5. `notebooks/2_results/02_ablation.ipynb` reruns LOSO without latitude, longitude and tilt (saved to `results/ablation/`) and produces Tables 9–10.

The training notebooks report metrics on raw predictions. The values in the paper are computed in `01_main_results.ipynb`, after negative predictions are clipped to zero, as described in the paper.

### Google Colab

Colab storage is temporary and long runs may disconnect, so clone the repository into your Google Drive:

```
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive
!git clone https://github.com/rhmano/loso-pv-generalisation.git
!pip install -q -r loso-pv-generalisation/requirements.txt
```

Place the raw files under `loso-pv-generalisation/data/raw/` in one of these ways:

- upload them with the Colab file browser;
- download them directly, for example `!wget -P loso-pv-generalisation/data/raw/ "<file-url>"` (Zenodo and Figshare sources);
- copy them from your Drive with `!cp`.

The Yulara file must be downloaded manually from the DKASC website.

Before running a notebook, change into its folder, for example:

```
%cd /content/drive/MyDrive/loso-pv-generalisation/notebooks/1_training
```

If a session disconnects, rerun the notebook; results already saved on Drive are kept, and the ablation notebook skips algorithms that are complete.

## Reproducibility

The random seed is 42. LSTM training is not fully deterministic, so LSTM results may differ slightly between runs. Training times depend on the hardware.

## Licence

The code is released under the MIT Licence (see `LICENSE`). The processed data files keep the licences of their original sources, listed in the Data section.
