# Attention-based multimodal fusion for single fine-particle classification

This repository contains the best-performing model from the paper: an attention-based fusion network that combines transmission electron microscopy (TEM) particle images with elemental composition and particle-size metadata to classify individual fine particles into seven types: dust, fly ash, metals, organic-rich, sea spray, soot and sulfur-rich.

## Requirements

The results in the paper were produced with Python 3.6.9, TensorFlow 2.3.1 and CUDA 11.2 on an NVIDIA GeForce RTX 2080 Ti (11 GB). Install dependencies with:

```bash
pip install -r requirements.txt
```


## Input data format
The script expects a single Excel file with one row per particle and the following columns:

| `Image Path` | Absolute or relative path to the TEM image of the particle |
| `diameter_um` | Particle diameter in µm |
| `Final_particle_type` | Ground-truth class label (string) |
| `Ag`, `Al`, `Ar`, … `Zr` (46 columns) | Elemental signal for each element; see `columns_of_interest` in the script for the full list. `Cu` is dropped automatically. |

A template with the exact column headers and a few placeholder rows is provided in [`example_data/metadata_template.xlsx`](example_data/metadata_template.xlsx).

Hyperparameters used in the paper (image size, batch size 16, 70 epochs, learning rate 1e-4, SMOTE ratio 1, 5 folds) are set at the top of the script and can be edited there.


## Citation

If you use this code, please cite the paper above and the Zenodo archive of this repository (DOI: 10.5281/zenodo.XXXXXXX).


