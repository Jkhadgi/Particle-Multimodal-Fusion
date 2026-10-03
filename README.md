# Advancing automatic classification of single fine particles via multimodal deep learning framework

This repository contains the official implementation of the best-performing model from the paper: an attention based fusion network that combines transmission electron microscopy (TEM) particle images with elemental composition and particle size metadata to classify individual fine particles into seven types: dust, fly ash, metals, organic-rich, sea spray, soot and sulfur-rich.

## Requirements

The results in the paper were produced with Python 3.6.9, TensorFlow 2.3.1, and CUDA 11.2 on an NVIDIA GeForce RTX 2080 Ti (11 GB). Install dependencies with:

bash
pip install -r requirements.txt

## Input data format
| Column | Description |
| :--- | :--- |
| `"Image Path"` | Absolute or relative path to the TEM image of the particle |
| `"diameter_um"` | Particle diameter in $\mu m$ |
| `"Final_particle_type"` | Ground-truth class label (string) |
| `"Ag", "Al", "Ar", … "Zr"` | Elemental signal for each element (46 columns total). "Cu" is dropped automatically; see `"columns_of_interest"` in the script for the full list. |

A template with the exact column headers and a few placeholder rows is provided in example_data/metadata_template.xlsx.

Hyperparameters used in the paperare set at the top of the script and can be changed.

## Citation

If you use this code, please cite the paper above and the Zenodo archive of this repository (DOI: 10.5281/zenodo.23114547).
