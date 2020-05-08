# Publication Venue Recommender System
This repository contains code for the project Modular-Hierarchical Attention Based Scholarly Venue Recommender System

The model is built using a hierarchical attention based Bi-LSTM. The dataset used is [A-Miner](https://aminer.org/billboard/aminernetwork) Dataset which contains approximately 18Lac Papers related to Computer Science distributed accross 1.5Lac Journals.
We have taken only those Journals which have published more than 500 papers. 

# Requirements
* Anaconda3
* python3
* PyTorch (run `pip install pytorch`)
* Scikit-learn (run `pip install scikit-learn`)

# Usage
Convert your dataset into the format mentioned in `format.txt`\\
Extract data from `format.txt` type files into pickle dumps using `extract_data.py`\\
Generate the PyTorch compatible dataset using the text and authors vocabularies using `generate_dataset.py`\\
To train a model on the training data, use `train.py`\\
To test the model on some test data, use `test.py`

The code is meant to run with both CPU and GPU support on the TensorFlow backend. 