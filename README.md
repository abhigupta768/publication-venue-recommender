# Publication Venue Recommender System
This repository contains code for the project Publication Venue Recommender System using Deep Learning

The initial model is built using a hierarchical attention based LSTM. The dataset used is [A-Miner](https://aminer.org/billboard/aminernetwork) Dataset which contains approximately 18Lac Papers related to Computer Science distributed accross 1.5Lac Journals.
We have taken only those Journals which have published more than 2000 papers. 

# Requirements
* Anaconda3
* python3
* Keras (run `pip install keras`)
* [Microsoft cntk backend](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-cntk-on-linux)
* Scikit-learn (run `pip install scikit-learn`)
* tqdm (run `pip install tqdm`)
* [Glove](https://nlp.stanford.edu/projects/glove/) Embeddings

# Usage
Just run `model.ipynb` using Jupyter Notebook

The code is meant to run with both CPU and GPU support on the cntk backend. 