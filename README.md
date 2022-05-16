# Tagalog Morphological Analysis

Produced research papers:
* https://animorepository.dlsu.edu.ph/etdm_comsci/1/
* https://ieeexplore.ieee.org/document/9310516

Authors: **Arian Yambao** - MSCS, **Charibeth K. Cheng** - PhD.

This repository shows an advancement towards the effort of solving different morphological phenomena that existis in the Tagalog language using deep learning. Originally there are 4 deep learning models used for the research. However, only two of the top performing are provided in this repository's analysis and application: (1) LSTM and (2) BERT. For the actual usage of the application, please refer to the notebook included.

In case of LSTM/BERT retraining, you may supply your corresponding model paths to either `lstm_predictor.py` and/or `bert_predictor.py`


**Important Note**: With regards to the transformer model, it is very important to
```pip install transformers==4.3.2``` due to known bugs and version mismatch during the time of development.

Proper Citation:
```
Yambao, A. N., & Cheng, C. K. (2020, December). Feedforward Approach to Sequential Morphological Analysis in the Tagalog Language. In 2020 International Conference on Asian Language Processing (IALP) (pp. 81-85). IEEE.
```