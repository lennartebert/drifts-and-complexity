# Drifts and Complexity - An Emprical Study
By Lennart Ebert (lennart.ebert@hu-berlin.de)

## Getting started

1. In a python 3.11 environment, install dependencies (see *requirements.txt*).
2. Place your data in folder /data/...
3. Create a local copy of the data dictionary (see configuration/data_dictionary_example.json) in same folder and configure for your datasets. The data_dictionary should be saved at configuration/data_dictionary.json
4. Run assess_datasets.py (e.g., python assess_dataset.py) for assessement of complexity in dataset windows
5. Run combine_results.py (e.g., pyhton combine_results.py) for computing aggregate tables across datasets
