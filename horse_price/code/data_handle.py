# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import openpyxl

sample_path = "./data/sample_submission.csv"
data_path = "./data/output_predict_20250515_165913.csv"

sample_data = pd.read_csv(sample_path)
data_data = pd.read_csv(data_path)

print(len(sample_data.columns))
print(len(data_data.columns))
print(sample_data.head())
print(data_data.head())