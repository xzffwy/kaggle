# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import openpyxl
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error 
from sklearn.model_selection import train_test_split
from datetime import datetime
from itertools import combinations
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

name_y='SalePrice'

class HorsePrice:
    def __init__(self, dir):
        self.dir = dir

    def read_data(self):
        self.train_file_path = os.path.join(self.dir, "data/train.csv")
        self.test_file_path = os.path.join(self.dir, "data/test.csv")
        self.train_data = pd.read_csv(self.train_file_path)
        self.test_data = pd.read_csv(self.test_file_path)


    def handle_data(self):
        # prepare data
        self.train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
        y = self.train_data[name_y]
        X = self.train_data.drop([name_y], axis= 1)
        self.X_train_full, self.X_valid_full, self.y_train, self.y_valid = train_test_split(
            X, y, train_size=0.8, test_size = 0.2, random_state = 0
        )

    def build_prepocessor(self):
        # Select categorical columns with relatively low cardinality (convenient but arbitrary)
        categorical_cols = [cname for cname in self.X_train_full.columns if self.train_data[cname].nunique()< 10
                            and self.X_train_full[cname].dtype == 'object']

        # select numerical columns
        numerical_cols = [cname for cname in self.X_train_full.columns 
                          if self.X_train_full[cname].dtype in ['int64', 'float64']]

        my_cols = categorical_cols + numerical_cols

        # Preprocessing for numerical data
        numerical_transformer = SimpleImputer(strategy='constant')

        # Preprocessing for categorical data
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Bundle prepossing for numerical and categorical data
        self.prepocessor = ColumnTransformer(
            transformers = [
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )

    def build_model(self):
        # build model
        self.model = RandomForestRegressor(n_estimators=100, random_state=0)

    def build_pipeline(self):
        # build preprocessing and modeling code in pipeline
        self.pipeline = Pipeline(steps=[('prepocessor', self.prepocessor),
                                      ('model', self.model)])

    def cal_mae(self):
        self.pipeline.fit(self.X_train_full, self.y_train)
        preds = self.pipeline.predict(self.X_valid_full)

        score = mean_absolute_error(self.y_valid, preds)

        print('MAE:', score)

    def cal_cvs(self):
        self.pipeline.fit(self.X_train_full, self.y_train)
        preds = self.pipeline.predict(self.X_valid_full)
        y = self.train_data[name_y]
        X = self.train_data.drop([name_y], axis= 1)

        scores = -1 * cross_val_score(self.pipeline, 
                                      X, y,
                                      cv = 5,
                                      scoring='neg_mean_absolute_error')
        print("CVS scores:\n", scores)



HP = HorsePrice(".")
HP.read_data()
HP.handle_data()
HP.build_prepocessor()
HP.build_model()
HP.build_pipeline()
HP.cal_cvs()
        


