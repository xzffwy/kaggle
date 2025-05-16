# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import openpyxl
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error 
from sklearn.model_selection import train_test_split
from datetime import datetime
from itertools import combinations

class HorsePrice:
    def __init__(self, dir):
        self.dir = dir

    def handle_data(self):
        self.train_file_path = os.path.join(self.dir, "data/train.csv")
        self.test_file_path = os.path.join(self.dir, "data/test.csv")
        self.train_data = pd.read_csv(self.train_file_path)
        self.test_data = pd.read_csv(self.test_file_path)

    def handle_str(self):
        for col in self.train_data.select_dtypes(include='object').columns:
            # handle trian data
            self.train_data[col] = self.train_data[col].astype('category')
            self.train_data[col+'_num'] = self.train_data[col].cat.codes
            del self.train_data[col]
            # handle test data
            self.test_data[col] = self.test_data[col].astype('category')
            self.test_data[col+'_num'] = self.test_data[col].cat.codes
            del self.test_data[col]

    def build_regresson(self, random_s):
        self.random_state = random_s
        self.model = RandomForestRegressor(random_state = random_s)
        #self.model = HistGradientBoostingRegressor(random_state = random_s)

    def cal_mae(self, features_X, name_y):
        print(features_X)
        X = self.train_data[features_X]
        y = self.train_data[name_y]
        train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = self.random_state)
        self.model.fit(train_X, train_y)
        predictions = self.model.predict(val_X)
        val_mae = mean_absolute_error(predictions, val_y)
        return val_mae

    def predict_data(self, features_X, name_y, number = 10):
        X = self.train_data[features_X]
        y = self.train_data[name_y]
        test_X = self.test_data[features_X]
        self.model.fit(X, y)
        predictions = self.model.predict(test_X)
        output = pd.DataFrame({'Id':self.test_data.Id, name_y:predictions})
        print(len(output.columns))
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_name = f'data/output_predict_{number}_{timestamp}.csv'
        output.to_csv(os.path.join(self.dir, output_name), index = False)

    def get_column_names(self):
        column_names = list(self.test_data.columns)
        column_names.remove('Id')
        return column_names

    def write_to_excel(self, data):
        output_data = pd.DataFrame(data)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_name = f'data/output_{timestamp}.xlsx'
        output_data.to_excel(output_name, index = False)

    def handle_single_feature(self, column_names):
        mae_dict = {}
        for name in column_names:
            print(name)
            mae = self.cal_mae([name], 'SalePrice')
            mae_dict[name] = mae

        mae_sorted = sorted(mae_dict.items(), key = lambda x:x[1])
        return mae_sorted

    def handle_two_features(self, column_names):
        mae_dict = {}
        for i, j in combinations(column_names, 2):
            print(i, j)
            mae = self.cal_mae([i,j], 'SalePrice')
            mae_dict[i, j] = mae

        mae_sorted = sorted(mae_dict.items(), key = lambda x:x[1])
        return mae_sorted

    def handle_mae_to_pick_features(self, mae_sorted, number):
        features_d = {}
        # get the number of features
        print(mae_sorted)
        for item in mae_sorted:
            print(item)
            features_d[item[0]] = 0
            #for i in range(len(item[0])):
            #    features_d[item[0][i]] = 0
            if len(features_d) > number: 
                break

        return features_d.keys()


## hanlde code
Hp = HorsePrice(".")
Hp.handle_data()
Hp.handle_str()
column_names = Hp.get_column_names()
Hp.build_regresson(1)

# only one feature
mae_sorted = Hp.handle_single_feature(column_names)
#mae_sorted = Hp.handle_two_features(column_names)

# get the better features
for item in [10, 20, 30]:
    features = Hp.handle_mae_to_pick_features(mae_sorted, item)
    print(features, item)
    mae_value = Hp.cal_mae(features, 'SalePrice')
    print(mae_value)
    Hp.predict_data(features, 'SalePrice', item)
Hp.write_to_excel(mae_sorted)
        
# two features cal and 30 features to cal
#features = ['OverallQual', 'GrLivArea', 'Neighborhood_num', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea', 'Fireplaces', 'FullBath', 'MSSubClass', 'BedroomAbvGr', 'GarageType_num', '2ndFlrSF', 'FireplaceQu_num', 'LotFrontage', 'BsmtQual_num', 'HalfBath', 'KitchenQual_num', 'MasVnrArea', 'GarageFinish_num', 'BldgType_num', 'MSZoning_num', 'Foundation_num', 'BsmtFullBath', 'LandSlope_num', 'BsmtFinSF1', 'HouseStyle_num', 'YearBuilt', 'TotalBsmtSF', 'LotArea', 'GarageCond_num', 'MasVnrType_num']
# one features cal and 30 features to cal
#features = ['OverallQual', 'Neighborhood_num', 'BsmtQual_num', 'GarageCars', 'KitchenQual_num', 'GrLivArea', 'GarageArea', 'ExterQual_num', 'YearBuilt', 'GarageYrBlt', 'GarageFinish_num', 'TotalBsmtSF', 'YearRemodAdd', 'GarageType_num', 'FullBath', '1stFlrSF', 'Foundation_num', 'MSSubClass', '2ndFlrSF', 'TotRmsAbvGrd', 'BsmtFinType1_num']
#Hp.predict_data(features, 'SalePrice')