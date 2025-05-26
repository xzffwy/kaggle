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
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

name_y='SalePrice'

class HorsePrice:
    def __init__(self, dir):
        self.dir = dir

    def handle_data(self):
        self.train_file_path = os.path.join(self.dir, "data/train.csv")
        self.test_file_path = os.path.join(self.dir, "data/test.csv")
        self.train_data = pd.read_csv(self.train_file_path)
        self.test_data = pd.read_csv(self.test_file_path)
        self.train_y = self.train_data[name_y]
        self.train_data.drop([name_y], axis=1, inplace = True)

    def handle_nan_variables_drop(self):
        # show information
        print(self.train_data.shape)
        threshold = self.train_data.shape[0]/2
        missing_val_count_by_column = (self.train_data.isnull().sum())
        print(missing_val_count_by_column[missing_val_count_by_column > 0])

        # delete columns with more null, more than half rows
        self.train_data = self.train_data.loc[:,self.train_data.isna().sum() <= threshold]
        self.test_data = self.test_data.loc[:, self.test_data.isna().sum() <= threshold]
        print(self.train_data.shape)

        # remove rows with missing traget
        self.train_data.dropna(axis=0, subset=['SalePrice'], inplace = True)

    #    self.train_data.fillna(self.train_data.median(numeric_only = True), inplace = True)
    #    self.train_data.fillna("None", inplace = True)

    #    self.test_data.fillna(self.train_data.median(numeric_only = True), inplace = True)
    #    self.test_data.fillna("None", inplace = True)

    # kaggle
    def handle_nan_variables_impute(self):
        from sklearn.impute import SimpleImputer
        from sklearn.compose import ColumnTransformer

        categorical_cols = [cname for cname in self.train_data.columns 
                            if self.train_data[cname].dtype == 'object']

        numerical_cols = [cname for cname in self.train_data.columns
                          if self.train_data[cname].dtype in ['int64', 'float64']]
        print(len(self.train_data.columns))

        preprocessor = ColumnTransformer(transformers = [
            ('num', SimpleImputer(strategy='mean'), numerical_cols),
            ('cat', SimpleImputer(strategy='most_frequent'), categorical_cols),
        ], remainder = 'passthrough')
        impute_X_train = preprocessor.fit_transform(self.train_data)
        impute_X_test = preprocessor.transform(self.test_data)
        print(impute_X_train.shape)
        print(self.train_data.shape)
        
        self.train_data = pd.DataFrame(impute_X_train, 
                                       columns = self.train_data.columns, 
                                       index = self.train_data.index)
        self.test_data = pd.DataFrame(impute_X_test,
                                      columns = self.test_data.columns,
                                      index = self.test_data.index)


    def handle_nan_extension_to_impute(self):
        ### extent and then impute ###
        pass

    def handle_categories_drop(self):
        pass

    def handle_categories_ordinal_encoding(self):
        from sklearn.preprocessing import OrdinalEncoder
        print(self.train_data.head())
        print(self.test_data.head())

        # exclude the cols of data Id
        exclude_cols = ['Id']
        s = (self.train_data.dtypes == 'object')
        object_cols = list(s[s].index)
        for col in exclude_cols:
            if col in object_cols:
                object_cols.remove(col)
        
        # Columns that can be safely ordinal encoded
        good_label_cols = [col for col in object_cols if 
                            set(self.test_data[col]).issubset(set(self.train_data[col]))]

        # Problematic columns that will be dropped from the dataset
        bad_label_cols = list(set(object_cols) - set(good_label_cols))
        print("Categorical columns that will be ordinal encoded:", len(good_label_cols), good_label_cols)
        print("categorical columns that will be dropped from the dataset:", len(bad_label_cols), bad_label_cols)
        self.train_data.drop(bad_label_cols, axis=1, inplace=True)
        self.test_data.drop(bad_label_cols, axis=1, inplace=True)

        label_X_train = self.train_data.copy()
        label_X_test = self.test_data.copy()

        ordianl_encoder = OrdinalEncoder()
        label_X_train[good_label_cols] = ordianl_encoder.fit_transform(self.train_data[good_label_cols])
        label_X_test[good_label_cols] = ordianl_encoder.transform(self.test_data[good_label_cols])

        self.train_data = label_X_train
        self.test_data = label_X_test
        #for col in self.train_data.select_dtypes(include='object'):
        #    self.train_data[col] = OrdinalEncoder().fit_transform(self.train_data[col])
        #    self.test_data[col] = OrdinalEncoder().transform(self.test_data[col])

        #for col in self.train_data.select_dtypes(include='object').columns:
        #    # handle trian data
        #    self.train_data[col] = self.train_data[col].astype('category')
        #    self.train_data[col+'_num'] = self.train_data[col].cat.codes
        #    del self.train_data[col]
        #    # handle test data
        #    self.test_data[col] = self.test_data[col].astype('category')
        #    self.test_data[col+'_num'] = self.test_data[col].cat.codes
        #    del self.test_data[col]

        # low cardinality can use hot encoding
    def handle_categories_hot_encoding(self):
        from sklearn.preprocessing import OneHotEncoder

        s = (self.train_data.dtypes == 'object')
        object_cols = list(s[s].index)

        low_cardinality_cols = [col for col in object_cols if self.train_data[col].unique < 10]
        high_cardinality_cols = list(set(object_cols) - set(low_cardinality_cols))

        # drop high_cardinality_cols
        self.train_data.drop(high_cardinality_cols, axis=1, inplace='True')
        self.test_data.drop(high_cardinality_cols, axis=1, inplace='True')

        # Apply one-hot encoder to each column with categorical data
        OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        OH_train_data = pd.DataFrame(OH_encoder.fit_transform(self.train_data[low_cardinality_cols]))
        OH_test_data = pd.DataFrame(OH_encoder.transform(self.test_data[low_cardinality_cols]))

        # One-hot encoding removed index; put it back
        OH_train_data.index = self.train_data.index
        OH_test_data.index = self.test_data.index

        # Remove categorical columns (will replace with one-hot encoding)
        num_X_train = self.train_data.drop(object_cols, axis=1)
        num_X_test = self.test_data.drop(object_cols, axis=1)

        # Add one-hot encoded columns to numerical features
        self.train_data = pd.concat([num_X_train, OH_train_data], axis=1)
        self.test_data = pd.concat([num_X_test, OH_test_data], axis=1)

        # Ensure all columns have string type
        self.train_data.columns = self.train_data.columns.astype(str)
        self.test_data.columns = self.test_data.columns.astype(str)

    def handle_data_transform(self):
        print(self.test_data.head())
        self.train_X = self.train_data
        self.test_X = self.test_data
        self.train_y = np.log1p(self.train_y)


    def build_regresson(self, random_s):
        self.random_state = random_s
        self.model = []
        self.model.append(RandomForestRegressor(random_state = random_s, n_estimators = 100))
        self.model.append(GradientBoostingRegressor(random_state = random_s))
        self.model.append(RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0))

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
        }
        #self.model.append(GridSearchCV(estimator = RandomForestRegressor(random_state=42), 
        #                               param_grid = param_grid,
        #                               cv=5,
        #                               n_jobs = 1,
        #                               verbose = 1))

    def cal_mae(self, features_X):
        X = self.train_X[features_X]
        y = self.train_y
        train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = self.random_state)
        val_mae = 0
        for model in self.model:
            model.fit(train_X, train_y)
            predictions = model.predict(val_X)
            val_mae += mean_absolute_error(predictions, val_y)

        return val_mae/len(self.model)

    def cal_mae_cross_validation(self, features_X):
        from sklearn.model_selection import cross_val_score
        X = self.train_X[features_X]
        y = self.train_y

        model_scores = []
        for model in self.model:
            scores = -1 * cross_val_score(model, X, y, cv=5,
                                        scoring = 'neg_mean_absolute_error')
            print(model.__class__.__name__, " MAE scores:\n", scores)

            model_scores.append(sum(scores)/len(scores))

        return sum(model_scores)/len(model_scores)


    def predict_data(self, features_X, number = 10):
        X = self.train_X[features_X]
        y = self.train_y
        test_X = self.test_X[features_X]
        predictions = []
        for model in self.model:
            model.fit(X, y)
            predict = model.predict(test_X)
            predictions.append(self.handle_data_transform_return(predict))
        
        prediction = np.mean(predictions, axis=0)

        output = pd.DataFrame({'Id':int(self.test_X.Id), name_y:prediction})
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_name = f'data/output_predict_{number}_{timestamp}.csv'
        output.to_csv(os.path.join(self.dir, output_name), index = False)

    def get_column_names(self):
        column_names = list(self.test_X.columns)
        if 'Id' in column_names:
            column_names.remove('Id')
        return column_names

    def write_to_excel(self, data):
        output_data = pd.DataFrame(data)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_name = f'data/output_{timestamp}.xlsx'
        output_data.to_excel(output_name, index = False)

    def handle_mae_sorted(self, mae_sorted):
        features_dict = {}
        for num in [20]:
            features = Hp.handle_mae_to_pick_features(mae_sorted, num)
            mae_value = Hp.cal_mae_cross_validation(features)
            features_dict[mae_value] = features

        return features_dict

    def handle_single_feature(self, column_names):
        mae_dict = {}
        for name in column_names:
            mae_value = self.cal_mae_cross_validation([name])
            mae_dict[mae_value] = name
            #print(name, mae_value)

        mae_sorted = sorted(mae_dict.items(), key = lambda x:x[0])
        features_dict = self.handle_mae_sorted(mae_sorted)
        return features_dict

    def handle_two_features(self, column_names):
        mae_dict = {}
        for i, j in combinations(column_names, 2):
            mae_value = self.cal_mae_cross_validation([i,j])
            mae_dict[mae_value] = [i, j]
            #print(i, j, mae_value)

        mae_sorted = sorted(mae_dict.items(), key = lambda x:x[0])
        features_dict = self.handle_mae_sorted(mae_sorted)
        return features_dict

    def handle_all_features(self, column_names):
        mae_dict = {}
        mae_value = Hp.cal_mae(column_names)
        print(column_names, mae_value)
        # sort the mae_value
        mae_dict[mae_value] = column_names

        features_dict = {}
        features = Hp.handle_mae_to_pick_features(mae_dict, len(column_names))
        mae_value = Hp.cal_mae(features)
        features_dict[mae_value] = features
        return features_dict

    def handle_mae_to_pick_features(self, mae_sorted, number):
        features_d = {}
        # get the number of features
        #print(mae_sorted)
        for _, item in mae_sorted:
            if isinstance(item, str):
                features_d[item] = 0
                if len(features_d) > number: 
                    break
            else:
                for i in range(len(item)):
                    features_d[item[i]] = 0
                    if len(features_d) > number: 
                        break

        return features_d.keys()


    def handle_pick_features(self):
        print(type(self.train_X))
        X = self.train_X
        selector = SelectKBest(score_func = f_regression, k=30)
        X_new = selector.fit_transform(X, Hp.train_y)
        y = self.train_y
        train_X, val_X, train_y, val_y = train_test_split(X_new, y, random_state = self.random_state)
        self.model[0].fit(train_X, train_y)
        predictions = self.model[0].predict(val_X)
        val_mae = mean_absolute_error(predictions, val_y)
        print(val_mae)

        X_new_test = selector.transform(self.test_X)

        predictions = self.model[0].predict(X_new_test)
        output = pd.DataFrame({'Id':int(self.test_X.Id), name_y:predictions})
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_name = f'data/output_predict_pick_features_{timestamp}.csv'
        output.to_csv(os.path.join(self.dir, output_name), index = False)

    def handle_data_transform_return(self, predictions):
        return np.expm1(predictions)


## hanlde code
Hp = HorsePrice(".")
Hp.handle_data()
Hp.handle_nan_variables_impute()
Hp.handle_categories_ordinal_encoding()

Hp.handle_data_transform()
column_names = Hp.get_column_names()
Hp.build_regresson(1)

# use selector to pick 30 features
# the test result is not better
# Hp.handle_pick_features()

# pick small number features
features_dict = Hp.handle_single_feature(column_names)
#features_dict = Hp.handle_two_features(column_names)

# use small features to learn
#features_dict = Hp.handle_all_features(column_names)

#print(features_dict)

for item in features_dict.values():
    Hp.predict_data(item, len(item))

#Hp.write_to_excel(features_dict)


        
# two features cal and 30 features to cal
#features = ['OverallQual', 'GrLivArea', 'Neighborhood_num', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea', 'Fireplaces', 'FullBath', 'MSSubClass', 'BedroomAbvGr', 'GarageType_num', '2ndFlrSF', 'FireplaceQu_num', 'LotFrontage', 'BsmtQual_num', 'HalfBath', 'KitchenQual_num', 'MasVnrArea', 'GarageFinish_num', 'BldgType_num', 'MSZoning_num', 'Foundation_num', 'BsmtFullBath', 'LandSlope_num', 'BsmtFinSF1', 'HouseStyle_num', 'YearBuilt', 'TotalBsmtSF', 'LotArea', 'GarageCond_num', 'MasVnrType_num']
# one features cal and 30 features to cal
#features = ['OverallQual', 'Neighborhood_num', 'BsmtQual_num', 'GarageCars', 'KitchenQual_num', 'GrLivArea', 'GarageArea', 'ExterQual_num', 'YearBuilt', 'GarageYrBlt', 'GarageFinish_num', 'TotalBsmtSF', 'YearRemodAdd', 'GarageType_num', 'FullBath', '1stFlrSF', 'Foundation_num', 'MSSubClass', '2ndFlrSF', 'TotRmsAbvGrd', 'BsmtFinType1_num']
#Hp.predict_data(features, 'SalePrice')