import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_log_error
from xgboost import XGBRegressor, XGBClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(r'/Users/praneethkumarpalepu/Downloads/Ames_House_Price.csv')

# basic description of the dataset.

class CreateModel:
    def __init__(self, dataframe, target_column, cardinality_threshold=0.75, recurring_value_threshold=0.85, imputable_threshold = 0.5, datatype_threshold = 20):
        self.dataframe = dataframe
        self.total_rows = len(self.dataframe)
        self.cardinality_threshold = cardinality_threshold
        self.recurring_threshold = recurring_value_threshold
        self.target_column = target_column
        all_columns = list(self.dataframe.columns)
        all_columns.remove(self.target_column)
        self.feature_columns = all_columns
        self.imputable_threshold = imputable_threshold
        self.datatype_threshold = datatype_threshold
    
    def get_datatype_of_column(self):
        df = self.dataframe
        datatype_dict = {}
        for column in list(df.columns):
            if df[column].dtypes in ['int64', 'float64']:
                if df[column].nunique() < (self.total_rows/self.datatype_threshold):
                    datatype_dict[column] = 'Categorical_Numeric'
                else:
                    datatype_dict[column] = 'Continuous'
            elif df[column].dtypes == 'object':
                if len(df[column].unique()) < (self.total_rows/self.datatype_threshold):
                    datatype_dict[column] = 'Categorical_Text'
                else:
                    datatype_dict[column] = 'object'
            
            else:
                datatype_dict[column] = df[column].dtypes
        return datatype_dict

    def get_cardinality_columns(self):
        cardinal_columns = {}
        for column in self.feature_columns:
            unique_rows = len(self.dataframe[column].unique())
            cardinality = unique_rows/self.total_rows
            if cardinality>=self.cardinality_threshold:
                cardinal_columns[column] = cardinality
        return cardinal_columns
    
    def get_recurring_value_columns(self):
        recurring_columns = {}
        for column in self.feature_columns:
            highest_recurring_value = self.dataframe[column].value_counts(normalize=True).iloc[0]/self.total_rows
            if highest_recurring_value >=self.recurring_threshold:
                recurring_columns[column] = highest_recurring_value
        return recurring_columns
    
    def get_columns_with_mv(self):
        mv_columns = {}
        non_imputable_columns = []
        for column in self.feature_columns:
            missing_value_percent = self.dataframe[column].isnull().sum()/self.total_rows
            if missing_value_percent > 0:
                mv_columns[column] = missing_value_percent
        for column in mv_columns.keys():
            if mv_columns[column] >= self.imputable_threshold:
                non_imputable_columns.append(column)
        return (mv_columns, non_imputable_columns)
    
    def get_deletable_columns(self):
        # identify columns with high cardinality,
        # with recurring values,
        # with missing values greater than the imputable threshold
        # add all these columns in a list
        cardinality_dict = self.get_cardinality_columns()
        recurring_dict = self.get_recurring_value_columns()
        mv_dict, non_imputable_columns = self.get_columns_with_mv()
        deletable_columns = []
        for column in cardinality_dict.keys():
            deletable_columns.append(column)
        for column in recurring_dict.keys():
            deletable_columns.append(column)
        for column in non_imputable_columns:
            deletable_columns.append(column)
        return deletable_columns
    
    def impute_missing_values(self, neighbors=5, custom_deletable_columns = None):
        df = self.dataframe.copy()
        deletable_columns = self.get_deletable_columns()
        if custom_deletable_columns is not None:
            for column in custom_deletable_columns:
                deletable_columns.append(column)
        deletable_columns = list(set(deletable_columns))
        df.drop(deletable_columns, axis=1, inplace=True)
        df.drop(self.target_column, axis=1, inplace=True)
        le = LabelEncoder()
        for column in df.columns:
            if df[column].dtypes == 'object':
                if df[column].isnull().sum() > 0:
                    not_null_mask = df[column].notnull()
                    df[column][not_null_mask] = le.fit_transform(df[column][not_null_mask])
                else:
                    df[column] = le.fit_transform(df[column])
                joblib.dump(le, f"helpers/{column}_le.joblib")

        x = df.to_numpy()
        knn = KNNImputer(n_neighbors=neighbors)
        imputed_arr = knn.fit_transform(x)
        imputed_df = pd.DataFrame(imputed_arr, columns=df.columns)
        joblib.dump(knn, "helpers/Imputer.joblib")
        return imputed_df
    
    def get_feature_importance(self, neighbors=5, estimators=250, custom_deletable_columns = None):
        imputed_df = self.impute_missing_values(neighbors=neighbors, custom_deletable_columns = custom_deletable_columns)
        columns = list(imputed_df.columns)
        x = imputed_df.to_numpy()
        y = self.dataframe[self.target_column]
        dtype_of_columns = self.get_datatype_of_column()
        if dtype_of_columns[self.target_column] in ['Categorical_Text', 'Categorical_Numeric']:
            rfc = RandomForestClassifier(n_estimators=estimators, random_state=0)
            rfc.fit(x, y)
            feature_importances = rfc.feature_importances_
        else:
            rfr = RandomForestRegressor(n_estimators=estimators, random_state=0)
            rfr.fit(x, y)
            feature_importances = rfr.feature_importances_
        feature_importance_dict = dict(zip(columns, feature_importances))
        feature_importance_dict_sorted = dict(sorted(feature_importance_dict.items(), key = lambda x:x[1], reverse=True))
        return feature_importance_dict_sorted, imputed_df, dtype_of_columns
    
    def fit_model(self, top_n_features=None, neighbors=5, estimators=250, size=0.2, custom_deletable_columns = None):
        feature_imp, imputed_df, dtype_of_columns = self.get_feature_importance(neighbors=neighbors, estimators = estimators, custom_deletable_columns = custom_deletable_columns)
        if top_n_features is not None:
            if top_n_features > len(imputed_df.columns):
                top_n_features = len(imputed_df.columns)
        else:
            top_n_features = len(imputed_df.columns)
        features = list(feature_imp.keys())[:top_n_features]
        X = imputed_df[features].to_numpy()
        y = self.dataframe[self.target_column]
        xtr, xte, ytr, yte = train_test_split(X, y, test_size=size, random_state=0)
        target_type = dtype_of_columns[self.target_column]
        if target_type in ['Categorical_Text', 'Categorical_Numeric']:
            le = LabelEncoder()
            ytr = le.fit_transform(ytr)
            yte = le.transform(yte)
            joblib.dump(le, "helpers/target_le.joblib")
            xgbc = XGBClassifier(n_estimators=estimators)
            xgbc.fit(xtr, ytr)
            xgb_pred = xgbc.predict(xte)
            accuracy = accuracy_score(yte, xgb_pred)
            joblib.dump(xgbc, "Models/Classifier_model.joblib")
            return round(accuracy*100, 2)
        elif target_type in ['Continuous']:
            xgbr = XGBRegressor(n_estimators=estimators)
            xgbr.fit(xtr, ytr)
            xgb_pred = xgbr.predict(xte)
            msle = mean_squared_log_error(yte, xgb_pred)
            joblib.dump(xgbr, "Models/regressor_model.joblib")
            return round(msle, 5)
