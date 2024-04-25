import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

class Model:
    def __init__(self):
        self.df = pd.read_csv("CarsData.csv",header=None)
    
    def set_df(self, df):
        self.df = df

    def set_brand(self, brand: str):
        self.brand = brand
        self.df.loc[self.df['brand'] == self.brand]

    def split(data):
        X = data.drop("price", axis=1) #create X
        y = data["price"] #create y
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 14)
        return X_train, X_val, y_train, y_val
    
    def train(self, X, y):
        self.X_train = X
        self.y_train = y
        
        model = DecisionTreeRegressor() 
        model.fit(self.X_train, self.y_train)
        self.model = model

    def get_metrics(self, X_val, y_val):
        self.pred = self.model.predict(X_val)
        self.rmse = mean_squared_error(y_val, self.pred, squared = False)
        self.mae = mean_absolute_error(y_true = y_val, y_pred = self.pred)

    def predict(self, X_test):
        self.pred = self.model.predict(X_test)