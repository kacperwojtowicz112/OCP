import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, confusion_matrix
import functions
import io
from fastapi import FastAPI, File, UploadFile
import uvicorn

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Welcome to application's API!"}

model=functions.Model()

class brando:
    def __init__(self):
        self.brand = ""
    def set_brand(self, brand):
         self.brand = brand
brand_selected= brando()

    
@app.post("/train/")
async def train(brand: str):
    if brand is not None:
        brand_selected.set_brand(brand)

        data = model.df.loc[model.df['brand'] == brand_selected]
        data = data.drop(["brand"], axis=1)

        X_train, X_val, y_train, y_val = model.split(data)
        model.train(X_train, y_train)
        model.get_metrics(X_val, y_val)

        pred = {'obs': y_val, 'pred': model.pred} 
        data = pd.DataFrame(pred)
        response = {"rmse": model.rmse, "mae": model.mae,"data": model.df}
        return response
    else:
        response = {"rmse": "", "mae": "", "data": ""}
        return response
    
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    df = df.loc[df['brand'] == brand_selected.brand]
    df = df.drop(["brand"], axis=1) 
    model.predict(df)
    pred = pd.DataFrame(model.pred, columns = ["pred"])
    pred["pred"] = pred['pred'].astype(int)
    response = {"pred": pred.to_dict(orient = "records")} 
    return response