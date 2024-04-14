import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, confusion_matrix
import pickle
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any

app = FastAPI()

@app.get('http://localhost:8501/train')
async def index(request: Request):
    data= await request.json('data','')
    return data

    

@app.post("http://localhost:8501/app")
async def index(data:Dict[str, Any]):
    df=pd.read_json(data)
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train_x=train.drop("Manufacturer")
    train_y=train["Manufacturer"]
    test_x=test.drop("Manufacturer")
    test_y=test["Manufacturer"]

    model = DecisionTreeRegressor()
    model.fit(train_x, train_y)
        
    y_pred = model.predict(test_x)
    model=pickle.dumps(model)
    cm = confusion_matrix(test_y, y_pred)
    mse = mean_squared_error(test_y, y_pred)
    return JSONResponse({
        'y_pred': y_pred.to_dict(orient='records'),
        'test_y': test_y.to_dict(orient='values'),
        'confusion_matrix': cm.tolist(),
        'model': model,
        'mse':mse,
        })
