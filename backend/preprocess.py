import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

df=pd.read_csv("CarsData.csv")

app = FastAPI()

@app.get("http://localhost:8501/process")
async def index(request: Request):
    d =  await request.json('data','')
    data=d.get('data','')
    return data

@app.post('/train')
async def index(data: str):
    if data=="ALL":
        
        df2=df[df["Manufacturer"]==data]
        
        
        return JSONResponse(content={'message':"Preprocess received response from Database",
                    'database': df2.to_dict(orient='records')})
    else:
        return JSONResponse(content={'message':"Preprocess received response from Database",
                    'database': df.to_dict(orient="records")})
