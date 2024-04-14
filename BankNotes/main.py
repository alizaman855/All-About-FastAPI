#Importing the Libraries
import uvicorn
from fastapi import FastAPI
from BankNotes import BankNote
import numpy as np
import pickle 
import pandas as pandas 

#Loading the model
pickle_in = open("classifier.pkl","rb")
model = pickle.load(pickle_in)

#Create app object
app = FastAPI()

#Index Route 
@app.get('/')
async def index():
    return {'Message: ',"Hello World"}

#Route with single parameter
@app.get('/{name}')
async def get_name(name:str):
    return {'Hello: ',f'{name}'}

@app.post('/predict')
async def prediction(data: BankNote):
    variance = data.varience
    skewness = data.skewness
    entropy = data.entropy
    curtosis = data.curtosis
    #Predicting the value 
    prediction = model.predict([[variance,skewness,curtosis,entropy]])
    if prediction[0]>0.5:
        prediction = "Fake Note"
    else:
        prediction = "Its a Bank Note"
    return{
        'Prediction: ':prediction
    }


