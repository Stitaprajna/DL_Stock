
import uvicorn
import numpy as np
from fastapi import FastAPI,UploadFile, File
import pickle
from image import Image
import cv2
import os
from fastapi.responses import FileResponse
from keras.models import load_model 
#FastAPI
app = FastAPI()

def load_model1(pickel_file):
  pickle_in = open(pickel_file,'rb')
  model1 = pickle.load(pickle_in)
  return model1


def stock_predictor(url):
  today = cv2.imread(url)
  today_resized = cv2.resize(today,[224,224])
  today_f = np.array(today_resized, dtype='float')/255.0
  today_f = np.reshape(today_f,(1,224,224,3))
  Model1 = load_model('Model\Stock_Predictor.h5')
  f = Model1.predict(today_f)
  aaa = np.argmax(f,axis=1)
  if aaa[0] ==0:
    return 'Holding'
  elif aaa[0] == 1:
    return 'Bearish'
  else:
    return 'Bullish'



@app.get('/')
def index():
    return {'message': 'Hi'}



@app.post("/files")
async def UploadImage(file: bytes = File(...)):
    with open('image.jpg','wb') as image:
        image.write(file)
        image.close()
    return 'got it'



@app.get('/predict')
async def Predict_Stock_type():
 
  l = stock_predictor('image.jpg')
  return l
  

if __name__ == '__main__':
  uvicorn.run(app, host='127.0.0.1', port=8000)
