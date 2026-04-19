import os
import sys
sys.path.insert(0, os.getcwd())
from models.predict import predict_crop, predict_soil, predict_disease, crop_model, soil_model, disease_model
print('crop_model', crop_model)
print('soil_model', soil_model)
print('disease_model', disease_model)
print('crop result', predict_crop(10,10,10,25,50,7,0))
from PIL import Image
im = Image.new('RGB', (10,10))
print('soil', predict_soil(im))
print('disease', predict_disease(im))
