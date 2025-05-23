import os
import torch


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET = 'weather'
FEATURES = ['year', 'month', 'day', 'hour_value', 'minute',
                'temperature', 'wind_speed', 'wind_angle', 'humidity', 'pressure', 'visibility']
GROSS_BIN = 8

