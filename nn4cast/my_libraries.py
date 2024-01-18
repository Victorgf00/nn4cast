#import the necessary libraries to run the predefined_functions.py utilities

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from scipy import stats as sts
import scipy.stats as stats
from scipy import signal
from scipy.fft import fft 
from cartopy import crs as ccrs # Cartography library
import cartopy as car
import matplotlib.patches as mpatches
import matplotlib.colors as colors
import cartopy.io.img_tiles as cimgt
import numpy.linalg as linalg
import numpy.ma as ma
from scipy.stats import pearsonr
import matplotlib.dates as mdates
from cartopy.util import add_cyclic_point #esto es para la banda de latitud que queda en blanco en 0ª
import matplotlib.dates as mdates
import cartopy.io.shapereader as shpreader
import xskillscore as xs
import time
import random
from tensorflow.keras.utils import plot_model
from kerastuner.tuners import RandomSearch
from sklearn.model_selection import KFold
import os
import shutil

#The following two lines are coded to avoid the warning unharmful message.
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import load_model

