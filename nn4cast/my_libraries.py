#import the necessary libraries to run the predefined_functions.py utilities

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from scipy import stats as sts
import scipy.stats as stats
from scipy import signal
from cartopy import crs as ccrs 
import cartopy as car
import numpy.linalg as linalg
import numpy.ma as ma
from scipy.stats import pearsonr
import matplotlib.dates as mdates
from cartopy.util import add_cyclic_point 
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

