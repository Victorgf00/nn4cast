#import the necessary libraries
import sys
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
import matplotlib.colors as colors
from matplotlib.colors import from_levels_and_colors
from cartopy.util import add_cyclic_point 
import xskillscore as xs
import time
import random
from tensorflow.keras.utils import plot_model
from kerastuner.tuners import RandomSearch
from sklearn.model_selection import KFold
import os
import shutil
import yaml
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import math
import alibi
import matplotlib.patches as mpatches  
from alibi.explainers import IntegratedGradients

#The following two lines are coded to avoid the warning unharmful message.
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import load_model

class ClimateDataPreprocessing:
    """
    Class for performing data preprocessing and manipulation on climate data.

    Attributes:
        relative_path (str): The relative path to the NetCDF climate data file.
        lat_lims (tuple): A tuple containing the latitude limits (min, max) for the selected region.
        lon_lims (tuple): A tuple containing the longitude limits (min, max) for the selected region.
        time_lims (tuple): A tuple containing the time limits (start, end) for the selected time period.
        scale (float, optional): A scaling factor to apply to the data. Default is 1.
        regrid (bool, optional): Perform grid regridding. Default is False.
        regrid_degree (int, optional): The degree of grid regridding. Default is 2.
        overlapping (bool, optional): Create a cyclic point for overlapping data. Default is False.
        variable_name (str, optional): The name of the variable to extract. Default is None.
        months (list): List of months to select.
        months_to_drop (list): List of months to drop, default is 'None'
        years_out (list): List of output years.
        detrend (bool, optional): Detrend the data using linear regression. Default is False.
        one_output (bool, optional): Return one output for each year. Default is False.
        mean_seasonal_method(bool): Decides how to compute the seasonal aggregates. True if mean, False if sum (default True)
        
    Methods:
        preprocess_data(): Preprocess the climate data based on the specified parameters.
    """

    def __init__(
        self, relative_path, lat_lims, lon_lims, time_lims, scale=1,regrid_degree=1, variable_name=None, latitude_regrid=False,
        months=None, months_to_drop=None, years_out=None, detrend=False, detrend_window=10, jump_year=0, mean_seasonal_method=True,train_years=None):
        """
        Initialize the ClimateDataProcessing class with the specified parameters.

        Args:
            See class attributes for details.
        """
        self.relative_path = relative_path
        self.lat_lims = lat_lims
        self.lon_lims = lon_lims
        self.time_lims = time_lims
        self.scale = scale
        self.regrid_degree = regrid_degree
        self.variable_name = variable_name
        self.latitude_regrid = latitude_regrid
        self.months = months
        self.months_to_drop = months_to_drop
        self.years_out = years_out
        self.detrend = detrend
        self.detrend_window = detrend_window
        self.jump_year = jump_year
        self.mean_seasonal_method = mean_seasonal_method
        self.train_years = train_years
        
    def preprocess_data(self):
        """
        Preprocess the climate data based on the specified parameters.

        Returns:
            data (xarray.DataArray): The preprocessed climate data.
            latitude (xarray.DataArray): The latitude coordinates.
            longitude (xarray.DataArray): The longitude coordinates.
            data_red (xarray.DataArray): Selected and preprocessed data.
            anomaly (xarray.DataArray): Anomaly data (detrended and normalized).
            normalization (xarray.DataArray): Normalized data.
            mean_reference (xarray.DataArray): Mean reference data for the reference period.
            std_reference (xarray.DataArray): Standard deviation reference data for the reference period.
        """
        data = xr.open_dataset(self.relative_path, decode_times=True) / self.scale
        time = data['time'].astype('datetime64[M]')
        data = data.assign_coords(time=time)

        if 'latitude' not in data.coords or 'longitude' not in data.coords:
            data = data.rename({'lat': 'latitude', 'lon': 'longitude'})

        data = data.sortby('latitude', ascending=False)
        
        if self.lon_lims[1] > 180:
            data = data.assign_coords(longitude=np.where(data.longitude < 0, 360 + data.longitude, data.longitude)).sortby('longitude')
        else:
            data = data.assign_coords(longitude=(((data.longitude + 180) % 360) - 180)).sortby('longitude')

        if self.lat_lims[0] < self.lat_lims[1]:
            data = data.sel(latitude=slice(self.lat_lims[1], self.lat_lims[0]), longitude=slice(self.lon_lims[0], self.lon_lims[1]), time=slice(str(self.time_lims[0]), str(self.time_lims[1])))
        else:
            data = data.sel(latitude=slice(self.lat_lims[0], self.lat_lims[1]), longitude=slice(self.lon_lims[0], self.lon_lims[1]), time=slice(str(self.time_lims[0]), str(self.time_lims[1])))

        if self.regrid_degree!=0:
            lon_regrid = np.arange(self.lon_lims[0], self.lon_lims[1], self.regrid_degree)
            lon_regrid = lon_regrid[(lon_regrid >= self.lon_lims[0]) & (lon_regrid <= self.lon_lims[1])]
            if self.lat_lims[0] < self.lat_lims[1]:
                lat_regrid = np.arange(self.lat_lims[1], self.lat_lims[0]-self.regrid_degree, -self.regrid_degree)
                lat_regrid = lat_regrid[(lat_regrid >= self.lat_lims[1]) & (lat_regrid <= self.lat_lims[0])]
            else:
                lat_regrid = np.arange(self.lat_lims[0], self.lat_lims[1]-self.regrid_degree, -self.regrid_degree)
                lat_regrid = lat_regrid[(lat_regrid >= self.lat_lims[1]) & (lat_regrid <= self.lat_lims[0])]
            
            data = data.interp(longitude=np.array(lon_regrid), method='linear').interp(latitude=np.array(lat_regrid), method='linear')
        
        latitude = data.latitude
        longitude = data.longitude
        data = data[str(self.variable_name)]
        
        data_red = data.sel(time=slice(f"{self.time_lims[0]}", f"{self.time_lims[-1]}"))
        data_red = data_red.sel(time=np.isin(data_red['time.month'], self.months))

        if self.months_to_drop != ['None']:
            data_red = data_red.drop_sel(time=self.months_to_drop)

        data_red = data_red.groupby('time.month')
        
        mean_data = 0
        for i in self.months:
            mean_data += np.array(data_red[i])
            
        if self.mean_seasonal_method==True:  
            mean_data /= len(self.months)

        years_out= self.years_out + self.jump_year

        data_red = xr.DataArray(
            data=mean_data, dims=["year", "latitude", "longitude"],
            coords=dict(
                longitude=(["longitude"], np.array(data.longitude)),
                latitude=(["latitude"], np.array(data.latitude)), year=years_out
            ))

        mean_reference = (data_red.sel(year=slice(str(self.train_years[0]+self.jump_year), str(self.train_years[1]+self.jump_year)))).mean(dim='year')
        std_reference = (data_red.sel(year=slice(str(self.train_years[0]+self.jump_year), str(self.train_years[1]+self.jump_year)))).std(dim='year')

        anomaly = data_red - mean_reference
        normalization = anomaly / std_reference
        
        if self.detrend:
            print(f'Detrending {self.variable_name} data...')
            # Compute the rolling mean (past events only)
            rolling_mean = anomaly.rolling(year=self.detrend_window, center=False).mean()
            anomaly[self.detrend_window:] = anomaly[self.detrend_window:] - rolling_mean[self.detrend_window:]
            rolling_mean = normalization.rolling(year=self.detrend_window, center=False).mean()
            normalization[self.detrend_window:] = normalization[self.detrend_window:] - rolling_mean[self.detrend_window:]
            
        return latitude, longitude, data_red, anomaly, normalization, mean_reference, std_reference

class DataSplitter:
    """
    Class for preparing data for training the model.

    Attributes:
        train_years (list): List containing the start and end years of the training data.
        validation_years (list): List containing the start and end years of the validation data.
        testing_years (list): List containing the start and end years of the test data.
        predictor (xarray.DataArray): DataArray containing the predictor data.
        predictant (xarray.DataArray): DataArray containing the predictant data.
        jump_year (int, optional): Integer specifying the number of years of difference from the predictor and the predictand in each sample (default is 0).
        replace_nans_with_0_predictor (bool, optional): Substitute NaN values with 0 in the predictor (default is False).
        replace_nans_with_0_predictant (bool, optional): Substitute NaN values with 0 in the predictant (default is False).

    Methods:
        prepare_data(): Prepare the data for training the model.

    """

    def __init__(self, train_years, validation_years, testing_years, predictor, predictant, jump_year=0):#, replace_nans_with_0_predictor=False, replace_nans_with_0_predictant=False):
        """
        Initialize the DataSplitter class with the specified parameters.

        Args:
            See class attributes for details.
        """
        self.train_years = train_years
        self.validation_years = validation_years
        self.testing_years = testing_years
        self.predictor = predictor
        self.predictant = predictant
        self.jump_year = jump_year
        #self.replace_nans_with_0_predictor = replace_nans_with_0_predictor
        #self.replace_nans_with_0_predictant = replace_nans_with_0_predictant

    def prepare_data(self):
        """
        Prepare the data for training the model.

        Returns:
            X_train (numpy array): Numpy array containing the cleaned predictor data for the training data.
            X_valid (numpy array): Numpy array containing the cleaned predictor data for the validation data.
            X_test (numpy array): Numpy array containing the cleaned predictor data for the test data.
            Y_train (xarray.DataArray): DataArray containing the predictant data for the training data.
            Y_valid (xarray.DataArray): DataArray containing the predictant data for the validation data.
            Y_test (xarray.DataArray): DataArray containing the predictant data for the test data.
            input_shape (list): List containing the shape of the inputs.
            output_shape (list): List containing the shape of the outputs.
        """
        X_train = self.predictor.sel(year=slice(self.train_years[0], self.train_years[1]))
        X_valid = self.predictor.sel(year=slice(self.validation_years[0], self.validation_years[1]))
        X_test = self.predictor.sel(year=slice(self.testing_years[0], self.testing_years[1]))

        Y_train = self.predictant.sel(year=slice(self.train_years[0] + self.jump_year, self.train_years[1] + self.jump_year))
        Y_valid = self.predictant.sel(year=slice(self.validation_years[0] + self.jump_year, self.validation_years[1] + self.jump_year))
        Y_test = self.predictant.sel(year=slice(self.testing_years[0] + self.jump_year, self.testing_years[1] + self.jump_year))

        predictor = self.predictor.fillna(value=0)
        X = predictor.where(np.isfinite(predictor), 0)
        X_train = X.sel(year=slice(self.train_years[0], self.train_years[1]))
        X_valid = X.sel(year=slice(self.validation_years[0], self.validation_years[1]))
        X_test = X.sel(year=slice(self.testing_years[0], self.testing_years[1]))

        predictant = self.predictant.fillna(value=0)
        Y = predictant.where(np.isfinite(predictant), 0)
        Y = (Y.stack(space=('latitude','longitude'))).reset_index('space') #to convert the predictand to a (time, space) matrix
        Y_train = Y.sel(year=slice(self.train_years[0] + self.jump_year, self.train_years[1] + self.jump_year))
        Y_valid = Y.sel(year=slice(self.validation_years[0] + self.jump_year, self.validation_years[1] + self.jump_year))
        Y_test = Y.sel(year=slice(self.testing_years[0] + self.jump_year, self.testing_years[1] + self.jump_year))

        input_shape,output_shape = X_train[0].shape, Y_train[0].shape

        if np.ndim(Y_train[0])==1:
            output_shape= output_shape[0]

        if np.ndim(X_train[0])>=2:
            # Reshape the input data to include the "#channel" dimension necessary for 2DConvolutions
            X_train = np.expand_dims(X_train, axis=-1)
            X_valid = np.expand_dims(X_valid, axis=-1)
            X_test = np.expand_dims(X_test, axis=-1)
            input_shape= X_train[0].shape

        return X, X_train, X_valid, X_test, Y, Y_train, Y_valid, Y_test, input_shape, output_shape

class NeuralNetworkModel:
    """
    Class for creating a customizable deep neural network model, training it, and plotting its performance.

    Attributes:
        - input_shape (tuple): Shape of the input data.
        - output_shape (int or tuple): Shape of the output data.
        - layer_sizes (list of int): List of sizes for each hidden layer.
        - activations (list of str): List of activation functions for each hidden layer.
        - dropout_rates (list of float, optional): List of dropout rates for each hidden layer (optional).
        - kernel_regularizer (str, optional): Kernel regularizer for hidden layers (optional).
        - num_conv_layers (int): Number of convolutional layers (default: 0).
        - num_filters (int): Number of filters in each convolutional layer (default: 32).
        - pool_size (int): Pooling size for convolutional layers (default: 2).
        - kernel_size (int): Kernel size for convolutional layers (default: 3).
        - use_batch_norm (bool): Flag indicating whether to use batch normalization (default: False).
        - use_initializer (bool): Flag indicating whether to use He initialization (default: False).
        - use_dropout (bool): Flag indicating whether to use dropout (default: False).
        - use_initial_skip_connections (bool): Flag indicating whether to use skip connections from the initial layer (default: False).
        - use_intermediate_skip_connections (bool): Flag indicating whether to use skip connections between hidden layers (default: False).
        - one_output (bool): Flag indicating whether the output layer should have a single unit (default: False).
        - learning_rate (float): Learning rate for model training (default: 0.001).
        - epochs (int): Number of training epochs (default: 100).
        - random_seed (int): Seed for random number generation (default: 42).
        - outputs_path (directory): Path to store the outputs.

    Methods:
        - create_model(): Create a customizable deep neural network model.
        - train_model(): Train the created model.
        - performance_plot(): Plot the training and validation loss over epochs for a given model.

    """

    def __init__(self, input_shape, output_shape, layer_sizes, activations, dropout_rates=None, kernel_regularizer=None, num_conv_layers=0, num_filters=32, pool_size=2, kernel_size=3,use_batch_norm=False, use_initializer=False, use_dropout=False, use_initial_skip_connections=False, use_intermediate_skip_connections=False, one_output=False,learning_rate=0.001, epochs=100, random_seed=42):
        """
        Initialize the NeuralNetworkModel class with the specified parameters.

        Args:
            See class attributes for details.
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.dropout_rates = dropout_rates
        self.kernel_regularizer = kernel_regularizer
        self.num_conv_layers = num_conv_layers
        self.num_filters = num_filters
        self.pool_size = pool_size
        self.kernel_size = kernel_size
        self.use_batch_norm = use_batch_norm
        self.use_initializer = use_initializer
        self.use_dropout = use_dropout
        self.use_initial_skip_connections = use_initial_skip_connections
        self.use_intermediate_skip_connections = use_intermediate_skip_connections
        self.one_output = one_output
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_seed = random_seed

    def create_model(self, outputs_path=None, best_model=False):
        """
        Create a customizable deep neural network model.
        Args:
            - outputs_path (bool, default=None): For saving the models architectures
            - best_model (bool, default=False): For plotting and saving correctly the models architectures

        Returns:
            model (tf.keras.Model): Customizable deep neural network model.

        """
        # Define the input layer
        inputs = tf.keras.layers.Input(shape=self.input_shape, name="input_layer")
        x = inputs
        
        # Store skip connections
        skip_connections = []
        skip_connections.append(x)
        
        # Add convolutional layers with skip connections
        for i in range(self.num_conv_layers):
            x = tf.keras.layers.Conv2D(filters=self.num_filters, kernel_size=(self.kernel_size, self.kernel_size), padding='same', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.L2(l2=0.05), name=f"conv_layer_{i+1}")(x)
            if self.use_batch_norm:
                x = tf.keras.layers.BatchNormalization(name=f"batch_norm_{i+1}")(x)
            x = tf.keras.layers.Activation('elu', name=f'activation_{i+1}')(x)
            # Store the current layer as a skip connection
            x = tf.keras.layers.MaxPooling2D(pool_size=(self.pool_size, self.pool_size), name=f"max_pooling_{i+1}")(x)

        # Flatten the output
        x = tf.keras.layers.Flatten()(x)
        # Add dropout if applicable
        if self.use_dropout:
            x = tf.keras.layers.Dropout(self.dropout_rates[0], name=f"dropout_{1}")(x)
        # Add the output layer
        if self.kernel_regularizer is not "None":
            x = tf.keras.layers.Dense(units=self.layer_sizes[0], activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2=0.01), kernel_initializer='he_normal', name="dense_wit_reg_"+str(self.kernel_regularizer))(x)

        # Add fully connected layers
        for i in range(len(self.layer_sizes) - 1):
            # Store the current layer as a skip connection
            skip_connections.append(x)
            # Add a dense layer with He initialization if specified
            if self.use_initializer:
                x = tf.keras.layers.Dense(units=self.layer_sizes[i], kernel_initializer='he_normal', name=f"dense_{i+1}")(x)
            else:
                x = tf.keras.layers.Dense(units=self.layer_sizes[i], name=f"dense_{i+1}")(x)
            
            if self.use_batch_norm:
                x = tf.keras.layers.BatchNormalization(name=f"batch_norm_{i+1+self.num_conv_layers}")(x)
                
            x = tf.keras.layers.Activation(self.activations[i], name=f'activation_{i+1+self.num_conv_layers}')(x)
            
            # Merge with the skip connection if specified
            if self.use_intermediate_skip_connections:
                skip_connection_last = skip_connections[-1]
                skip_connection_last = tf.keras.layers.Dense(units=self.layer_sizes[i], kernel_initializer='he_normal', name=f"dense_skip_connect_{i+1}")(skip_connection_last)
                x = tf.keras.layers.Add(name=f"merge_skip_connect_{i+1}")([x, skip_connection_last])
        
        if self.kernel_regularizer is not "None":
            x = tf.keras.layers.Dense(units=self.layer_sizes[-1], activation='relu', kernel_regularizer=self.kernel_regularizer, kernel_initializer='he_normal', name="dense_wit_reg2_"+str(self.kernel_regularizer))(x)
            
        if self.use_initial_skip_connections:
            skip_connection_first= tf.keras.layers.Flatten()(skip_connections[0])
            skip_connection_first = tf.keras.layers.Dense(units=self.layer_sizes[-1], kernel_initializer='he_normal', name=f"initial_skip_connect")(skip_connection_first)
            x = tf.keras.layers.Add(name=f"merge_skip_connect")([x, skip_connection_first])
            
        # Define the output layer
        if np.ndim(self.output_shape)==0:
            outputs = tf.keras.layers.Dense(units=self.output_shape, kernel_initializer='he_normal', name=f"output_layer")(x) 
        else: 
            outputs = tf.keras.layers.Dense(units=self.output_shape[0]*self.output_shape[1], kernel_initializer='he_normal', name=f"output_layer")(x)
            outputs = tf.keras.layers.Reshape(self.output_shape)(outputs)


        # Create and summarize the model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        '''
        if outputs_path:
            if best_model==True:
                fig= tf.keras.utils.plot_model(model,to_file= outputs_path+'model_structure_best_model.png',show_shapes=True,show_dtype=False,show_layer_names=True,rankdir='TB',expand_nested=False,dpi=100,layer_range=None,show_layer_activations=True)
                #fig= tf.keras.utils.plot_model(model,to_file= outputs_path+'model_structure_best_model.png',show_shapes=True,show_dtype=False,show_layer_names=True,rankdir='TB',expand_nested=False,dpi=100)
            else:
                fig= tf.keras.utils.plot_model(model,to_file= outputs_path+'model_structure.png',show_shapes=True,show_dtype=False,show_layer_names=True,rankdir='TB',expand_nested=False,dpi=100,layer_range=None,show_layer_activations=True)
                #fig= tf.keras.utils.plot_model(model,to_file= outputs_path+'model_structure.png',show_shapes=True,show_dtype=False,show_layer_names=True,rankdir='TB',expand_nested=False,dpi=100)
        '''
        return model

    def train_model(self, X_train, Y_train, X_valid, Y_valid, outputs_path=None):
        """
        Train the created model.

        Args:
            - X_train (numpy.ndarray): Training input data.
            - Y_train (numpy.ndarray): Training output data.
            - X_valid (numpy.ndarray): Validation input data.
            - Y_valid (numpy.ndarray): Validation output data.
        Returns:
            model (tf.keras.Model): Customizable deep neural network model.
            history (tf.keras.callbacks.History): Training history of the model.
        """
        # Set random seeds for reproducibility
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        tf.compat.v1.random.set_random_seed(self.random_seed)

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
        model = self.create_model(outputs_path)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss=tf.keras.losses.MeanSquaredError())
        record = model.fit(X_train, Y_train, epochs=self.epochs, verbose=0, validation_data=(X_valid, Y_valid), callbacks=[callback])

        return model, record.history

    def performance_plot(self, history):
        """
        Plot the training and validation loss over epochs for a given model.

        Args:
            - history (tf.keras.callbacks.History): Training history of the model.

        Returns:
            - fig (matplotlib.figure.Figure): Figure of the training and validation loss over epochs
        """
        # Create a figure and subplot
        fig1 = plt.figure(figsize=(10, 5))
        ax1 = fig1.add_subplot(111)
        
        # Set the figure title
        fig1.suptitle('Models Performance')
        
        # Plot the training and validation loss
        ax1.plot(history['loss'], label='Training Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        
        # Set subplot title, labels, and legend
        ax1.set_title('Model 1')
        ax1.set_xlabel('# Epochs')
        ax1.set_ylabel('Loss Magnitude')
        ax1.legend(['Training', 'Validation'])
        plt.text(0.6, 0.7, 'Loss training: %.2f and validation %.2f '%(history['loss'][-1],history['val_loss'][-1]), transform=plt.gca().transAxes,bbox=dict(facecolor='white', alpha=0.8))
        # Show the plot
        plt.show()
        return fig1 
    
class ClimateDataEvaluation:
    """
    Class for evaluating a climate prediction model.

    Attributes:
        - X (numpy.ndarray): Input data for the climate model.
        - X_train (numpy.ndarray): Training input data.
        - X_test (numpy.ndarray): Testing input data.
        - Y (numpy.ndarray): Output data for the climate model.
        - Y_train (numpy.ndarray): Training output data.
        - Y_test (numpy.ndarray): Testing output data.
        - lon_y (numpy.ndarray): Longitudes for the output data.
        - lat_y (numpy.ndarray): Latitudes for the output data.
        - std_y (float): Standard deviation for the output data.
        - model (tf.keras.Model): Trained climate prediction model.
        - time_lims (tuple): A tuple containing the time limits (start, end) for the selected time period.
        - train_years (tuple): Years for training data.
        - testing_years (tuple): Years for testing data.
        - jump_year (int): Year offset for the predictions (default: 0).
        - map_nans (numpy.ndarray): Output data original map with nans, necessary to plot correctly the results.

    Methods:
        - plotter(): Tool for plotting model outputs.
        - prediction_vs_observation1D(): Plot the model's predictions vs. observations for both training and testing data.
        - evaluation(): Evaluate a trained model's predictions against test data.
        - correlations(): Calculate and visualize various correlation and RMSE metrics for climate predictions.
        - cross_validation(): Perform cross-validation for the climate prediction model.
        - correlations_pannel(): Visualize correlations for each ensemble member in a panel plot.
    """
    def __init__(
        self, X, X_train, X_test, Y, Y_train, Y_test, lon_y, lat_y, std_y, model, time_lims, train_years, testing_years,map_nans, jump_year=0, detrend_x=False, detrend_x_window=10, detrend_y=False, detrend_y_window=10, importances=False, region_atributted=None):
        """
        Initialize the ClimateDataEvaluation class with the specified parameters.
        
        Args:
            See class attributes for details.
        """
        self.X = X
        self.X_train= X_train
        self.X_test = X_test
        self.Y = Y
        self.Y_train= Y_train
        self.Y_test = Y_test
        self.lon_y = lon_y
        self.lat_y = lat_y
        self.std_y = std_y
        self.model = model
        self.time_lims = time_lims
        self.train_years = train_years
        self.testing_years = testing_years
        self.jump_year = jump_year
        self.map_nans = map_nans
        self.detrend_x = detrend_x
        self.detrend_x_window = detrend_x_window
        self.detrend_y = detrend_y
        self.detrend_y_window = detrend_y_window
        self.importances = importances
        self.region_atributted = region_atributted

    def plotter(self, data, levs, cmap1, l1, titulo, ax, pixel_style=False, plot_colorbar=True, acc_norm=None, extend='neither'):
        """
        Create a filled contour map using Cartopy.

        Args:
            data (2D array-like): Data values to be plotted.
            levs (list): Contour levels.
            cmap1 (Colormap): Matplotlib colormap for the plot.
            l1 (str): Label for the colorbar.
            titulo (str): Title for the plot.
            ax (matplotlib.axes.Axes): Matplotlib axis to draw the plot on.
            pixel_style (bool): Whether to use a pcolormesh to plot the map.

        Returns:
            fig (matplotlib.figure.Figure): Matplotlib figure.
        """
        # Create a filled contour plot on the given axis
        # Create a BoundaryNorm object
        cmap1 = plt.cm.get_cmap(cmap1)
        norm = colors.BoundaryNorm(levs, ncolors=cmap1.N, clip=True)
        if acc_norm:
            norm = acc_norm
            
        if pixel_style==True:
            # Create a pixel-based colormap plot on the given axis
            im = ax.pcolormesh(self.lon_y, self.lat_y, data, cmap=cmap1, transform=ccrs.PlateCarree(), norm=norm)
        else:
            im = ax.contourf(self.lon_y, self.lat_y, data, cmap=cmap1, levels=levs, extend=str(extend), transform=ccrs.PlateCarree(), norm=norm)

        # Add coastlines to the plot
        ax.coastlines(linewidth=0.75)

        # Set the title for the plot
        ax.set_title(titulo, fontsize=18)

        gl = ax.gridlines(draw_labels=True)
        gl.xlines = False
        gl.ylines = False
        gl.top_labels = False  # Disable top latitude labels
        gl.right_labels = False  # Disable right longitude labels
        
        if plot_colorbar==True:
            # Create a colorbar for the plot
            cbar = plt.colorbar(im, extend='neither', orientation='vertical', shrink=0.9, format="%2.1f")
            cbar.set_label(l1, size=15)
            cbar.ax.tick_params(labelsize=12)
            if acc_norm:
                cbar.set_ticks(ticks=levs, labels=levs)

        return im
    
    def evaluation(self, X_test_other=None, Y_test_other=None, model_other=None, years_other=None):
        """
        Evaluate a trained model's predictions against test data.

        Args:
            X_test_other (numpy.ndarray, optional): Other testing input data (default: None).
            Y_test_other (numpy.ndarray, optional): Other testing output data (default: None).
            model_other (tf.keras.Model, optional): Other trained model (default: None).
            years_other (list, optional): Other years for testing data (default: None).

        Returns:
            predicted1 (xarray.DataArray): Predicted sea-level pressure anomalies.
            correct_value (xarray.DataArray): True sea-level pressure anomalies.
        """
        # Predict sea-level pressure using the model
        if X_test_other is not None:
            X_test, Y_test, model, testing_years = X_test_other, Y_test_other, model_other, years_other
        else:
            X_test, Y_test, model, testing_years = self.X_test, self.Y_test, self.model, self.testing_years

        predicted = model.predict(np.array(X_test))
        test_years = np.arange(testing_years[0]+self.jump_year, testing_years[1]+self.jump_year+1,1)
        
        if np.ndim(predicted)<=2:
            def pongonans(matred,mat):
                out= mat.mean(axis = 0)
                out= np.tile(out, (matred.shape[0],1))
                out[:] = np.nan
                for i in range(0,matred.shape[0]):
                    out_1d= out[i,:]
                    out_1d[~np.isnan(mat.mean(axis = 0))] = matred[i,:]
                    out[i,:]=out_1d
                return out

            map_orig= self.map_nans
            nt, nlat, nlon= map_orig.shape
            map_reshape= np.reshape(np.array(map_orig),(nt, nlat*nlon))
            
            predicted= pongonans(predicted,np.array(map_reshape)) #if the predictant has nans
            Y_test= pongonans(Y_test,np.array(map_reshape)) #if the predictant has nans
            
            nt, nm= predicted.shape
            predicted= np.reshape(predicted, (nt, len(np.array(self.lat_y)), len(np.array(self.lon_y))))
            Y_test= np.reshape(Y_test, (nt, len(np.array(self.lat_y)), len(np.array(self.lon_y))))
        # Create xarray DataArray for predicted values
        predicted = xr.DataArray(
            data=predicted,
            dims=["time", "latitude", "longitude"],
            coords=dict(
                longitude=(["longitude"], np.array(self.lon_y)),
                latitude=(["latitude"], np.array(self.lat_y)),
                time=test_years))

        correct_value = xr.DataArray(
            data=Y_test,
            dims=["year", "latitude", "longitude"],
            coords=dict(
                longitude=(["longitude"], np.array(self.lon_y)),
                latitude=(["latitude"], np.array(self.lat_y)),
                year=test_years))

        # De-standardize the predicted and true values to obtain anomalies
        predicted = predicted * self.std_y
        correct_value = correct_value * self.std_y
        
        return predicted, correct_value

    def correlations(self, predicted, correct_value, outputs_path, threshold, units, months_x, months_y, var_x, var_y, predictor_region, best_model=False):
        """
        Calculate and visualize various correlation and RMSE metrics for climate predictions.

        Args:
            - predicted (xr.DataArray): Predicted sea-level pressure anomalies.
            - correct_value (xr.DataArray): True sea-level pressure anomalies.
            - outputs_path (str): Output path for saving the plot.
            - threshold (float): Threshold for significance in correlation.
            - units (str): Units for the plot.
            - periodo (str): Time period for the correlation calculation.

        Returns:
            - fig (matplotlib.figure.Figure): Matplotlib figure.
        """

        # Calculate spatial and temporal correlations, RMSE, and p-values
        predictions = predicted
        observations = correct_value.rename({'year': 'time'})
        spatial_correlation = xr.corr(predictions, observations, dim='time')
        p_value = xs.pearson_r_p_value(predictions, observations, dim='time')
        # Plot significant pixels with point hatching
        skill = ma.masked_where(p_value > threshold, spatial_correlation)

        # Calculate temporal correlations
        temporal_correlation = xr.corr(predictions, observations, dim=('longitude', 'latitude'))
        
        # Calculate spatial and temporal RMSE
        spatial_rmse = np.sqrt(((predictions - observations) ** 2).mean(dim='time'))
        temporal_rmse = np.sqrt(((predictions - observations) ** 2).mean(dim=('longitude', 'latitude')))

        # Determine significant pixels
        sig_pixels = np.abs(p_value) <= threshold

        # Set non-significant pixels to NaN
        spatial_correlation_sig = spatial_correlation.where(sig_pixels)

        # Create a new figure
        fig = plt.figure(figsize=(15, 7))

        # Subplot 1: Spatial Correlation Map
        ax = fig.add_subplot(221, projection=ccrs.PlateCarree(0))
        data = spatial_correlation
        rango = 1
        acc_clevs = [-1,-0.9,-0.8,-0.7,-0.6,-0.4,-0.2,0.2,0.4,0.6,0.7,0.8,0.9,1]
        # 4 valores mas
        colors = np.array([[0,0,255],[0,102,201],[119,153,255],[119,187,255],[170,221,255], [170,255,255],[170,170,170], [255,255,0],
                        [255,204,0], [255,170,0],[255,119,0],[255,0,0],[119,0,34]], np.float32) / 255.0

        acc_map, acc_norm = from_levels_and_colors(acc_clevs, colors)
        ClimateDataEvaluation.plotter(self, data, acc_clevs, acc_map,'Correlation', 'ACC map', ax, pixel_style=True,acc_norm=acc_norm)
        lon_sig, lat_sig = spatial_correlation_sig.stack(pixel=('longitude', 'latitude')).dropna('pixel').longitude, \
                        spatial_correlation_sig.stack(pixel=('longitude', 'latitude')).dropna('pixel').latitude
        
        hatch_mask = p_value < threshold
        ax.contourf(hatch_mask.longitude, hatch_mask.latitude, hatch_mask, levels=[0, 0.5, 1], hatches=['//', ''], alpha=0, transform=ccrs.PlateCarree())

        #ax.scatter(lon_sig, lat_sig, s=5, c='k', marker='.', alpha=0.5, transform=ccrs.PlateCarree(), label='Significant')

        # Subplot 2: Temporal Correlation Plot
        ax1 = fig.add_subplot(222)
        data = {'time': temporal_correlation.time, 'Predictions correlation': temporal_correlation,}
        df = pd.DataFrame(data)
        df.set_index('time', inplace=True)
        color_dict = {'Predictions correlation': 'blue'}
        width = 0.5
        for i, col in enumerate(df.columns):
            ax1.bar(df.index, df[col], width=width, color=color_dict[col], label=col)

        ax1.set_ylim(ymin=-1, ymax=+1)
        ax1.set_title('Time series of global ACC', fontsize=18)
        ax1.legend(loc='upper right')

        # Subplot 3: Spatial RMSE Map
        ax4 = fig.add_subplot(223, projection=ccrs.PlateCarree(0))
        data = spatial_rmse
        rango= int(np.nanmax(np.array(data)))
        ClimateDataEvaluation.plotter(self, data, np.linspace(0, rango+1, 10), 'OrRd','RMSE', 'RMSE map', ax4, pixel_style=True)
        
        # Subplot 4: Temporal RMSE Plot
        ax5 = fig.add_subplot(224)
        data = {'time': temporal_rmse.time, 'Predictions RMSE': temporal_rmse}
        df = pd.DataFrame(data)
        df.set_index('time', inplace=True)
        color_dict = {'Predictions RMSE': 'orange'}
        for i, col in enumerate(df.columns):
            ax5.bar(df.index, df[col], width=width, color=color_dict[col], label=col)

        ax5.set_title('Time series of global RMSE', fontsize=18)
        ax5.legend(loc='upper right')
        ax5.set_ylabel(f'{units}')
        fig.suptitle(f'Comparison of metrics of {var_y} from months "{months_y}"  when predicting with {predictor_region} {var_x} from months "{months_x}"', fontsize=20)
        # Adjust layout and save the figure
        plt.tight_layout()
        if best_model==True:
            plt.savefig(outputs_path + 'correlations_best_model.png')
        else:
            plt.savefig(outputs_path + 'correlations.png')

        return fig

    def atributions(self, model, X_test, lat_lims, lon_lims, std_y, test_years):
        # Create latitude and longitude grids
        latitudes, longitudes = np.array(self.lat_y), np.array(self.lon_y)
        longitude_grid, latitude_grid = np.meshgrid(longitudes, latitudes)

        # Flatten latitude and longitude grids
        flat_latitude,flat_longitude = latitude_grid.flatten(), longitude_grid.flatten()

        # Define latitude and longitude limits
        lat_min, lat_max = lat_lims[0], lat_lims[1]
        lon_min, lon_max = lon_lims[0], lon_lims[1]

        # Find indices of points within limits
        indices_within_limits = list(np.where((flat_latitude >= lat_min) & (flat_latitude <= lat_max) &(flat_longitude >= lon_min) & (flat_longitude <= lon_max))[0])
        # Ensure indices are within valid target range
        valid_target_range = np.prod(model.output_shape[1:]) 
        indices_within_limits = [i for i in indices_within_limits if i < valid_target_range]
        
        if not indices_within_limits:
            raise ValueError("No indices within limits found within valid target range.")
        
        sample_index = 0
        X_test_sample = X_test[sample_index:sample_index + 1]
        baseline = X_test_sample * 0

        ig = IntegratedGradients(model,layer=None,method="gausslegendre",n_steps=50,internal_batch_size=100)

        importances_mean_year= []
        print('Computing importances for each year over the selected region...')
        for j in range(0,X_test.shape[0]):
            # Select a specific instance (single event)
            sample_index = j
            # Get a single sample
            X_test_sample = X_test[sample_index:sample_index + 1]
            baseline = X_test_sample * 0
            attributions_all= []
            for i in indices_within_limits:
                explanation = ig.explain(np.array(X_test_sample), baselines=np.array(baseline), target=int(i))
                attributions = explanation.attributions[0]
                #nex we multiply the importance by the std_slp of each point
                attributions= attributions*(np.array(std_y).flatten()[int(i)])
                attributions_all.append(attributions)
            importances_mean= np.mean(np.array(attributions_all), axis=0).ravel() #if the input is flattened
            #importances_mean= np.reshape(np.mean(np.array(attributions_all), axis=0), (np.mean(np.array(attributions_all), axis=0).shape[1:3])) #if the input is (time, lat, lon, channel)
            importances_mean_year.append(importances_mean)
        
        nt,nlat, nlon= (np.array(X_test)).shape
        importances = xr.DataArray(
            data=np.reshape(np.array(importances_mean_year),(nt,nlat,nlon)),
            dims=["time", "latitude", 'longitude'],
            coords=dict(
                time=(np.array(test_years)),
                latitude= np.array(X_test.latitude),
                longitude= np.array(X_test.longitude)))
        return importances

    def cross_validation(self, n_folds, model_class):
        """
        Perform cross-validation for the climate prediction model.

        Args:
            - n_folds (int): Number of folds for cross-validation.
            - model_class (Type[NeuralNetworkModel]): Class for creating a customizable deep neural network model.

        Returns:
            - predicted_global (xarray.DataArray): Concatenated predicted values from all folds.
            - correct_value (xarray.DataArray): Concatenated true values from all folds.
        """

        # Define the KFold object
        kf = KFold(n_splits=n_folds, shuffle=False)

        # create an empty list to store predicted values from each fold
        predicted_list = []
        correct_value_list = []
        importances_list = []
        years = np.arange(self.time_lims[0], self.time_lims[-1]+1,1)
        years_division_list = []
        # Loop over the folds
        for i, (train_index, testing_index) in enumerate(kf.split(self.X)):
            print(f'Fold {i+1}/{n_folds}')
            print('Training on:',years[train_index])
            print('Testing on:', years[testing_index])
            # Get the training and validation data for this fold
            mean_reference_x, mean_reference_y  = np.mean((self.X[train_index]), axis=0), np.mean((self.Y[train_index]), axis=0)
            std_reference_x, std_reference_y  = np.std((self.X[train_index]), axis=0), np.std((self.Y[train_index]), axis=0)
            X, Y = (self.X-mean_reference_x)/std_reference_x, (self.Y-mean_reference_y)/std_reference_y
            X,Y = X.fillna(value=0), Y.fillna(value=0)
            X,Y = X.where(np.isfinite(X), 0), Y.where(np.isfinite(Y), 0)
            if np.ndim(Y)>2:
                Y = (Y.stack(space=('latitude','longitude'))).reset_index('space') #to convert the predictand to a (time, space) matrix

            if self.detrend_x:
                rolling_mean_x = X.rolling(year=self.detrend_x_window, center=False).mean()
                X[self.detrend_x_window:] = X[self.detrend_x_window:] - rolling_mean_x[self.detrend_x_window:]

            if self.detrend_y:
                rolling_mean_y = Y.rolling(year=self.detrend_y_window, center=False).mean()
                Y[self.detrend_y_window:] = Y[self.detrend_y_window:] - rolling_mean_y[self.detrend_y_window:]

            X_train_fold = X[train_index]
            Y_train_fold = Y[train_index]
            X_testing_fold = X[testing_index]
            Y_testing_fold = Y[testing_index]

            model_cv = model_class.create_model()
            model_cv, record= model_class.train_model(X_train=X_train_fold, Y_train=Y_train_fold, X_valid=X_train_fold, Y_valid=Y_train_fold)
            predicted_value,observed_value= ClimateDataEvaluation.evaluation(self, X_test_other=X_testing_fold, Y_test_other=Y_testing_fold, model_other=model_cv, years_other=[(years[testing_index])[0],(years[testing_index])[-1]])
            predicted_list.append(predicted_value)
            correct_value_list.append(observed_value)
            years_division_list.append(years[testing_index])
            if self.importances:
                imp_fold= ClimateDataEvaluation.atributions(self,model_cv,X_testing_fold, lat_lims=self.region_atributted[0], lon_lims=self.region_atributted[1], std_y=std_reference_y, test_years=years[testing_index])
                importances_list.append(imp_fold)
                
        # concatenate all the predicted values in the list into one global dataarray
        predicted_global = xr.concat(predicted_list, dim='time')
        correct_value = xr.concat(correct_value_list, dim='year')
        if self.importances:
            importances_global= xr.concat(importances_list, dim='time')
            return predicted_global,correct_value,years_division_list,importances_global
        else:
            return predicted_global,correct_value,years_division_list

    def correlations_pannel(self, n_folds, predicted_global, correct_value,threshold, years_division, outputs_path,months_x, months_y, var_x, var_y, predictor_region, best_model=False, plot_differences=False):
        """
        Visualize correlations for each ensemble member in a panel plot.

        Args:
            - n_folds (int): Number of folds for cross-validation.
            - predicted_global (xarray.DataArray): Concatenated predicted values from all folds.
            - correct_value (xarray.DataArray): Concatenated true values from all folds.
            - months_x (str): Description of the months used for predictions.
            - months_y (str): Description of the months used for true values.
            - titulo_corr (str): Title specifying the correlation type.

        Returns:
            - fig (matplotlib.figure.Figure): Matplotlib figure.
        """

        # Create a list of ensemble members
        years= np.arange(self.time_lims[0], self.time_lims[-1]+1,1)

        # Calculate correlation for each ensemble member
        fig, axes = plt.subplots(nrows=(n_folds // 3 + 1), ncols=3, figsize=(15, 5),
                                subplot_kw={'projection': ccrs.PlateCarree()})

        # Flatten the 2D array of subplots to simplify indexing
        axes = axes.flatten()

        predictions_member = predicted_global
        correct_value = correct_value.rename({'year': 'time'})
        spatial_correlation_global = xr.corr(predicted_global, correct_value, dim='time')
        p_value_global = xs.pearson_r_p_value(predicted_global, correct_value, dim='time')
        acc_clevs = [-1,-0.9,-0.8,-0.7,-0.6,-0.4,-0.2,0.2,0.4,0.6,0.7,0.8,0.9,1]
        colors = np.array([[0,0,255],[0,102,201],[119,153,255],[119,187,255],[170,221,255], [170,255,255],[170,170,170], [255,255,0],
                        [255,204,0], [255,170,0],[255,119,0],[255,0,0],[119,0,34]], np.float32) / 255.0

        acc_map, acc_norm = from_levels_and_colors(acc_clevs, colors)
        
        for i in range(0,n_folds):
            if i < len(axes):  # Only proceed if there are available subplots
                years_fold_div= years_division[i]
                
                predictions_loop = predictions_member.sel(time=slice(years_fold_div[0],years_fold_div[-1]))
                spatial_correlation_member = xr.corr(predictions_loop, correct_value.sel(time=slice(years_fold_div[0],years_fold_div[-1])), dim='time')
                p_value = xs.pearson_r_p_value(predictions_loop, correct_value.sel(time=slice(years_fold_div[0],years_fold_div[-1])), dim='time')
                
                # Plot the correlation map
                ax = axes[i]

                rango=1
                if plot_differences==True:
                    data_member = spatial_correlation_member-spatial_correlation_global
                    im= ClimateDataEvaluation.plotter(self, data=data_member,levs=acc_clevs, cmap1='PiYG_r', l1='Correlation', titulo='Model tested in '+str(i+1)+': '+str(years_fold_div[0])+'-'+str(years_fold_div[-1]), ax=ax, plot_colorbar=False)
                    hatch_mask = p_value < threshold
                    ax.contourf(hatch_mask.longitude, hatch_mask.latitude, hatch_mask, levels=[0, 0.5, 1], hatches=['//', ''], alpha=0, transform=ccrs.PlateCarree())

                else:
                    data_member = spatial_correlation_member
                    im= ClimateDataEvaluation.plotter(self, data=data_member,levs=acc_clevs, cmap1=acc_map, l1='Correlation', titulo='Model tested in '+str(i+1)+': '+str(years_fold_div[0])+'-'+str(years_fold_div[-1]), ax=ax, plot_colorbar=False,acc_norm=acc_norm)
                    hatch_mask = p_value < threshold
                    ax.contourf(hatch_mask.longitude, hatch_mask.latitude, hatch_mask, levels=[0, 0.5, 1], hatches=['//', ''], alpha=0, transform=ccrs.PlateCarree())

                if i==n_folds-1:
                    rango=1
                    # Plot the correlation map
                    ax = axes[i+1]
                    data_member = spatial_correlation_global
                    im2= ClimateDataEvaluation.plotter(self, data=data_member,levs=acc_clevs, cmap1=acc_map, l1='Correlation', titulo='ACC Global', ax=ax, plot_colorbar=False,acc_norm=acc_norm)
                    hatch_mask = p_value_global < threshold
                    ax.contourf(hatch_mask.longitude, hatch_mask.latitude, hatch_mask, levels=[0, 0.5, 1], hatches=['//', ''], alpha=0, transform=ccrs.PlateCarree())

        # remove unused axes
        for ax in axes[n_folds+1:]:
            ax.remove()
    
        # Add a common colorbar for all subplots
        if plot_differences==True:
            cbar_ax = fig.add_axes([0.92, 0.35, 0.02, 0.5])  # Adjust the position for your preference
            cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical', label='Correlation difference: ACC_member-ACC_global', format="%2.1f")
            cbar.set_ticks(ticks=acc_clevs, labels=acc_clevs)

            cbar_ax = fig.add_axes([0.92, 0.025, 0.02, 0.3])  # Adjust the position for your preference
            cbar = fig.colorbar(im2, cax=cbar_ax, orientation='vertical', label='Correlation', format="%2.1f")
            cbar.set_ticks(ticks=acc_clevs, labels=acc_clevs)

        else: 
            cbar_ax = fig.add_axes([0.92, 0.025, 0.02, 0.8])  # Adjust the position for your preference
            cbar = fig.colorbar(im2, cax=cbar_ax, orientation='vertical', label='Correlation', format="%2.1f")
            cbar.set_ticks(ticks=acc_clevs, labels=acc_clevs)

        # Add a common title for the entire figure
        fig.suptitle(f'Correlations for predicting each time period of {var_y} months "{months_y}" \n with months "{months_x}" of {var_x} from {predictor_region}',fontsize=18)

        # Adjust layout for better spacing
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])

        if best_model==True:
            plt.savefig(outputs_path + 'correlations_pannel_best_model.png')
        else:
            plt.savefig(outputs_path + 'correlations_pannel.png')
        return fig

class BestModelAnalysis:
    """
    Class for evaluating a climate prediction model.

    Attributes:
        - input_shape (tuple): Shape of the input data.
        - output_shape (int or tuple): Shape of the output data.
        - X (array): Input data for all samples.
        - X_train (array): Input data for training.
        - X_valid (array): Input data for validation.
        - X_test (array): Input data for testing.
        - Y (array): Output/target data for all samples.
        - Y_train (array): Output/target data for training.
        - Y_valid (array): Output/target data for validation.
        - Y_test (array): Output/target data for testing.
        - lon_y (array): Longitude data for the output.
        - lat_y (array): Latitude data for the output.
        - std_y (array): Standard deviation data for the output.
        - time_lims (tuple): A tuple containing the time limits (start, end) for the selected time period.
        - train_years (list): Years used for training.
        - testing_years (list): Years used for testing.
        - jump_year (int, optional): Time interval, default is zero.
        - params_selection (dictionary): Dictionary containing hyperparameter search space.
        - epochs (int): Number of training epochs.
        - random_seed (int, optional): Seed for random number generation, default is 42.
        - outputs_path (directory): Path to store the outputs.
        - output_original (array): Xarray dataset with the original output dataset with nans.
    
    Methods:
        - build_model(): Build a deep learning model with hyperparameters defined by the Keras Tuner.
        - tuner_searcher(): Perform hyperparameter search using Keras Tuner.
        - bm_evaluation(): Evaluate the best model though different methods.
    """

    def __init__(
        self, input_shape, output_shape, X, X_train,X_valid, X_test, Y, Y_train, Y_valid, Y_test, lon_y, lat_y, std_y, time_lims, train_years, testing_years, params_selection, epochs, 
        outputs_path,output_original, random_seed=42, jump_year=0, threshold=0.1, detrend_x=False, detrend_x_window=10, detrend_y=False, detrend_y_window=10):
        """
        Initialize the BestModelAnalysis class with the specified parameters.

        Args:
            See class attributes for details.
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.X = X
        self.X_train= X_train
        self.X_valid= X_valid
        self.X_test = X_test
        self.Y = Y
        self.Y_train= Y_train
        self.Y_valid= Y_valid
        self.Y_test = Y_test
        self.lon_y = lon_y
        self.lat_y = lat_y
        self.std_y = std_y
        self.time_lims = time_lims
        self.train_years = train_years
        self.testing_years = testing_years
        self.jump_year = jump_year
        self.params_selection = params_selection
        self.epochs = epochs
        self.random_seed = random_seed
        self.outputs_path = outputs_path
        self.output_original = output_original
        self.threshold = threshold
        self.detrend_x = detrend_x
        self.detrend_x_window = detrend_x_window
        self.detrend_y = detrend_y
        self.detrend_y_window = detrend_y_window
        
    def build_model(self, hp):
        """
        Build a deep learning model with hyperparameters defined by the Keras Tuner.

        Args:
            hp (HyperParameters): The hyperparameters object for defining search spaces.
        Returns:
            keras.models.Model: A compiled Keras model.
        """
        
        # Access hyperparameters from params_selection
        pos_number_layers = self.params_selection['pos_number_layers']
        pos_layer_sizes = self.params_selection['pos_layer_sizes']
        pos_activations = self.params_selection['pos_activations']
        pos_dropout = self.params_selection['pos_dropout']
        pos_kernel_regularizer = self.params_selection['pos_kernel_regularizer']
        search_skip_connections = self.params_selection['search_skip_connections']
        pos_conv_layers = self.params_selection['pos_conv_layers']
        pos_learning_rate = self.params_selection['pos_learning_rate']

        # Debugging: Print generated hyperparameter values
        print("Generated hyperparameters:")
        
        # Define the number of layers within the specified range
        num_layers = hp.Int('num_layers', 2, pos_number_layers)
        print("Number of layers:", num_layers)
        
        # Define layer sizes based on the specified choices
        layer_sizes = [hp.Choice('units_' + str(i), pos_layer_sizes) for i in range(num_layers)]
        print("Layer sizes:", layer_sizes)
        
        # Define activations for hidden layers and output layer
        activations = [hp.Choice('activations_' + str(i), pos_activations) for i in range(len(layer_sizes)-1)] + ['linear']
        print("Activations:", activations)
        
        # Choose kernel regularizer
        kernel_regularizer = hp.Choice('kernel_regularizer', pos_kernel_regularizer)
        print("Kernel regularizer:", kernel_regularizer)

        # Choose whether to use batch normalization
        use_batch_norm = hp.Choice('batch_normalization', ['True','False'])
        print("Use batch normalization:", use_batch_norm)

        # Choose whether to use He initialization
        use_initializer = hp.Choice('he_initialization', ['True','False'])
        print("Use He initialization:", use_initializer)

        #Define the learning rate
        learning_rate=hp.Choice('learning_rate', pos_learning_rate)

        # If dropout is chosen, define dropout rate
        dropout_rate = [hp.Choice('dropout', pos_dropout)]
        print("Dropout rate:", dropout_rate)


        # Choose whether to use skip connections
        if search_skip_connections=='True':
            use_initial_skip_connections = hp.Choice('initial_skip_connection', ['True','False'])
            print("Use initial skip connections:", use_initial_skip_connections)
            use_intermediate_skip_connections = hp.Choice('intermediate_skip_connections', ['True','False'])
            print("Use intermediate skip connections:", use_intermediate_skip_connections)
        else:
            use_initial_skip_connections, use_intermediate_skip_connections= False, False
            
        # Define hyperparameters related to convolutional layers
        if pos_conv_layers>0:    
            num_conv_layers = hp.Int('#_convolutional_layers', 0, pos_conv_layers)
            if num_conv_layers>0:
                print("Number of Convolutional Layers:", num_conv_layers)
                num_filters = hp.Choice('number_of_filters_per_conv', [1,4,16])
                print("Number of filters:", num_filters)
                pool_size = hp.Choice('pool size', [2,4])
                print("Pool size:", pool_size)
                kernel_size = hp.Choice('kernel size', [3,6])
                print("kernel size:", kernel_size)
            else:
                num_filters, pool_size, kernel_size= 0,0,0
        else: 
            num_conv_layers,num_filters, pool_size, kernel_size=0,0,0,0

        # Check for empty lists and adjust if needed
        if not layer_sizes:
            layer_sizes = [32]  # Default value if no units are specified

        if not activations:
            activations = ['elu']

        if not dropout_rate:
            dropout_rate = [0.0]

        # Handling the kernel_regularizer choice
        if kernel_regularizer == "l1_l2":
            reg_function = "l1_l2"
        else:
            reg_function = None  # No regularization


        # Set random seeds for reproducibility
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        tf.compat.v1.random.set_random_seed(self.random_seed)

        # Create the model with specified hyperparameters
        neural_network_cv = NeuralNetworkModel(input_shape=self.input_shape, output_shape=self.output_shape, layer_sizes=layer_sizes, activations=activations, dropout_rates=dropout_rate, kernel_regularizer=reg_function,
                                        num_conv_layers=num_conv_layers, num_filters=num_filters, pool_size=pool_size, kernel_size=kernel_size, use_batch_norm=True, use_initializer=True, use_dropout=True, use_initial_skip_connections=False, use_intermediate_skip_connections=False, learning_rate=learning_rate,
                                        epochs=self.epochs)
        model = neural_network_cv.create_model()
        # Define the optimizer with a learning rate hyperparameter
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        # Compile the model with the specified optimizer and loss function
        model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())
        return model
    
    def tuner_searcher(self, max_trials):
        """
        Perform hyperparameter search using Keras Tuner.

        Args:
            max_trials: Maximum number of trials for hyperparameter search.

        Returns:
            tuner: The Keras Tuner instance.
            best_model: The best model found during the search.
            fig: Visualization of the best model architecture.
        """
        # Directory for storing trial information
        trial_dir = self.outputs_path + 'trials_bm_search'

        # Check if the trial directory already exists, if so, it will be deleted
        if os.path.exists(trial_dir):
            shutil.rmtree(trial_dir)

        tuner = RandomSearch(
            lambda hp: BestModelAnalysis.build_model(self,hp=hp),  # Pass the dictionary as an argument
            objective='val_loss',
            max_trials=max_trials,
            executions_per_trial=1, 
            directory=trial_dir)

        tuner.search(np.array(self.X_train), np.array(self.Y_train),
                    epochs=self.epochs,
                    validation_data=(np.array(self.X_valid), np.array(self.Y_valid)),
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)])

        return tuner
    
    def bm_evaluation(self, tuner, units, var_x, var_y, months_x, months_y,predictor_region, n_cv_folds=0, cross_validation=False, threshold=0.1):
        """
        Evaluate the best model.

        Args:
            tuner: The Keras Tuner instance.
            n_cv_folds: Number of folds for cross-validation.
            cross_validation: Flag indicating whether to perform cross-validation.

        Returns:
            predicted_value, observed_value, fig1 (, fig2): Evaluation results and visualization, obtaining two plots if the cross-validation flag is set to True.
        """     

        best_hparams = tuner.oracle.get_best_trials(1)[0].hyperparameters.values
        num_layers = best_hparams['num_layers']
        units_list = [best_hparams[f'units_{i}'] for i in range(num_layers)]
        activations_list = [best_hparams[f'activations_{i}'] for i in range(num_layers-1)]

        kernel_regularizer = best_hparams['kernel_regularizer']
        batch_normalization = best_hparams['batch_normalization']
        he_initialization = best_hparams['he_initialization']
        learning_rate = best_hparams['learning_rate']
        dropout = best_hparams['dropout']
        dropout_list = [best_hparams[f'dropout']]

        if not self.params_selection['pos_conv_layers']==0:
            num_conv_layers = best_hparams['#_convolutional_layers']
            number_of_filters_per_conv = best_hparams['number_of_filters_per_conv']
            pool_size = best_hparams['pool size']
            kernel_size = best_hparams['kernel size']
        else:
            num_conv_layers,number_of_filters_per_conv,pool_size,kernel_size = 0,0,0,0

        print('Now creating and training the best model')
        start_time = time.time()
        neural_network_bm = NeuralNetworkModel(input_shape=self.input_shape, output_shape=self.output_shape, layer_sizes=units_list, activations=activations_list, dropout_rates=dropout_list, kernel_regularizer=kernel_regularizer,
                                            num_conv_layers=num_conv_layers, num_filters=number_of_filters_per_conv, pool_size=pool_size, kernel_size=kernel_size, use_batch_norm=batch_normalization, use_initializer=he_initialization, use_dropout=dropout, 
                                            use_initial_skip_connections=False, use_intermediate_skip_connections=False, learning_rate=learning_rate,epochs=self.epochs)
        model_bm = neural_network_bm.create_model(outputs_path=self.outputs_path, best_model=True)
        if cross_validation==False:
            model_bm, record= neural_network_bm.train_model(self.X_train, self.Y_train, self.X_valid, self.Y_valid, self.outputs_path)
            neural_network_bm.performance_plot(record)
            end_time = time.time()
            time_taken = end_time - start_time
            print(f'Training done (Time taken: {time_taken:.2f} seconds)')
            print('Now evalutating the best model on the test set')
            evaluations_toolkit_bm= ClimateDataEvaluation(self.X, self.X_train, self.X_test, self.Y, self.Y_train, self.Y_test, self.lon_y, self.lat_y, self.std_y, model_bm, self.time_lims, self.train_years, self.testing_years, self.output_original, jump_year=self.jump_year, detrend_x = self.detrend_x,detrend_x_window = self.detrend_x_window,detrend_y = self.detrend_y,detrend_y_window = self.detrend_y_window)
            predicted_value,observed_value= evaluations_toolkit_bm.evaluation()
            fig1= evaluations_toolkit_bm.correlations(predicted_value,observed_value,self.outputs_path, threshold=threshold, units=units, months_x=months_x, months_y=months_y, var_x=var_x, var_y=var_y, predictor_region=predictor_region, best_model=True)
            return predicted_value, observed_value
        else:
            print('Now evalutating the best model via Cross Validation')
            evaluations_toolkit_bm= ClimateDataEvaluation(self.X, self.X_train, self.X_test, self.Y, self.Y_train, self.Y_test, self.lon_y, self.lat_y, self.std_y, model_bm, self.time_lims, self.train_years, self.testing_years, self.output_original, jump_year=self.jump_year, detrend_x = self.detrend_x,detrend_x_window = self.detrend_x_window,detrend_y = self.detrend_y,detrend_y_window = self.detrend_y_window)
            predicted_global,correct_value,years_division_list= evaluations_toolkit_bm.cross_validation(n_folds=n_cv_folds, model_class=neural_network_bm)
            fig2= evaluations_toolkit_bm.correlations(predicted_global,correct_value,self.outputs_path, threshold=threshold, units=units, months_x=months_x, months_y=months_y, var_x=var_x, var_y=var_y, predictor_region=predictor_region, best_model=True)
            fig3= evaluations_toolkit_bm.correlations_pannel(n_folds=n_cv_folds,predicted_global=predicted_global, correct_value=correct_value,years_division=years_division_list,threshold=self.threshold, outputs_path= self.outputs_path, months_x=months_x, months_y=months_y, predictor_region=predictor_region,var_x=var_x, var_y=var_y, best_model=True)
            return predicted_global, correct_value

def Dictionary_saver(dictionary):
    # your_dictionary is the dictionary you want to save
    output_file_path = dictionary['outputs_path']+'dict_hyperparms.yaml'

    # Check if the file already exists
    print('Checking if the file already exists in the current directory...')
    if os.path.isfile(output_file_path):
        overwrite_confirmation = input(f'The file {output_file_path} already exists. Do you want to overwrite it? (yes/no): ')
        if overwrite_confirmation.lower() != 'yes':
            print('Operation aborted. The existing file was not overwritten.')
            # Add further handling or exit the script if needed

    with open(output_file_path, 'w') as yaml_file:
        yaml.dump(dictionary, yaml_file, default_flow_style=False)

    print(f'Dictionary saved to {output_file_path}')

def Preprocess(dictionary_hyperparams):
    print('Preprocessing the data')
    start_time = time.time()

    data_mining_x = ClimateDataPreprocessing(relative_path=dictionary_hyperparams['path']+dictionary_hyperparams['path_x'],lat_lims=dictionary_hyperparams['lat_lims_x'],lon_lims=dictionary_hyperparams['lon_lims_x'],
        time_lims=dictionary_hyperparams['time_lims'],scale=dictionary_hyperparams['scale_x'],regrid_degree=dictionary_hyperparams['regrid_degree_x'],variable_name=dictionary_hyperparams['name_x'],
        months=dictionary_hyperparams['months_x'],months_to_drop=dictionary_hyperparams['months_skip_x'], years_out=dictionary_hyperparams['years_finally'], 
        detrend=dictionary_hyperparams['detrend_x'],detrend_window=dictionary_hyperparams['detrend_x_window'],mean_seasonal_method=dictionary_hyperparams['mean_seasonal_method_x'],train_years=dictionary_hyperparams['train_years'])
    
    data_mining_y = ClimateDataPreprocessing(relative_path=dictionary_hyperparams['path']+dictionary_hyperparams['path_y'],lat_lims=dictionary_hyperparams['lat_lims_y'],lon_lims=dictionary_hyperparams['lon_lims_y'],
        time_lims=dictionary_hyperparams['time_lims'],scale=dictionary_hyperparams['scale_y'],regrid_degree=dictionary_hyperparams['regrid_degree_y'],variable_name=dictionary_hyperparams['name_y'],
        months=dictionary_hyperparams['months_y'],months_to_drop=dictionary_hyperparams['months_skip_y'], years_out=dictionary_hyperparams['years_finally'], 
        detrend=dictionary_hyperparams['detrend_y'],detrend_window=dictionary_hyperparams['detrend_y_window'], jump_year= dictionary_hyperparams['jump_year'],mean_seasonal_method=dictionary_hyperparams['mean_seasonal_method_y'],train_years=dictionary_hyperparams['train_years'])
        
    # Preprocess data
    lat_x,lon_x, data_x, anom_x,norm_x,mean_x,std_x = data_mining_x.preprocess_data()
    lat_y,lon_y, data_y, anom_y,norm_y,mean_y,std_y = data_mining_y.preprocess_data()

    data_splitter = DataSplitter(train_years=dictionary_hyperparams['train_years'], validation_years=dictionary_hyperparams['validation_years'], testing_years=dictionary_hyperparams['testing_years'], 
        predictor=norm_x, predictant=norm_y, jump_year=dictionary_hyperparams['jump_year'])
    X, X_train, X_valid, X_test, Y, Y_train, Y_valid, Y_test, input_shape,output_shape = data_splitter.prepare_data()

    end_time = time.time()
    time_taken = end_time - start_time
    print(f'Preprocessing done (Time taken: {time_taken:.2f} seconds)')
    preprocessing_results = {'input': {'lat': lat_x,'lon': lon_x,'data': data_x,'anomaly': anom_x,'normalized': norm_x,'mean': mean_x,'std': std_x,},
        'output': {'lat': lat_y,'lon': lon_y,'data': data_y,'anomaly': anom_y,'normalized': norm_y,'mean': mean_y,'std': std_y,},
        'data_split': {'X': X,'X_train': X_train,'X_valid': X_valid,'X_test': X_test,'Y': Y,'Y_train': Y_train,'Y_valid': Y_valid,'Y_test': Y_test,'input_shape': input_shape,'output_shape': output_shape,}}
    return preprocessing_results

def Model_searcher(dictionary_hyperparams, dictionary_preprocess, dictionary_possibilities, max_trials=10, n_cv_folds=12):
    output_directory = os.path.join(dictionary_hyperparams['outputs_path'], 'best_model/')

    bm_class= BestModelAnalysis(dictionary_preprocess['data_split']['input_shape'], dictionary_preprocess['data_split']['output_shape'], dictionary_preprocess['input']['data'], dictionary_preprocess['data_split']['X_train'],dictionary_preprocess['data_split']['X_valid'], dictionary_preprocess['data_split']['X_test'], dictionary_preprocess['output']['data'], dictionary_preprocess['data_split']['Y_train'], dictionary_preprocess['data_split']['Y_valid'], dictionary_preprocess['data_split']['Y_test'], 
        dictionary_preprocess['output']['lon'], dictionary_preprocess['output']['lat'], dictionary_preprocess['output']['std'], dictionary_hyperparams['time_lims'],  dictionary_hyperparams['train_years'], dictionary_hyperparams['testing_years'], dictionary_possibilities, dictionary_hyperparams['epochs'], output_directory, dictionary_preprocess['output']['normalized'], jump_year=dictionary_hyperparams['jump_year'], threshold=dictionary_hyperparams['p_value'])
    tuner = bm_class.tuner_searcher(max_trials=max_trials)

    predicted_value, observed_value= bm_class.bm_evaluation(tuner, cross_validation=False, threshold=dictionary_hyperparams['p_value'], units=dictionary_hyperparams['units_y'], var_x=dictionary_hyperparams['name_x'], var_y=dictionary_hyperparams['name_y'], months_x=dictionary_hyperparams['months_x'], months_y=dictionary_hyperparams['months_y'], predictor_region=dictionary_hyperparams['region_predictor'],)
    predicted_global, observed_global= bm_class.bm_evaluation(tuner, n_cv_folds=n_cv_folds, cross_validation=True, threshold=dictionary_hyperparams['p_value'], units=dictionary_hyperparams['units_y'], var_x=dictionary_hyperparams['name_x'], var_y=dictionary_hyperparams['name_y'], months_x=dictionary_hyperparams['months_x'], months_y=dictionary_hyperparams['months_y'], predictor_region=dictionary_hyperparams['region_predictor'],)
    datasets, names = [predicted_value, observed_value, predicted_global,observed_global], ['predicted_test_period_bm.nc', 'observed_test_period_bm.nc','predicted_global_cv_bm.nc', 'observed_global_cv_bm.nc']
    variable_names = ['predicted_value', 'observed_value', 'predicted_global', 'observed_global']
    os.makedirs(output_directory, exist_ok=True)
    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.save(output_directory+'best_model.h5')  # Save the model in HDF5 format
    # Save the hyperparameters
    with open(os.path.join(output_directory, 'hyperparameters_bm.txt'), 'w') as f:
        f.write(str(tuner.get_best_hyperparameters()[0].values))
    # Save each dataset to a NetCDF file in the 'data_outputs' folder
    for i, ds in enumerate(datasets, start=1):
        ds.to_netcdf(os.path.join(output_directory, names[i-1]), format='NETCDF4', mode='w', group='/', engine='netcdf4')
    bm_model_outputs_cv = {'predictions':predicted_global, 'observations':observed_global}
    return bm_model_outputs_cv
    
def Model_build_and_test(dictionary_hyperparams, dictionary_preprocess, cross_validation=False, n_cv_folds=0, plot_differences=False, importances=False,region_importances=None):
    print('Now creating and training the model')
    start_time = time.time()
    neural_network = NeuralNetworkModel(input_shape=dictionary_preprocess['data_split']['input_shape'], output_shape=dictionary_preprocess['data_split']['output_shape'], layer_sizes=dictionary_hyperparams['layer_sizes'], activations=dictionary_hyperparams['activations'], dropout_rates=dictionary_hyperparams['dropout_rates'], kernel_regularizer=dictionary_hyperparams['kernel_regularizer'],
                                        num_conv_layers=dictionary_hyperparams['num_conv_layers'],use_batch_norm=dictionary_hyperparams['use_batch_norm'], use_initializer=dictionary_hyperparams['use_initializer'], use_dropout=dictionary_hyperparams['use_dropout'], use_initial_skip_connections=dictionary_hyperparams['use_initial_skip_connections'], use_intermediate_skip_connections=dictionary_hyperparams['use_intermediate_skip_connections'], learning_rate=dictionary_hyperparams['learning_rate'],
                                        epochs=dictionary_hyperparams['epochs'])

    model = neural_network.create_model(outputs_path=dictionary_hyperparams['outputs_path'])
    model, record= neural_network.train_model(dictionary_preprocess['data_split']['X_train'], dictionary_preprocess['data_split']['Y_train'], dictionary_preprocess['data_split']['X_valid'], dictionary_preprocess['data_split']['Y_valid'], dictionary_hyperparams['outputs_path'])
    evaluations_toolkit= ClimateDataEvaluation(dictionary_preprocess['input']['data'], dictionary_preprocess['data_split']['X_train'], dictionary_preprocess['data_split']['X_test'], dictionary_preprocess['output']['data'], dictionary_preprocess['data_split']['Y_train'], dictionary_preprocess['data_split']['Y_test'], 
        dictionary_preprocess['output']['lon'], dictionary_preprocess['output']['lat'], dictionary_preprocess['output']['std'], model, dictionary_hyperparams['time_lims'],  dictionary_hyperparams['train_years'], dictionary_hyperparams['testing_years'],dictionary_preprocess['output']['normalized'], jump_year=dictionary_hyperparams['jump_year'],detrend_x=dictionary_hyperparams['detrend_x'],detrend_x_window=dictionary_hyperparams['detrend_x_window'],detrend_y=dictionary_hyperparams['detrend_y'],detrend_y_window=dictionary_hyperparams['detrend_y_window'], importances=importances, region_atributted=region_importances)
    
    output_directory = os.path.join(dictionary_hyperparams['outputs_path'], 'data_outputs')
    os.makedirs(output_directory, exist_ok=True)
    (dictionary_preprocess['input']['anomaly']).to_netcdf(os.path.join(output_directory,'predictor_anomalies.nc'), format='NETCDF4', mode='w', group='/', engine='netcdf4')
    
    if cross_validation==False:
        neural_network.performance_plot(record)
        predicted_value,correct_value= evaluations_toolkit.evaluation()        
        fig1= evaluations_toolkit.correlations(predicted_value,correct_value,outputs_path=dictionary_hyperparams['outputs_path'], threshold=dictionary_hyperparams['p_value'], units=dictionary_hyperparams['units_y'], var_x=dictionary_hyperparams['name_x'], var_y=dictionary_hyperparams['name_y'], months_x=dictionary_hyperparams['months_x'], months_y=dictionary_hyperparams['months_y'], predictor_region=dictionary_hyperparams['region_predictor'], best_model=False)
        datasets, names = [predicted_value, correct_value], ['predicted_test_period.nc', 'observed_test_period.nc']
        variable_names = ['predicted_value', 'observed_value']
        # Save each dataset to a NetCDF file in the 'data_outputs' folder
        for i, ds in enumerate(datasets, start=1):
            ds.to_netcdf(os.path.join(output_directory, names[i-1]), format='NETCDF4', mode='w', group='/', engine='netcdf4')

    else:
        if importances:
            predicted_value,correct_value,years_division_list, importances_region= evaluations_toolkit.cross_validation(n_folds=n_cv_folds, model_class=neural_network)
        else:
            predicted_value,correct_value,years_division_list= evaluations_toolkit.cross_validation(n_folds=n_cv_folds, model_class=neural_network)
        
        fig1= evaluations_toolkit.correlations(predicted_value,correct_value,outputs_path= dictionary_hyperparams['outputs_path'], threshold=dictionary_hyperparams['p_value'], units=dictionary_hyperparams['units_y'], var_x=dictionary_hyperparams['name_x'], var_y=dictionary_hyperparams['name_y'], months_x=dictionary_hyperparams['months_x'], months_y=dictionary_hyperparams['months_y'], predictor_region=dictionary_hyperparams['region_predictor'], best_model=False)
        fig2= evaluations_toolkit.correlations_pannel(n_folds=n_cv_folds,predicted_global=predicted_value, correct_value=correct_value,years_division=years_division_list, threshold=dictionary_hyperparams['p_value'],outputs_path= dictionary_hyperparams['outputs_path'], months_x=dictionary_hyperparams['months_x'], months_y=dictionary_hyperparams['months_y'], predictor_region=dictionary_hyperparams['region_predictor'],var_x=dictionary_hyperparams['name_x'],var_y=dictionary_hyperparams['name_y'], best_model=False, plot_differences=plot_differences)
        if importances:
            datasets, names = [predicted_value, correct_value, importances_region], ['predicted_global_cv.nc', 'observed_global_cv.nc', 'importances_region_cv.nc']
        else:
            datasets, names = [predicted_value, correct_value], ['predicted_global_cv.nc', 'observed_global_cv.nc']

        # Save each dataset to a NetCDF file in the 'data_outputs' folder
        for i, ds in enumerate(datasets, start=1):
            ds.to_netcdf(os.path.join(output_directory, names[i-1]), format='NETCDF4', mode='w', group='/', engine='netcdf4')

    end_time = time.time()
    time_taken = end_time - start_time
    print(f'Training done (Time taken: {time_taken:.2f} seconds)')
    if importances:
        model_outputs = {'predictions': predicted_value,'observations':correct_value,'importances':importances_region, 'region_attributed':region_importances}
    else:
        model_outputs = {'predictions': predicted_value,'observations':correct_value}
    return model_outputs

def Results_plotter(hyperparameters, dictionary_preprocess, rang_x, rang_y, predictions, observations, years_to_plot=None, plot_with_contours=False, importances=None,region_importances=None, rang_atr=None):
    evaluations_toolkit_input= ClimateDataEvaluation(dictionary_preprocess['data_split']['X'], dictionary_preprocess['data_split']['X_train'], dictionary_preprocess['data_split']['X_test'], dictionary_preprocess['data_split']['Y'], dictionary_preprocess['data_split']['Y_train'], dictionary_preprocess['data_split']['Y_test'], 
            dictionary_preprocess['input']['lon'], dictionary_preprocess['input']['lat'], dictionary_preprocess['output']['std'], None, hyperparameters['time_lims'],  hyperparameters['train_years'], hyperparameters['testing_years'],dictionary_preprocess['output']['normalized'], jump_year=hyperparameters['jump_year'])
    evaluations_toolkit_output= ClimateDataEvaluation(dictionary_preprocess['data_split']['X'], dictionary_preprocess['data_split']['X_train'], dictionary_preprocess['data_split']['X_test'], dictionary_preprocess['data_split']['Y'], dictionary_preprocess['data_split']['Y_train'], dictionary_preprocess['data_split']['Y_test'], 
            dictionary_preprocess['output']['lon'], dictionary_preprocess['output']['lat'], dictionary_preprocess['output']['std'], None, hyperparameters['time_lims'],  hyperparameters['train_years'], hyperparameters['testing_years'],dictionary_preprocess['output']['normalized'], jump_year=hyperparameters['jump_year'])
    
    if years_to_plot:
        plotting_years= years_to_plot
    else:
        plotting_years= np.arange(hyperparameters['years_finally'][0],hyperparameters['years_finally'][-1]+1,1)
    
    output_directory = os.path.join(hyperparameters['outputs_path'], 'individual_predictions')
    os.makedirs(output_directory, exist_ok=True)
    for i in plotting_years:
        fig = plt.figure(figsize=(15,5))
        data_output_pred, data_output_obs= predictions.sel(time=i+hyperparameters['jump_year']), observations.sel(year=i+hyperparameters['jump_year'])
        
        if plot_with_contours==True:
            ax = fig.add_subplot(121, projection=ccrs.PlateCarree(central_longitude=-180))
            ax2 = fig.add_subplot(122, projection=ccrs.PlateCarree())
            im2= evaluations_toolkit_output.plotter(np.array(data_output_pred), np.arange(-rang_y, rang_y, rang_y/10), 'RdBu_r',f'Anomalies {hyperparameters["units_y"]}', '', ax2, pixel_style=False, plot_colorbar=False, extend='both')
            im3= ax2.contour(data_output_obs.longitude,data_output_obs.latitude,data_output_obs,colors='black',levels=np.arange(-rang_y, rang_y, rang_y/5),extend='both',transform=ccrs.PlateCarree())
            ax2.clabel(im3, inline=True, fontsize=10, fmt="%1.1f")
            ax2.set_title(f"{hyperparameters['name_y']} of months '{hyperparameters['months_y']}' from year {str(i+hyperparameters['jump_year'])}. Pred=colours and Obs=lines",fontsize=10)

        else:
            ax = fig.add_subplot(131, projection=ccrs.PlateCarree(central_longitude=-180))
            ax2 = fig.add_subplot(132, projection=ccrs.PlateCarree())
            ax3 = fig.add_subplot(133, projection=ccrs.PlateCarree())
            im2= evaluations_toolkit_output.plotter(np.array(data_output_pred), np.arange(-rang_y, rang_y, rang_y/10), 'RdBu_r',f'Anomalies {hyperparameters["units_y"]}', '', ax2, pixel_style=False, plot_colorbar=False, extend='both')
            im3= evaluations_toolkit_output.plotter(np.array(data_output_obs), np.arange(-rang_y, rang_y, rang_y/10), 'RdBu_r',f'Anomalies {hyperparameters["units_y"]}', '', ax3, pixel_style=False, plot_colorbar=False, extend='both')
            ax2.set_title(f"Predictions for {hyperparameters['name_y']} of months '{hyperparameters['months_y']}' from year {str(i+hyperparameters['jump_year'])}",fontsize=10)
            ax3.set_title(f"Observations for {hyperparameters['name_y']} of months '{hyperparameters['months_y']}' from year {str(i+hyperparameters['jump_year'])}",fontsize=10)
            if rang_y>100:
                cbar3 = plt.colorbar(im3, extend='neither', spacing='proportional',orientation='horizontal', shrink=0.9, format="%1.1f")
                tick_values = cbar3.get_ticks() 
                cbar3.set_ticks(tick_values)
                tick_labels = [f'{val/rang_y:.1f}' for val in tick_values]
                cbar3.ax.set_xticklabels(tick_labels)
                cbar3.set_label(f'*{rang_y:1.1e} {hyperparameters["units_y"]}', size=10)
            else:
                cbar3 = plt.colorbar(im3, extend='neither', spacing='proportional',orientation='horizontal', shrink=0.9, format="%2.1f")
                cbar3.set_label(f'{hyperparameters["units_y"]}', size=10)
                
            cbar3.ax.tick_params(labelsize=10)
            
        data_input= dictionary_preprocess['input']['anomaly'].sel(year=i)
        if importances is not None:
            rang_imp = np.max(np.abs(importances))/5
            if rang_atr:
                rang_imp = rang_atr
            im= evaluations_toolkit_input.plotter(np.array(importances.sel(time=i)), np.arange(-rang_imp, rang_imp, rang_imp/10), 'RdBu_r',f'Importances {hyperparameters["units_y"]}', '', ax, pixel_style=True, plot_colorbar=False, extend='both')
            im3= ax.contour(data_input.longitude,data_input.latitude,data_input,colors='black',levels=np.arange(-rang_x, rang_x, rang_x/5),extend='both',transform=ccrs.PlateCarree())
            ax.clabel(im3, inline=True, fontsize=10, fmt="%1.1f")
            ax.set_title(f"Importances and {hyperparameters['name_x']} of months '{hyperparameters['months_x']}' from year {str(i)}",fontsize=10)
            rect = mpatches.Rectangle((region_importances[1][0], region_importances[0][1]), region_importances[1][1] - region_importances[1][0],region_importances[0][0] - region_importances[0][1], linewidth=2, edgecolor='green', facecolor='none',fill=True, transform=ccrs.PlateCarree(), label='Importances region')
            ax2.add_patch(rect)
            cbar1 = plt.colorbar(im, extend='neither', spacing='proportional',orientation='horizontal', shrink=1, format="%1.2f")
            cbar1.set_label(f'Importances {hyperparameters["units_y"]}', size=10)

        else:
            im= evaluations_toolkit_input.plotter(np.array(data_input), np.arange(-rang_x, rang_x, rang_x/10), 'RdBu_r',f'Anomalies {hyperparameters["units_x"]}', '', ax, pixel_style=False, plot_colorbar=False, extend='both')
            ax.set_title(f"{hyperparameters['name_x']} of months '{hyperparameters['months_x']}' from year {str(i)}",fontsize=10)
            if rang_x>100:
                cbar1 = plt.colorbar(im, extend='neither', spacing='proportional',orientation='horizontal', shrink=0.9, format="%1.1f")
                tick_values = cbar1.get_ticks() 
                cbar1.set_ticks(tick_values)
                tick_labels = [f'{val/rang_x:.1f}' for val in tick_values]
                cbar1.ax.set_xticklabels(tick_labels)
                cbar1.set_label(f'*{rang_x:1.1e} {hyperparameters["units_x"]}', size=10)
            else:
                cbar1 = plt.colorbar(im, extend='neither', spacing='proportional',orientation='horizontal', shrink=0.9, format="%2.1f")
                cbar1.set_label(f'{hyperparameters["units_x"]}', size=10)

        if rang_y>100:
            cbar2 = plt.colorbar(im2, extend='neither', spacing='proportional',orientation='horizontal', shrink=0.9, format="%1.1f")
            tick_values = cbar2.get_ticks() 
            cbar2.set_ticks(tick_values)
            tick_labels = [f'{val/rang_y:.1f}' for val in tick_values]
            cbar2.ax.set_xticklabels(tick_labels)
            cbar2.set_label(f'*{rang_y:1.1e} {hyperparameters["units_y"]}', size=10)
            
        else:
            cbar2 = plt.colorbar(im2, extend='neither', spacing='proportional',orientation='horizontal', shrink=0.9, format="%2.1f")
            cbar2.set_label(f'{hyperparameters["units_y"]}', size=10)

        cbar1.ax.tick_params(labelsize=10)        
        cbar2.ax.tick_params(labelsize=10)
        
        plt.savefig(output_directory + f"/prediction_evaluation_for_year_{str(i+hyperparameters['jump_year'])}.png")
    return

def PC_analysis(hyperparameters, prediction, observation, n_modes, n_clusters, cmap='RdBu_r'):
    if 'year' not in observation.coords:
        observation = observation.rename({'time': 'year'})
    if 'year' not in prediction.coords:
        prediction = prediction.rename({'time': 'year'})

    def quitonans(mat):
            out = mat[:,~np.isnan(mat.mean(axis = 0))]
            return out

    def pongonans(matred,mat):
        out = mat.mean(axis = 0 )
        #out= np.array(mat)
        out[:] = np.nan
        #out[~np.isnan(np.array(mat))] = matred
        out[~np.isnan(mat.mean(axis = 0))] = matred
        return out
        return out
    
    def eofs(data, n_modes):
        nt, nlat, nlon= data.shape
        map_orig= data[:,:,:]
        map_reshape= np.reshape(np.array(map_orig),(nt, nlat*nlon))
        lon, lat= data.longitude, data.latitude
        years= data.year

        data= np.reshape(np.array(data), (nt, nlat*nlon))
        data_sin_nan= quitonans(data)
        pca = PCA(n_components=n_modes, whiten=True)
        pca.fit(data_sin_nan)
        print('Explained variance ratio:', pca.explained_variance_ratio_[0:n_modes])
        eofs= pca.components_[0:n_modes]
        pcs= np.transpose(np.dot(eofs,np.transpose(data_sin_nan)))
        pcs= np.transpose((pcs-np.mean(pcs,axis=0))/np.std(pcs, axis=0))
        eofs_reg= np.dot(pcs,data_sin_nan)/(nt-1)
        return eofs_reg, map_reshape, lat, lon, nlat, nlon, years, pca.explained_variance_ratio_[0:n_modes], pcs

    def clustering(n_clusters, pcs,eofs, n_modes):
        pc_pred= np.array(pcs)

        # Apply K-means clustering to the principal components
        kmeans_pred = KMeans(n_clusters=n_clusters).fit(pcs)

        # Get cluster labels
        cluster_labels_pred = kmeans_pred.labels_

        cluster_centers = pd.DataFrame(
            kmeans_pred.cluster_centers_, 
            columns=[f'eof{i}' for i in np.arange(1,n_modes+1)]
            )

        cluster_center_array = xr.DataArray(
            cluster_centers.values, 
            coords=[np.arange(1, n_clusters+1), np.arange(1, n_modes+1) ], 
            dims=['centroids', 'mode'])

        nm, nlat, nlon= eofs.shape
        eofs_reshaped= np.reshape(eofs, (nm, nlat*nlon))
        clusters_reshaped= np.dot(np.array(cluster_center_array),eofs_reshaped)
        clusters= np.reshape(clusters_reshaped, (n_clusters, nlat, nlon))

        unique_values, counts = np.unique(np.array(cluster_labels_pred), return_counts=True)
        percentages= (counts / len(np.array(cluster_labels_pred))) * 100
        
        # Sort clusters based on percentages
        sorted_indices = np.argsort(percentages)[::-1]
        sorted_clusters = np.array([clusters[i] for i in sorted_indices])
        sorted_percentages = np.array([percentages[i] for i in sorted_indices])
        return sorted_clusters, sorted_percentages

    eofs_reg_pred, map_reshape_pred, lat_pred, lon_pred, nlat_pred, nlon_pred, years, explained_variance_pred, pcs_pred= eofs(prediction,n_modes)
    eofs_reg_obs, map_reshape_obs, lat_obs, lon_obs, nlat_obs, nlon_obs, years, explained_variance_obs, pcs_obs= eofs(observation,n_modes)
    eofs_pred_list, eofs_obs_list= [], []

    for i in range(0,n_modes):
        eof_pred= pongonans(eofs_reg_pred[i,:],np.array(map_reshape_pred))
        eof_pred= np.reshape(eof_pred, (nlat_pred, nlon_pred))
        eofs_pred_list.append(eof_pred)
        eof_obs= pongonans(eofs_reg_obs[i,:],np.array(map_reshape_obs))
        eof_obs= np.reshape(eof_obs, (nlat_obs, nlon_obs))
        eofs_obs_list.append(eof_obs)
        
        # Create a new figure
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 3))
        # Subplot 1: Spatial Correlation Map
        ax = fig.add_subplot(141, projection=ccrs.PlateCarree())
        rango = max((abs(np.nanmin(np.array(eof_pred))),abs(np.nanmax(np.array(eof_pred)))))
        num_levels = 20
        levels = np.linspace(-rango, rango, num_levels)
        # Create a pixel-based colormap plot on the given axis
        im = ax.contourf(lon_pred, lat_pred, eof_pred, cmap=cmap, transform=ccrs.PlateCarree(), levels=levels)  # 'norm' defines custom levels
        ax.coastlines(linewidth=0.75)
        cbar = plt.colorbar(im, extend='neither', spacing='proportional', orientation='horizontal', shrink=0.9, format="%2.1f")
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label(f'{hyperparameters["units_y"]}/std', size=15)
        gl = ax.gridlines(draw_labels=True)
        gl.xlines = False
        gl.ylines = False
        gl.top_labels = False  # Disable top latitude labels
        gl.right_labels = False  # Disable right longitude labels

        ax = fig.add_subplot(142)
        ax.plot(np.array(years),pcs_pred[i,:])
        ax.grid()
        
        ax = fig.add_subplot(143, projection=ccrs.PlateCarree())
        rango = max((abs(np.nanmin(np.array(eof_obs))),abs(np.nanmax(np.array(eof_obs)))))
        num_levels = 20
        levels = np.linspace(-rango, rango, num_levels)
        # Create a pixel-based colormap plot on the given axis
        im = ax.contourf(lon_obs, lat_obs, eof_obs, cmap=cmap, transform=ccrs.PlateCarree(), levels=levels)  # 'norm' defines custom levels
        ax.coastlines(linewidth=0.75)
        cbar = plt.colorbar(im, extend='neither', spacing='proportional', orientation='horizontal', shrink=0.9, format="%2.1f")
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label(f'{hyperparameters["units_y"]}/std', size=15)
        gl = ax.gridlines(draw_labels=True)
        gl.xlines = False
        gl.ylines = False
        gl.top_labels = False  # Disable top latitude labels
        gl.right_labels = False  # Disable right longitude labels

        ax = fig.add_subplot(144)
        ax.plot(np.array(years),pcs_obs[i,:])
        ax.grid()
        fig.suptitle(f'EOF and PC {hyperparameters["name_y"]} from months "{hyperparameters["months_y"]}"  for mode {i+1} with variance ratio predicted: {explained_variance_pred[i]*100:.2f} % and observed: {explained_variance_obs[i]*100:.2f} % ', fontsize=15)
        output_directory = os.path.join(hyperparameters['outputs_path'], 'PC_analysis')
        os.makedirs(output_directory, exist_ok=True)
        plt.savefig(output_directory + f'/EOF and PC {hyperparameters["name_y"]} from months "{hyperparameters["months_y"]}"  for mode {i+1}.png')
    
    #we pass the pcs to xarray dataset
    pcs_pred = xr.DataArray(
        data=np.transpose(pcs_pred),
        dims=["time","pc"],
        coords=dict(pc=np.arange(1,n_modes+1),
            time=np.array(years)))

    pcs_obs = xr.DataArray(
        data=np.transpose(pcs_obs),
        dims=["time","pc"],
        coords=dict(pc=np.arange(1,n_modes+1),
            time=np.array(years)))
    
    eofs_pred = xr.DataArray(
        data=np.array(eofs_pred_list),
        dims=["pc","latitude","longitude"],
        coords=dict(pc=np.arange(1,n_modes+1),
            latitude=np.array(observation.latitude), longitude=np.array(prediction.longitude)))

    eofs_obs = xr.DataArray(
        data=np.array(eofs_obs_list),
        dims=["pc","latitude","longitude"],
        coords=dict(pc=np.arange(1,n_modes+1),
            latitude=np.array(observation.latitude), longitude=np.array(prediction.longitude)))
    
    if n_clusters>0:
        clusters_pred, percent_pred = clustering(n_clusters, np.array(pcs_pred), np.array(eofs_pred_list), n_modes)
        clusters_obs, percent_obs = clustering(n_clusters, np.array(pcs_obs), np.array(eofs_obs_list), n_modes)

        for i in range(0,n_clusters):
            # Create a new figure
            plt.style.use('default')
            fig = plt.figure(figsize=(20, 3))
            # Subplot 1: Spatial Correlation Map
            ax = fig.add_subplot(121, projection=ccrs.PlateCarree())
            rango1 = max((abs(np.nanmin(np.array(clusters_pred))),abs(np.nanmax(np.array(clusters_pred)))))
            rango2 = max((abs(np.nanmin(np.array(clusters_obs))),abs(np.nanmax(np.array(clusters_obs)))))
            rango = max(rango1, rango2)
            num_levels = 20
            levels = np.linspace(-rango, rango, num_levels)
            # Create a pixel-based colormap plot on the given axis
            im = ax.contourf(lon_pred, lat_pred, clusters_pred[i,:,:], cmap=cmap, transform=ccrs.PlateCarree(), levels=levels)  # 'norm' defines custom levels
            ax.coastlines(linewidth=0.75)
            cbar = plt.colorbar(im, extend='neither', spacing='proportional', orientation='horizontal', shrink=0.9, format="%2.1f")
            cbar.ax.tick_params(labelsize=7)
            cbar.set_label(f'{hyperparameters["units_y"]}', size=15)
            gl = ax.gridlines(draw_labels=True)
            gl.xlines = False
            gl.ylines = False
            gl.top_labels = False  # Disable top latitude labels
            gl.right_labels = False  # Disable right longitude labels
            
            ax = fig.add_subplot(122, projection=ccrs.PlateCarree())
            num_levels = 20
            levels = np.linspace(-rango, rango, num_levels)
            # Create a pixel-based colormap plot on the given axis
            im = ax.contourf(lon_pred, lat_pred, clusters_obs[i,:,:], cmap=cmap, transform=ccrs.PlateCarree(), levels=levels)  # 'norm' defines custom levels
            ax.coastlines(linewidth=0.75)
            cbar = plt.colorbar(im, extend='neither', spacing='proportional', orientation='horizontal', shrink=0.9, format="%2.1f")
            cbar.ax.tick_params(labelsize=7)
            cbar.set_label(f'{hyperparameters["units_y"]}', size=15)
            gl = ax.gridlines(draw_labels=True)
            gl.xlines = False
            gl.ylines = False
            gl.top_labels = False  # Disable top latitude labels
            gl.right_labels = False  # Disable right longitude labels
            fig.suptitle(f'Weather regimes for {hyperparameters["name_y"]} from months "{hyperparameters["months_y"]}" for cluster {i+1} with occurrence predicted: {percent_pred[i]:.2f} % and observed: {percent_obs[i]:.2f} % ', fontsize=15)
            plt.savefig(output_directory + f'/Weather regimes for {hyperparameters["name_y"]} from months "{hyperparameters["months_y"]}" for cluster {i+1}.png')
    else:
        clusters_pred, clusters_obs= None, None
    

    datasets, names = [pcs_pred, pcs_obs, eofs_pred, eofs_obs], ['pcs_predicted.nc', 'pcs_observed.nc', 'eofs_pred.nc', 'eofs_obs.nc']
    # Save each dataset to a NetCDF file in the 'data_outputs' folder
    for i, ds in enumerate(datasets, start=1):
        ds.to_netcdf(os.path.join(output_directory, names[i-1]), format='NETCDF4', mode='w', group='/', engine='netcdf4')
    
    eof_analysis = {'pcs_predicted':pcs_pred, 'eofs_predicted':np.array(eofs_pred_list), 'pcs_observed':pcs_obs, 'eofs_observed':np.array(eofs_obs_list), 'clusters_pred':clusters_pred, 'clusters_obs':clusters_obs}       
    return eof_analysis
