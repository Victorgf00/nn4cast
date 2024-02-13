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

#The following two lines are coded to avoid the warning unharmful message.
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
plt.style.use('seaborn-v0_8')



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
        reference_period (list): Reference period for normalization [start_year, end_year].
        detrend (bool, optional): Detrend the data using linear regression. Default is False.
        one_output (bool, optional): Return one output for each year. Default is False.
        mean_seasonal_method(bool): Decides how to compute the seasonal aggregates. True if mean, False if sum (default True)
        
    Methods:
        preprocess_data(): Preprocess the climate data based on the specified parameters.
    """

    def __init__(
        self, relative_path, lat_lims, lon_lims, time_lims, scale=1,regrid_degree=1, overlapping=False, variable_name=None, rename=False, latitude_regrid=False,
        months=None, months_to_drop=None, years_out=None, reference_period=None, detrend=False, one_output=False, jump_year=0, mean_seasonal_method=True, signal_filtering=False, cut_off=10, filter_type='high'):
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
        self.overlapping = overlapping
        self.variable_name = variable_name
        self.rename = rename
        self.latitude_regrid = latitude_regrid
        self.months = months
        self.months_to_drop = months_to_drop
        self.years_out = years_out
        self.reference_period = reference_period
        self.detrend = detrend
        self.one_output = one_output
        self.jump_year = jump_year
        self.mean_seasonal_method = mean_seasonal_method
        self.signal_filtering = signal_filtering
        self.cut_off = cut_off
        self.filter_type = filter_type
        
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
        data = xr.open_dataset(self.relative_path) / self.scale
        time = data['time'].astype('datetime64[M]')
        data = data.assign_coords(time=time)

        if 'latitude' not in data.coords or 'longitude' not in data.coords:
            data = data.rename({'lat': 'latitude', 'lon': 'longitude'})

        data = data.sortby('latitude', ascending=False)
        
        if self.lon_lims[1] > 180:
            data = data.assign_coords(longitude=np.where(data.longitude < 0, 360 + data.longitude, data.longitude)).sortby('longitude')
        else:
            data = data.assign_coords(longitude=(((data.longitude + 180) % 360) - 180)).sortby('longitude')

        data = data.sel(latitude=slice(self.lat_lims[0], self.lat_lims[1]), longitude=slice(self.lon_lims[0], self.lon_lims[1]), time=slice(str(self.time_lims[0]), str(self.time_lims[1])))

        if self.regrid_degree!=0:
            lon_regrid = np.arange(self.lon_lims[0], self.lon_lims[1]+self.regrid_degree, self.regrid_degree)
            lat_regrid = np.arange(self.lat_lims[0], self.lat_lims[1]-self.regrid_degree, -self.regrid_degree)
            data = data.interp(longitude=np.array(lon_regrid), method='nearest').interp(latitude=np.array(lat_regrid), method='nearest')
        
        latitude = data.latitude
        longitude = data.longitude
        data = data[str(self.variable_name)]
        
        if self.overlapping:
            creg, longitude = add_cyclic_point(np.array(data), coord=longitude)
            data = xr.DataArray(
                data=creg,
                dims=["time", "latitude", "longitude"],
                coords=dict(
                    longitude=(["longitude"], np.array(longitude)),
                    latitude=(["latitude"], np.array(latitude)),
                    time=np.array(data.time)))

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

        years_out= self.years_out +self.jump_year

        if self.one_output:
            data_red = xr.DataArray(data=mean_data, dims=["year"], coords=dict(year=years_out))
        else:
            data_red = xr.DataArray(
                data=mean_data, dims=["year", "latitude", "longitude"],
                coords=dict(
                    longitude=(["longitude"], np.array(data.longitude)),
                    latitude=(["latitude"], np.array(data.latitude)), year=years_out
                ))

        if self.detrend:
            print(f'Detrending {self.variable_name} data...')
            adjust = (data_red.polyfit(dim='year', deg=1)).sel(degree=1).polyfit_coefficients
            time_step = np.arange(len(data_red['year']))
            adjust_expanded = adjust.expand_dims(year=data_red['year'])

            if self.one_output:
                data_red = data_red - adjust_expanded * time_step
            else:
                time_step_expanded = np.expand_dims(time_step, axis=(1, 2))
                data_red = data_red - adjust_expanded * time_step_expanded
                
        if self.signal_filtering:
            print(f'Filtering {self.variable_name} data...')
            def filter_analysis(data, cut_off, order=10, filter_type='high'):
                def filter_func(x):
                    # Find indices of non-NaN values
                    valid_indices = ~np.isnan(x)
                    
                    # If all values are NaN, return NaN array
                    if not np.any(valid_indices):
                        return np.full_like(x, np.nan)

                    # Perform filtering only on non-NaN values
                    fs = 1 # Sampling frequency
                    fc = 2 * 1 / cut_off
                    b, a = signal.butter(order, fc / (fs / 2), filter_type)
                    filtered_signal = signal.filtfilt(b, a, x[valid_indices])
                    
                    # Replace NaN values with NaN in the filtered array
                    filtered_array = np.full_like(x, np.nan)
                    filtered_array[valid_indices] = filtered_signal
                    return filtered_array

                # Apply the filter along the first dimension
                filtered_data = xr.apply_ufunc(
                    filter_func,
                    data,
                    input_core_dims=[["year"]],
                    output_core_dims=[["year"]],
                    vectorize=True,
                    dask="parallelized",
                    output_dtypes=[float])
                return filtered_data.transpose('year','latitude','longitude')
            
            data_red= filter_analysis(data_red, cut_off=self.cut_off, filter_type=self.filter_type)


                
        mean_reference = (data_red.sel(year=slice(str(self.reference_period[0]), str(self.reference_period[1])))).mean(dim='year')
        std_reference = (data_red.sel(year=slice(str(self.reference_period[0]), str(self.reference_period[1])))).std(dim='year')

        anomaly = data_red - mean_reference
        normalization = anomaly / std_reference

        return data, latitude, longitude, data_red, anomaly, normalization, mean_reference, std_reference

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

    def __init__(self, train_years, validation_years, testing_years, predictor, predictant, jump_year=0, replace_nans_with_0_predictor=False, replace_nans_with_0_predictant=False):
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
        self.replace_nans_with_0_predictor = replace_nans_with_0_predictor
        self.replace_nans_with_0_predictant = replace_nans_with_0_predictant

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

        # if our data has NaNs, we must remove it to avoid problems when training the model
        def quitonans(mat, reference):
            out = mat[:, ~np.isnan(reference.mean(axis=0))]
            return out

        if self.replace_nans_with_0_predictor:
            predictor = self.predictor.fillna(value=0)
            X = predictor.where(np.isfinite(predictor), 0)
            X_train = X.sel(year=slice(self.train_years[0], self.train_years[1]))
            X_valid = X.sel(year=slice(self.validation_years[0], self.validation_years[1]))
            X_test = X.sel(year=slice(self.testing_years[0], self.testing_years[1]))
        else:
            nt, nlat, nlon = self.predictor.shape
            nt_train, nlat, nlon = X_train.shape
            nt_valid, nlat, nlon = X_valid.shape
            nt_test, nlat, nlon = X_test.shape

            X = np.reshape(np.array(self.predictor), (nt, nlat * nlon))
            X_train_reshape = np.reshape(np.array(X_train), (nt_train, nlat * nlon))
            X_valid_reshape = np.reshape(np.array(X_valid), (nt_valid, nlat * nlon))
            X_test_reshape = np.reshape(np.array(X_test), (nt_test, nlat * nlon))

            X_train = quitonans(X_train_reshape, X)
            X_valid = quitonans(X_valid_reshape, X)
            X_test = quitonans(X_test_reshape, X)
            X = quitonans(X, X)

        if self.replace_nans_with_0_predictant:
            predictant = self.predictant.fillna(value=0)
            Y = predictant.where(np.isfinite(predictant), 0)
            Y_train = Y.sel(year=slice(self.train_years[0] + self.jump_year, self.train_years[1] + self.jump_year))
            Y_valid = Y.sel(year=slice(self.validation_years[0] + self.jump_year, self.validation_years[1] + self.jump_year))
            Y_test = Y.sel(year=slice(self.testing_years[0] + self.jump_year, self.testing_years[1] + self.jump_year))
        else:
            nt, nlat, nlon = self.predictant.shape
            nt_train, nlat, nlon = Y_train.shape
            nt_valid, nlat, nlon = Y_valid.shape
            nt_test, nlat, nlon = Y_test.shape

            Y = np.reshape(np.array(self.predictant), (nt, nlat * nlon))
            Y_train_reshape = np.reshape(np.array(Y_train), (nt_train, nlat * nlon))
            Y_valid_reshape = np.reshape(np.array(Y_valid), (nt_valid, nlat * nlon))
            Y_test_reshape = np.reshape(np.array(Y_test), (nt_test, nlat * nlon))

            Y_train = quitonans(Y_train_reshape, Y)
            Y_valid = quitonans(Y_valid_reshape, Y)
            Y_test = quitonans(Y_test_reshape, Y)
            Y = quitonans(Y, Y)

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

    def __init__(self, input_shape, output_shape, layer_sizes, activations, dropout_rates=None, kernel_regularizer=None, num_conv_layers=0, num_filters=32, pool_size=2, kernel_size=3,
                 use_batch_norm=False, use_initializer=False, use_dropout=False, use_initial_skip_connections=False, use_intermediate_skip_connections=False, one_output=False,
                 learning_rate=0.001, epochs=100, random_seed=42):
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
            x = tf.keras.layers.Conv2D(filters=self.num_filters, kernel_size=(self.kernel_size, self.kernel_size), padding='same', kernel_initializer='he_normal', name=f"conv_layer_{i+1}")(x)
            if self.use_batch_norm:
                x = tf.keras.layers.BatchNormalization(name=f"batch_norm_{i+1}")(x)
            x = tf.keras.layers.Activation('elu', name=f'activation_{i+1}')(x)
            # Store the current layer as a skip connection
            x = tf.keras.layers.MaxPooling2D(pool_size=(self.pool_size, self.pool_size), name=f"max_pooling_{i+1}")(x)

        # Flatten the output
        x = tf.keras.layers.Flatten()(x)
        
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
            
            # Add dropout if applicable
            if self.use_dropout and i < len(self.dropout_rates):
                x = tf.keras.layers.Dropout(self.dropout_rates[i], name=f"dropout_{i+1}")(x)
            
            # Merge with the skip connection if specified
            if self.use_intermediate_skip_connections:
                skip_connection_last = skip_connections[-1]
                skip_connection_last = tf.keras.layers.Dense(units=self.layer_sizes[i], kernel_initializer='he_normal', name=f"dense_skip_connect_{i+1}")(skip_connection_last)
                x = tf.keras.layers.Add(name=f"merge_skip_connect_{i+1}")([x, skip_connection_last])

        # Add the output layer
        if self.kernel_regularizer is not "None":
            x = tf.keras.layers.Dense(units=self.layer_sizes[-1], activation=self.activations[-1], kernel_regularizer=self.kernel_regularizer, kernel_initializer='he_normal', name="dense_wit_reg2_"+str(self.kernel_regularizer))(x)
        
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
        if outputs_path:
            if best_model==True:
                #fig= tf.keras.utils.plot_model(model,to_file= outputs_path+'model_structure_best_model.png',show_shapes=True,show_dtype=False,show_layer_names=True,rankdir='TB',expand_nested=False,dpi=100,layer_range=None,show_layer_activations=True)
                fig= tf.keras.utils.plot_model(model,to_file= outputs_path+'model_structure_best_model.png',show_shapes=True,show_dtype=False,show_layer_names=True,rankdir='TB',expand_nested=False,dpi=100)
            else:
                #fig= tf.keras.utils.plot_model(model,to_file= outputs_path+'model_structure.png',show_shapes=True,show_dtype=False,show_layer_names=True,rankdir='TB',expand_nested=False,dpi=100,layer_range=None,show_layer_activations=True)
                fig= tf.keras.utils.plot_model(model,to_file= outputs_path+'model_structure.png',show_shapes=True,show_dtype=False,show_layer_names=True,rankdir='TB',expand_nested=False,dpi=100)

        return model

    def train_model(self, X_train, Y_train, X_valid, Y_valid, outputs_path=None, best_model=False):
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
        self, X, X_train, X_test, Y, Y_train, Y_test, lon_y, lat_y, std_y, model, time_lims, train_years, testing_years,map_nans, jump_year=0):
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

    def plotter(self, data, levs, cmap1, l1, titulo, ax, pixel_style=False, plot_colorbar=True):
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
        if pixel_style==True:
            # Create a BoundaryNorm object
            cmap1 = plt.cm.get_cmap(cmap1)
            norm = colors.BoundaryNorm(levs, ncolors=cmap1.N, clip=True)
            # Create a pixel-based colormap plot on the given axis
            im = ax.pcolormesh(self.lon_y, self.lat_y, data, cmap=cmap1, transform=ccrs.PlateCarree(), norm=norm)
        else:
            im = ax.contourf(self.lon_y, self.lat_y, data, cmap=cmap1, levels=levs, extend='both', transform=ccrs.PlateCarree())

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
            cbar = plt.colorbar(im, extend='neither', spacing='proportional', orientation='vertical', shrink=0.7, format="%2.1f")
            cbar.set_label(l1, size=15)
            cbar.ax.tick_params(labelsize=15)
        return im
    
    def prediction_vs_observation1D(self, outputs_path):
        """
        Plot the model's predictions vs. observations for both training and testing data.

        Args:
            outputs_path (str): Output path for saving the plot.

        Returns:
            None
        """
        # Calculate predictions and observations for training data
        prediction_train, observation_train = (self.model.predict(self.X_train)) * np.array(self.std_y), self.Y_train * self.std_y

        # Plot training data
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(self.train_years[0], self.train_years[1] + 1, 1), prediction_train, label='Prediction train')
        plt.plot(np.arange(self.train_years[0], self.train_years[1] + 1, 1), observation_train, label='Observation train')

        # Calculate correlation and RMSE for training data
        correlation_train, rmse_train = np.corrcoef(prediction_train.ravel(), observation_train)[0, 1], np.sqrt(
            np.mean((np.array(prediction_train).ravel() - np.array(observation_train)) ** 2))
        
        # Add correlation and RMSE text to the plot
        plt.text(0.05, 0.9, f'Correlation train: {correlation_train:.2f}', transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8))
        plt.text(0.05, 0.85, f'RMSE train: {rmse_train:.2f}', transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8))

        # Calculate predictions and observations for testing data
        prediction_test, observation_test = (self.model.predict(self.X_test)) * np.array(self.std_y), self.Y_test * self.std_y

        # Plot testing data
        plt.plot(np.arange(self.testing_years[0], self.testing_years[1], 1), prediction_test, label='Prediction test')
        plt.plot(np.arange(self.testing_years[0], self.testing_years[1], 1), observation_test, label='Observation test')
        plt.axvline(x=self.train_years[1], color='black')
        plt.legend(loc='lower left')

        # Calculate correlation and RMSE for testing data
        correlation_test, rmse_test = np.corrcoef(prediction_test.ravel(), observation_test)[0, 1], np.sqrt(
            np.mean((np.array(prediction_test).ravel() - np.array(observation_test)) ** 2))
        
        # Add correlation and RMSE text to the plot
        plt.text(0.75, 0.9, f'Correlation test: {correlation_test:.2f}', transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8))
        plt.text(0.75, 0.85, f'RMSE test: {rmse_test:.2f}', transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8))
        
        # Save the plot
        plt.savefig(outputs_path + 'Comparacion_prediccion_observacion.png')

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
        plt.style.use('default')
        fig = plt.figure(figsize=(15, 7))

        # Subplot 1: Spatial Correlation Map
        ax = fig.add_subplot(221, projection=ccrs.PlateCarree(0))
        data = spatial_correlation
        rango = 1
        ClimateDataEvaluation.plotter(self, data, np.linspace(-rango, +rango, 11), 'RdBu_r','Correlation', 'Spatial Correlation', ax, pixel_style=True)
        lon_sig, lat_sig = spatial_correlation_sig.stack(pixel=('longitude', 'latitude')).dropna('pixel').longitude, \
                        spatial_correlation_sig.stack(pixel=('longitude', 'latitude')).dropna('pixel').latitude
        #ax.scatter(lon_sig, lat_sig, s=5, c='k', marker='.', alpha=0.5, transform=ccrs.PlateCarree(), label='Significant')

        # Subplot 2: Temporal Correlation Plot
        plt.style.use('seaborn')
        ax1 = fig.add_subplot(222)
        data = {'time': temporal_correlation.time, 'Predictions correlation': temporal_correlation,}
        df = pd.DataFrame(data)
        df.set_index('time', inplace=True)
        color_dict = {'Predictions correlation': 'blue'}
        width = 0.5
        for i, col in enumerate(df.columns):
            ax1.bar(df.index, df[col], width=width, color=color_dict[col], label=col)

        ax1.set_ylim(ymin=-1, ymax=+1)
        ax1.set_title('Temporal Correlation', fontsize=18)
        ax1.legend(loc='upper right')

        # Subplot 3: Spatial RMSE Map
        plt.style.use('default')
        ax4 = fig.add_subplot(223, projection=ccrs.PlateCarree(0))
        data = spatial_rmse
        rango= int(np.nanmax(np.array(data)))
        ClimateDataEvaluation.plotter(self, data, np.linspace(0, rango+1, 10), 'OrRd','RMSE', 'Spatial RMSE', ax4, pixel_style=True)
        
        plt.style.use('seaborn')
        # Subplot 4: Temporal RMSE Plot
        ax5 = fig.add_subplot(224)
        data = {'time': temporal_rmse.time, 'Predictions RMSE': temporal_rmse}
        df = pd.DataFrame(data)
        df.set_index('time', inplace=True)
        color_dict = {'Predictions RMSE': 'orange'}
        for i, col in enumerate(df.columns):
            ax5.bar(df.index, df[col], width=width, color=color_dict[col], label=col)

        ax5.set_title('Temporal RMSE', fontsize=18)
        ax5.legend(loc='upper right')
        ax5.set_ylabel(f'[{units}]')
        fig.suptitle(f'Comparison of metrics of {var_y} from months "{months_y}"  when predicting with {predictor_region} {var_x} from months "{months_x}"', fontsize=20)
        # Adjust layout and save the figure
        plt.tight_layout()
        if best_model==True:
            plt.savefig(outputs_path + 'correlations_best_model.png')
        else:
            plt.savefig(outputs_path + 'correlations.png')

        return fig

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
        years = np.arange(self.time_lims[0], self.time_lims[-1]+1,1)
        # Loop over the folds
        for i, (train_index, testing_index) in enumerate(kf.split(self.X)):
            print(f'Fold {i+1}/{n_folds}')
            print('Training on:',years[train_index])
            print('Testing on:', years[testing_index])

            # Get the training and validation data for this fold
            X_train_fold = self.X[train_index]
            Y_train_fold = self.Y[train_index]
            X_testing_fold = self.X[testing_index]
            Y_testing_fold = self.Y[testing_index]

            model_cv = model_class.create_model()
            model_cv, record= model_class.train_model(X_train=X_train_fold, Y_train=Y_train_fold, X_valid=X_train_fold, Y_valid=Y_train_fold)
            predicted_value,observed_value= ClimateDataEvaluation.evaluation(self, X_test_other=X_testing_fold, Y_test_other=Y_testing_fold, model_other=model_cv, years_other=[(years[testing_index])[0],(years[testing_index])[-1]])
            predicted_list.append(predicted_value)
            correct_value_list.append(observed_value)

        # concatenate all the predicted values in the list into one global dataarray
        predicted_global = xr.concat(predicted_list, dim='time')
        correct_value = xr.concat(correct_value_list, dim='year')
        return predicted_global,correct_value

    def correlations_pannel(self, n_folds, predicted_global, correct_value, outputs_path,months_x, months_y, var_x, var_y, predictor_region, best_model=False, plot_differences=False):
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
        time_range = int((len(years)/n_folds))

        # Calculate correlation for each ensemble member
        fig, axes = plt.subplots(nrows=(n_folds // 3 + 1), ncols=3, figsize=(20, 10),
                                subplot_kw={'projection': ccrs.PlateCarree()})

        # Flatten the 2D array of subplots to simplify indexing
        axes = axes.flatten()

        predictions_member = predicted_global
        correct_value = correct_value.rename({'year': 'time'})
        spatial_correlation_global = xr.corr(predicted_global, correct_value, dim='time')

        for i in range(0,n_folds):
            if i < len(axes):  # Only proceed if there are available subplots
                predictions_loop = predictions_member.sel(time=slice(years[0]+i*time_range,years[time_range-1]+i*time_range))
                spatial_correlation_member = xr.corr(predictions_loop, correct_value, dim='time')

                # Plot the correlation map
                ax = axes[i]
                rango=1
                if plot_differences==True:
                    data_member = spatial_correlation_member-spatial_correlation_global
                    im= ClimateDataEvaluation.plotter(self, data=data_member,levs=np.linspace(-rango, +rango, 11), cmap1='PiYG_r', l1='Correlation', titulo='Model tested in '+str(i+1)+': '+str(years[0]+i*time_range)+'-'+str(years[time_range-1]+i*time_range), ax=ax, plot_colorbar=False)

                else:
                    data_member = spatial_correlation_member
                    im= ClimateDataEvaluation.plotter(self, data=data_member,levs=np.linspace(-rango, +rango, 11), cmap1='RdBu_r', l1='Correlation', titulo='Model tested in '+str(i+1)+': '+str(years[0]+i*time_range)+'-'+str(years[time_range-1]+i*time_range), ax=ax, plot_colorbar=False)
    

                if i==n_folds-1:
                    rango=1
                    # Plot the correlation map
                    ax = axes[i+1]
                    data_member = spatial_correlation_global
                    im2= ClimateDataEvaluation.plotter(self, data=data_member,levs=np.linspace(-rango, +rango, 11), cmap1='RdBu_r', l1='Correlation', titulo='ACC Global', ax=ax, plot_colorbar=False)

        # Add a common colorbar for all subplots
        if plot_differences==True:
            cbar_ax = fig.add_axes([0.92, 0.35, 0.02, 0.5])  # Adjust the position for your preference
            cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical', label='Correlation difference: ACC_member-ACC_global', format="%2.1f")

            cbar_ax = fig.add_axes([0.92, 0.025, 0.02, 0.3])  # Adjust the position for your preference
            cbar = fig.colorbar(im2, cax=cbar_ax, orientation='vertical', label='Correlation', format="%2.1f")
        
        else: 
            cbar_ax = fig.add_axes([0.92, 0.025, 0.02, 0.8])  # Adjust the position for your preference
            cbar = fig.colorbar(im2, cax=cbar_ax, orientation='vertical', label='Correlation', format="%2.1f")
            
        # Add a common title for the entire figure
        fig.suptitle(f'Correlations for predicting each time period of {var_y} months "{months_y}" \n with months "{months_x}" of {var_x} from {predictor_region}',fontsize=18)

        # Adjust layout for better spacing
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])

        if best_model==True:
            plt.savefig(outputs_path + 'correlations_pannel_best_model.png')
        else:
            plt.savefig(outputs_path + 'correlations_pannel.png')
        return fig

def Model_build_and_test(dictionary_hyperparams, dictionary_preprocess, cross_validation=False, n_cv_folds=0, plot_differences=False):
    print('Now creating and training the model')
    start_time = time.time()
    neural_network = NeuralNetworkModel(input_shape=dictionary_preprocess['data_split']['input_shape'], output_shape=dictionary_preprocess['data_split']['output_shape'], layer_sizes=dictionary_hyperparams['layer_sizes'], activations=dictionary_hyperparams['activations'], dropout_rates=dictionary_hyperparams['dropout_rates'], kernel_regularizer=dictionary_hyperparams['kernel_regularizer'],
                                        num_conv_layers=dictionary_hyperparams['num_conv_layers'],use_batch_norm=dictionary_hyperparams['use_batch_norm'], use_initializer=dictionary_hyperparams['use_initializer'], use_dropout=dictionary_hyperparams['use_dropout'], use_initial_skip_connections=dictionary_hyperparams['use_initial_skip_connections'], use_intermediate_skip_connections=dictionary_hyperparams['use_intermediate_skip_connections'], learning_rate=dictionary_hyperparams['learning_rate'],
                                        epochs=dictionary_hyperparams['epochs'])

    model = neural_network.create_model(outputs_path=dictionary_hyperparams['outputs_path'])
    model, record= neural_network.train_model(dictionary_preprocess['data_split']['X_train'], dictionary_preprocess['data_split']['Y_train'], dictionary_preprocess['data_split']['X_valid'], dictionary_preprocess['data_split']['Y_valid'], dictionary_hyperparams['outputs_path'])
    evaluations_toolkit= ClimateDataEvaluation(dictionary_preprocess['data_split']['X'], dictionary_preprocess['data_split']['X_train'], dictionary_preprocess['data_split']['X_test'], dictionary_preprocess['data_split']['Y'], dictionary_preprocess['data_split']['Y_train'], dictionary_preprocess['data_split']['Y_test'], 
        dictionary_preprocess['output']['lon'], dictionary_preprocess['output']['lat'], dictionary_preprocess['output']['std'], model, dictionary_hyperparams['time_lims'],  dictionary_hyperparams['train_years'], dictionary_hyperparams['testing_years'],dictionary_preprocess['output']['normalized'], jump_year=dictionary_hyperparams['jump_year'])
    
    output_directory = os.path.join(dictionary_hyperparams['outputs_path'], 'data_outputs')
    os.makedirs(output_directory, exist_ok=True)
    (dictionary_preprocess['input']['anomaly']).to_netcdf(os.path.join(output_directory,'predictor_anomalies'))
    
    if cross_validation==False:
        neural_network.performance_plot(record)
        predicted_value,correct_value= evaluations_toolkit.evaluation()
        fig1= evaluations_toolkit.correlations(predicted_value,correct_value,outputs_path=dictionary_hyperparams['outputs_path'], threshold=dictionary_hyperparams['p_value'], units=dictionary_hyperparams['units_y'], var_x=dictionary_hyperparams['name_x'], var_y=dictionary_hyperparams['name_y'], months_x=dictionary_hyperparams['months_x'], months_y=dictionary_hyperparams['months_y'], predictor_region=dictionary_hyperparams['region_predictor'], best_model=False)
        datasets, names = [predicted_value, correct_value], ['predicted_test_period', 'observed_test_period']
        # Save each dataset to a NetCDF file in the 'data_outputs' folder
        for i, ds in enumerate(datasets, start=1):
            ds.to_netcdf(os.path.join(output_directory, names[i-1]))

    else:
        predicted_value,correct_value= evaluations_toolkit.cross_validation(n_folds=n_cv_folds, model_class=neural_network)
        predicted_value.to_netcdf()
        fig1= evaluations_toolkit.correlations(predicted_value,correct_value,outputs_path= dictionary_hyperparams['outputs_path'], threshold=dictionary_hyperparams['p_value'], units=dictionary_hyperparams['units_y'], var_x=dictionary_hyperparams['name_x'], var_y=dictionary_hyperparams['name_y'], months_x=dictionary_hyperparams['months_x'], months_y=dictionary_hyperparams['months_y'], predictor_region=dictionary_hyperparams['region_predictor'], best_model=False)
        fig2= evaluations_toolkit.correlations_pannel(n_folds=n_cv_folds,predicted_global=predicted_value, correct_value=correct_value,outputs_path= dictionary_hyperparams['outputs_path'], months_x=dictionary_hyperparams['months_x'], months_y=dictionary_hyperparams['months_y'], predictor_region=dictionary_hyperparams['region_predictor'],var_x=dictionary_hyperparams['name_x'],var_y=dictionary_hyperparams['name_y'], best_model=False, plot_differences=plot_differences)
        datasets, names = [predicted_value, correct_value], ['predicted_global_cv', 'observed_global_cv']
        # Save each dataset to a NetCDF file in the 'data_outputs' folder
        for i, ds in enumerate(datasets, start=1):
            ds.to_netcdf(os.path.join(output_directory, names[i-1]))

    end_time = time.time()
    time_taken = end_time - start_time
    print(f'Training done (Time taken: {time_taken:.2f} seconds)')
    return predicted_value, correct_value

def Results_plotter(hyperparameters, dictionary_preprocess, rang_x, rang_y, predictions, observations, years_to_plot=None, save_plots=False):
    evaluations_toolkit_input= ClimateDataEvaluation(dictionary_preprocess['data_split']['X'], dictionary_preprocess['data_split']['X_train'], dictionary_preprocess['data_split']['X_test'], dictionary_preprocess['data_split']['Y'], dictionary_preprocess['data_split']['Y_train'], dictionary_preprocess['data_split']['Y_test'], 
            dictionary_preprocess['input']['lon'], dictionary_preprocess['input']['lat'], dictionary_preprocess['output']['std'], None, hyperparameters['time_lims'],  hyperparameters['train_years'], hyperparameters['testing_years'],dictionary_preprocess['output']['normalized'], jump_year=hyperparameters['jump_year'])
    evaluations_toolkit_output= ClimateDataEvaluation(dictionary_preprocess['data_split']['X'], dictionary_preprocess['data_split']['X_train'], dictionary_preprocess['data_split']['X_test'], dictionary_preprocess['data_split']['Y'], dictionary_preprocess['data_split']['Y_train'], dictionary_preprocess['data_split']['Y_test'], 
            dictionary_preprocess['output']['lon'], dictionary_preprocess['output']['lat'], dictionary_preprocess['output']['std'], None, hyperparameters['time_lims'],  hyperparameters['train_years'], hyperparameters['testing_years'],dictionary_preprocess['output']['normalized'], jump_year=hyperparameters['jump_year'])
    
    if years_to_plot:
        plotting_years= years_to_plot
    else:
        plotting_years= np.arange(hyperparameters['years_finally'][0],hyperparameters['years_finally'][-1]+1,1)
        
    for i in plotting_years:
            fig = plt.figure(figsize=(10,4), constrained_layout=True)
            # Specify the relative subplot widths
            width_ratios = [4, 1]
            gs = gridspec.GridSpec(1, 2, width_ratios=width_ratios)
            ax = fig.add_subplot(121, projection=ccrs.PlateCarree(central_longitude=-180))
            data_input= dictionary_preprocess['input']['anomaly'].sel(year=i)
            im= evaluations_toolkit_input.plotter(np.array(data_input), np.arange(-rang_x, rang_x, rang_x/10), 'RdBu_r','Anomalies [$ºC$]', 'Pred', ax, pixel_style=False, plot_colorbar=False)
            ax.set_title(f"{hyperparameters['name_x']} of months '{hyperparameters['months_x']}' from year {str(i)}",fontsize=10)

            cbar = plt.colorbar(im, extend='neither', spacing='proportional',
                            orientation='horizontal', shrink=0.8, format="%2.1f")
            cbar.set_label(f'{hyperparameters["units_x"]}', size=10)
            cbar.ax.tick_params(labelsize=10)

            ax2 = fig.add_subplot(122, projection=ccrs.PlateCarree(central_longitude=0))
            data_output_pred, data_output_obs= predictions.sel(time=i+hyperparameters['jump_year']), observations.sel(year=i+hyperparameters['jump_year'])
            im2= evaluations_toolkit_output.plotter(np.array(data_output_pred), np.arange(-rang_y, rang_y, rang_y/10), 'RdBu_r','Anomalies [$hPa$]', 'Pred', ax2, pixel_style=False, plot_colorbar=False)
            im3= ax2.contour(data_output_obs.longitude,data_output_obs.latitude,data_output_obs,colors='black',levels=np.arange(-rang_y, rang_y, rang_y/10),extend='both',transform=ccrs.PlateCarree())
            ax2.clabel(im3, inline=True, fontsize=10, fmt="%1.1f")
            
            ax2.set_title(f"{hyperparameters['name_y']} of months '{hyperparameters['months_y']}' from year {str(i+hyperparameters['jump_year'])}",fontsize=10)
            cbar = plt.colorbar(im2, extend='neither', spacing='proportional',
                            orientation='horizontal', shrink=0.7, format="%2.1f")
            cbar.set_label(f'{hyperparameters["units_y"]}', size=10)
            cbar.ax.tick_params(labelsize=10)
            
            if save_plots==True:
                output_directory = os.path.join(hyperparameters['outputs_path']+'data_outputs/', 'individual_preditions')
                os.makedirs(output_directory, exist_ok=True)
                plt.savefig(output_directory + f"prediction_evaluation_for_year_{str(i+hyperparameters['jump_year'])}png")
            plt.show()
            
def PC_analysis(hyperparameters, prediction, observation, n_modes, n_clusters, cmap='RdBu_r', save_plots=False):
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
        pcs= np.dot(eofs,np.transpose(data_sin_nan))
        pcs= (pcs-np.mean(pcs,axis=0))/np.std(pcs, axis=0)
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
        if save_plots==True:
            output_directory = os.path.join(hyperparameters['outputs_path']+'data_outputs/', 'PC_analysis')
            os.makedirs(output_directory, exist_ok=True)
            plt.savefig(output_directory + f'EOF and PC {hyperparameters["name_y"]} from months "{hyperparameters["months_y"]}"  for mode {i+1}.png')
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
            plt.savefig(output_directory + f'Weather regimes for {hyperparameters["name_y"]} from months "{hyperparameters["months_y"]}" for cluster {i+1}.png')
    else:
        clusters_pred, clusters_obs= None, None
    
    return pcs_pred, np.array(eofs_pred_list), pcs_obs, np.array(eofs_obs_list), clusters_pred, clusters_obs   
