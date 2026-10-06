#import the necessary libraries
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
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
from scipy.stats import t
import matplotlib.dates as mdates
import matplotlib.colors as colors
from matplotlib.ticker import MaxNLocator
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
plt.style.use('seaborn-v0_8-darkgrid')


def _year_slice(years, jump_year=0):
    """
    Integer slice of the 'year' coordinate.

    Args:
        years (list): [start_year, end_year] (inclusive).
        jump_year (int, optional): Offset added to both limits. Default is 0.

    Returns:
        slice: slice(start_year + jump_year, end_year + jump_year).
    """
    return slice(int(years[0]) + jump_year, int(years[1]) + jump_year)


def _detrend(da, window, fit_years=None):
    """
    Remove a trend along the 'year' dimension.

    Args:
        da (xarray.DataArray): Field with a 'year' dimension (first dimension when window > 0).
        window (int): 0 for a linear trend; otherwise the length (years) of a trailing moving average.
        fit_years (array-like, optional): Years used to fit the linear trend (e.g. the training years of a
            cross-validation fold). The trend is evaluated and removed in every year. Default: all years.

    Returns:
        xarray.DataArray: Detrended field. With window > 0 the first `window` years are left unchanged, because no
        complete trailing window is available for them.
    """
    if window == 0:
        fit = da if fit_years is None else da.sel(year=fit_years)
        coeffs = fit.polyfit(dim='year', deg=1, skipna=True)
        return da - xr.polyval(da['year'], coeffs.polyfit_coefficients)
    rolling_mean = da.rolling(year=window, center=False).mean()
    out = da.copy()
    out[window:] = da[window:] - rolling_mean[window:]
    return out


def _to_lat_lon(field, lat, lon):
    """
    Return a (latitude, longitude) DataArray from a field that may be stacked along 'space'.

    Args:
        field (xarray.DataArray): Field with dims (latitude, longitude) or (space,) in latitude-major order.
        lat, lon (array-like): Output latitudes and longitudes.

    Returns:
        xarray.DataArray: Field with dims (latitude, longitude).
    """
    if 'space' in field.dims:
        return xr.DataArray(np.asarray(field).reshape(len(lat), len(lon)), dims=['latitude', 'longitude'],
                            coords=dict(latitude=np.asarray(lat), longitude=np.asarray(lon)))
    return field


class ClimateDataPreprocessing:
    """
    Read, select and aggregate a gridded climate variable, and compute its anomalies.

    The data are read from a NetCDF file, cut to a region and period, optionally regridded, aggregated over the
    requested months of each year (seasonal mean or sum), turned into anomalies with respect to the mean of the
    reference (`train_years`) period, optionally detrended, and standardised with the standard deviation of the
    reference period computed after detrending.

    Attributes:
        relative_path (str): Path to the NetCDF file.
        lat_lims (tuple): Latitude limits (either order).
        lon_lims (tuple): Longitude limits (min, max); a maximum > 180 selects the 0-360 convention.
        time_lims (tuple): First and last year read from the file.
        scale (float): The data are divided by this factor. Default is 1.
        regrid_degree (float): Resolution (degrees) of the output grid; 0 keeps the native grid. Default is 1.
        variable_name (str): Name of the variable in the file.
        latitude_regrid (bool): Unused, kept for compatibility. Default is False.
        months (list of int): Months aggregated in each year.
        months_to_drop (list): Time stamps removed before the aggregation (e.g. the first January/February when the
            season crosses the year boundary), or ['None'].
        years_out (list): [first_year, last_year] of the aggregated series.
        detrend (bool): Whether to remove a trend from the anomalies. Default is False.
        detrend_window (int): 0 for a linear trend, otherwise the trailing moving-average length. Default is 15.
        jump_year (int): Offset added to the year labels (lead in years of the predictand). Default is 0.
        mean_seasonal_method (bool): True for the seasonal mean, False for the seasonal sum. Default is True.
        train_years (list): [start_year, end_year] of the reference period for the mean and the standard deviation.

    Methods:
        preprocess_data(): Run the processing chain and return coordinates, data, anomalies and statistics.
    """

    def __init__(
        self, relative_path, lat_lims, lon_lims, time_lims, scale=1, regrid_degree=1, variable_name=None, latitude_regrid=False,
        months=None, months_to_drop=None, years_out=None, detrend=False, detrend_window=15, jump_year=0, mean_seasonal_method=True, train_years=None):
        """
        Initialize the ClimateDataPreprocessing class.

        Args:
            relative_path (str): Path to the NetCDF file.
            lat_lims (tuple): Latitude limits (either order).
            lon_lims (tuple): Longitude limits (min, max); a maximum > 180 selects the 0-360 convention.
            time_lims (tuple): First and last year read from the file.
            scale (float, optional): The data are divided by this factor. Default is 1.
            regrid_degree (float, optional): Output grid resolution (degrees); 0 keeps the native grid. Default is 1.
            variable_name (str, optional): Name of the variable in the file.
            latitude_regrid (bool, optional): Unused, kept for compatibility. Default is False.
            months (list of int, optional): Months aggregated in each year.
            months_to_drop (list, optional): Time stamps removed before the aggregation, or ['None'].
            years_out (list, optional): [first_year, last_year] of the aggregated series.
            detrend (bool, optional): Whether to remove a trend from the anomalies. Default is False.
            detrend_window (int, optional): 0 for a linear trend, otherwise the trailing moving-average length.
                Default is 15.
            jump_year (int, optional): Offset added to the year labels. Default is 0.
            mean_seasonal_method (bool, optional): True for the seasonal mean, False for the sum. Default is True.
            train_years (list, optional): [start_year, end_year] of the reference period.
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
        Run the processing chain.

        Steps:
            1. Read the file, divide by `scale`, harmonise the time/latitude/longitude names and conventions.
            2. Select the region and period and, if `regrid_degree` != 0, interpolate linearly to the output grid.
            3. Keep the requested months, drop `months_to_drop`, and aggregate the months of each year (mean or sum).
               The months are paired by their order within each month group, so seasons crossing the year boundary
               rely on `months_to_drop` to pair the right months. Grid points missing in every month stay NaN.
            4. Anomalies: subtract the mean of the reference period (`train_years`, shifted by `jump_year`).
            5. If `detrend`: remove a linear trend (window 0) or a trailing moving average (window > 0) from the
               anomalies.
            6. Standardise with the standard deviation of the reference period, computed after detrending.

        Returns:
            latitude (xarray.DataArray): Latitudes of the output grid.
            longitude (xarray.DataArray): Longitudes of the output grid.
            data_red (xarray.DataArray): Seasonal aggregates (year, latitude, longitude), before anomalies.
            anomaly (xarray.DataArray): Anomalies (detrended if requested), in the units of the data.
            normalization (xarray.DataArray): Standardised anomalies (anomaly / std_reference).
            mean_reference (xarray.DataArray): Mean of the reference period (before detrending).
            std_reference (xarray.DataArray): Standard deviation of the reference period (after detrending).
        """

        data = xr.open_dataset(self.relative_path, decode_times=True) / self.scale
        if 'time' not in data.coords:
            if 'date' in data.coords:
                data = data.rename({'date': 'time'})
            elif 'valid_time' in data.coords:
                data = data.rename({'valid_time': 'time'})
            else:
                raise ValueError("No recognizable time coordinate found in the dataset.")

        time = data['time'].astype('datetime64[M]')
        data = data.assign_coords(time=time)

        if 'latitude' not in data.coords or 'longitude' not in data.coords:
            data = data.rename({'lat': 'latitude', 'lon': 'longitude'})

        data = data.sortby('latitude', ascending=False)

        # Handle longitude crossing the 180° meridian
        if self.lon_lims[1] > 180:
            data = data.assign_coords(longitude=np.where(data.longitude < 0, 360 + data.longitude, data.longitude)).sortby('longitude')
        else:
            data = data.assign_coords(longitude=(((data.longitude + 180) % 360) - 180)).sortby('longitude')

        # Select data based on latitude, longitude, and time limits
        if self.lat_lims[0] < self.lat_lims[1]:
            data = data.sel(latitude=slice(self.lat_lims[1], self.lat_lims[0]), longitude=slice(self.lon_lims[0], self.lon_lims[1]), time=slice(str(self.time_lims[0]), str(self.time_lims[1])))
        else:
            data = data.sel(latitude=slice(self.lat_lims[0], self.lat_lims[1]), longitude=slice(self.lon_lims[0], self.lon_lims[1]), time=slice(str(self.time_lims[0]), str(self.time_lims[1])))

        # Perform regridding if required
        if self.regrid_degree != 0:
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

        # Select data for the specified months and years
        data_red = data.sel(time=slice(f"{self.time_lims[0]}", f"{self.time_lims[-1]}"))
        data_red = data_red.sel(time=np.isin(data_red['time.month'], self.months))

        # Drop specified months (if any)
        if self.months_to_drop != ['None']:
            data_red = data_red.drop_sel(time=self.months_to_drop)

        data_red = data_red.groupby('time.month')

        # Seasonal aggregate: sum (or mean) of the selected months of each year. Missing values in some of the months
        # are ignored as before; grid points missing in all the months (e.g. land) stay NaN instead of becoming 0.
        months_stack = np.stack([np.array(data_red[i]) for i in self.months])
        mean_data = np.nansum(months_stack, axis=0)
        mean_data[~np.isfinite(months_stack).any(axis=0)] = np.nan

        # Compute seasonal aggregates (mean or sum)
        if self.mean_seasonal_method == True:
            mean_data /= len(self.months)

        years_out = np.arange(self.years_out[0], self.years_out[1]+1, 1)
        years_out = years_out + self.jump_year

        data_red = xr.DataArray(
            data=mean_data, dims=["year", "latitude", "longitude"],
            coords=dict(
                longitude=(["longitude"], np.array(data.longitude)),
                latitude=(["latitude"], np.array(data.latitude)), year=years_out
            ))

        # Anomalies with respect to the mean of the reference period
        reference = _year_slice(self.train_years, self.jump_year)
        mean_reference = data_red.sel(year=reference).mean(dim='year')
        anomaly = data_red - mean_reference

        # Detrend the anomalies if requested
        if self.detrend:
            if self.detrend_window == 0:
                print(f'Detrending {self.variable_name} data with linear detrend...')
            else:
                print(f'Detrending {self.variable_name} data with moving average...')
            anomaly = _detrend(anomaly, self.detrend_window)

        # Standardise with the standard deviation of the reference period, computed after detrending
        std_reference = anomaly.sel(year=reference).std(dim='year')
        normalization = anomaly / std_reference
        return latitude, longitude, data_red, anomaly, normalization, mean_reference, std_reference

class DataSplitter:
    """
    Split standardised predictor and predictand fields into training, validation and testing sets.

    The split is used for the single train/validation/test evaluation (and the hyperparameter search); the
    cross-validation of `ClimateDataEvaluation` builds its own folds.

    Attributes:
        train_years (list): [start_year, end_year] of the training set.
        validation_years (list): [start_year, end_year] of the validation set.
        testing_years (list): [start_year, end_year] of the testing set.
        predictor (xarray.DataArray): Standardised predictor (year, latitude, longitude).
        predictant (xarray.DataArray): Standardised predictand (year, latitude, longitude).
        jump_year (int): Offset (years) between predictor and predictand. Default is 0.

    Methods:
        prepare_data(): Return the cleaned and split arrays and the input/output shapes.
    """

    def __init__(self, train_years, validation_years, testing_years, predictor, predictant, jump_year=0):
        """
        Initialize the DataSplitter class.

        Args:
            train_years (list): [start_year, end_year] of the training set.
            validation_years (list): [start_year, end_year] of the validation set.
            testing_years (list): [start_year, end_year] of the testing set.
            predictor (xarray.DataArray): Standardised predictor (year, latitude, longitude).
            predictant (xarray.DataArray): Standardised predictand (year, latitude, longitude).
            jump_year (int, optional): Offset (years) between predictor and predictand. Default is 0.
        """
        self.train_years = train_years
        self.validation_years = validation_years
        self.testing_years = testing_years
        self.predictor = predictor
        self.predictant = predictant
        self.jump_year = jump_year

    def prepare_data(self):
        """
        Clean and split the predictor and predictand.

        Steps:
            - Missing or non-finite values are set to 0 (the climatology in standardised units).
            - The predictand is reshaped to a (year, space) matrix, with space = (latitude, longitude) in
              latitude-major order.
            - The predictand years are shifted by `jump_year` with respect to the predictor years.
            - A channel dimension is appended to the predictor splits when they are 2-D maps (for convolutional
              layers); `X` keeps its original shape.

        Returns:
            tuple: (X, X_train, X_valid, X_test, Y, Y_train, Y_valid, Y_test, input_shape, output_shape), where X is
            the cleaned predictor (year, latitude, longitude), the X splits are numpy arrays with a channel dimension,
            Y is the cleaned predictand (year, space) and its splits are DataArrays, and output_shape is an int when
            the predictand is a vector.
        """

        # Fill NaNs in predictor and predictant data with zeros
        predictor = self.predictor.fillna(value=0)
        X = predictor.where(np.isfinite(predictor), 0)  # Ensure no NaNs remain in the predictor data

        # Split predictor data into training, validation, and testing sets after cleaning
        X_train = X.sel(year=_year_slice(self.train_years))
        X_valid = X.sel(year=_year_slice(self.validation_years))
        X_test = X.sel(year=_year_slice(self.testing_years))

        # Fill NaNs in the predictant data with zeros
        predictant = self.predictant.fillna(value=0)
        Y = predictant.where(np.isfinite(predictant), 0)  # Ensure no NaNs remain in the predictant data

        # Reshape the predictant data into (time, space) matrix for model input
        Y = Y.stack(space=('latitude', 'longitude')).reset_index('space')

        # Split predictant data into training, validation, and testing sets after cleaning
        Y_train = Y.sel(year=_year_slice(self.train_years, self.jump_year))
        Y_valid = Y.sel(year=_year_slice(self.validation_years, self.jump_year))
        Y_test = Y.sel(year=_year_slice(self.testing_years, self.jump_year))

        # Get the input and output shapes after preprocessing
        input_shape = X_train[0].shape
        output_shape = Y_train[0].shape

        # Adjust output shape if the predictant is reduced to a single dimension (e.g., a vector instead of 2D matrix)
        if np.ndim(Y_train[0]) == 1:
            output_shape = output_shape[0]

        # Reshape the predictor data to include a "channel" dimension for compatibility with 2D CNNs (if required)
        if np.ndim(X_train[0]) >= 2:
            X_train = np.expand_dims(X_train, axis=-1)
            X_valid = np.expand_dims(X_valid, axis=-1)
            X_test = np.expand_dims(X_test, axis=-1)
            input_shape = X_train[0].shape  # Update input shape after adding the channel dimension

        return X, X_train, X_valid, X_test, Y, Y_train, Y_valid, Y_test, input_shape, output_shape

class NeuralNetworkModel:
    """
    Build and train a (convolutional +) dense neural network that maps a predictor field to a predictand field.

    Architecture built by `create_model` (unchanged, it defines the trained models):
        input -> [num_conv_layers x (Conv2D(num_filters, kernel_size) -> BatchNorm? -> ReLU -> MaxPooling(pool_size,
        strides 1))] -> Flatten -> Dropout(dropout_rates[0])? -> Dense(layer_sizes[0], ReLU, L2(0.01)) if
        kernel_regularizer -> for i in 0..len(layer_sizes)-2: Dense(layer_sizes[i]) -> BatchNorm? -> activations[i]
        (+ intermediate skip connection?) -> (+ initial skip connection?) -> linear Dense output (reshaped to
        output_shape if it is not a vector).
        Note: with kernel_regularizer the first width is used twice, the last entry of `layer_sizes` is only used by the
        initial skip connection, and only the first len(layer_sizes)-1 activations are used.

    Attributes:
        input_shape (tuple): Shape of one input sample.
        output_shape (int or tuple): Number of outputs (or output shape).
        layer_sizes (list of int): Widths of the dense layers (see the architecture above).
        activations (list of str): Activations of the dense layers.
        dropout_rates (list of float): Dropout rate (only the first value is used, after Flatten).
        kernel_regularizer (str or None): Any non-empty value adds the regularised first dense layer (L2, 0.01).
        num_conv_layers (int): Number of convolutional blocks. Default is 0.
        num_filters (int): Filters per convolutional layer. Default is 32.
        pool_size (int): Max-pooling window. Default is 2.
        kernel_size (int): Convolution kernel size. Default is 3.
        use_batch_norm (bool): Batch normalisation after each layer. Default is False.
        use_initializer (bool): He-normal initialisation of the dense layers. Default is False.
        use_dropout (bool): Dropout after Flatten. Default is False.
        use_init_skip_connections (bool): Skip connection from the input to the last hidden layer. Default is False.
        use_inter_skip_connections (bool): Skip connections between consecutive dense layers. Default is False.
        one_output (bool): Unused, kept for compatibility. Default is False.
        learning_rate (float): Adam learning rate. Default is 0.001.
        epochs (int): Maximum number of epochs. Default is 100.
        random_seed (int): Seed of python, numpy and TensorFlow (weights, dropout). Default is 42.

    Methods:
        create_model(outputs_path=None, best_model=False): Build the (uncompiled) Keras model.
        train_model(X_train, Y_train, X_valid, Y_valid, ...): Seed, build, compile (Adam, MSE) and fit a new model.
        performance_plot(history): Plot the training and validation loss.
    """

    def __init__(self, input_shape, output_shape, layer_sizes, activations,
                 dropout_rates=None, kernel_regularizer=None, num_conv_layers=0, num_filters=32,
                 pool_size=2, kernel_size=3, use_batch_norm=False, use_initializer=False,
                 use_dropout=False, use_init_skip_connections=False, use_inter_skip_connections=False,
                 one_output=False, learning_rate=0.001, epochs=100, random_seed=42):
        """
        Initialize the NeuralNetworkModel class (see the class docstring for the meaning of each argument).

        Args:
            input_shape (tuple): Shape of one input sample.
            output_shape (int or tuple): Number of outputs (or output shape).
            layer_sizes (list of int): Widths of the dense layers.
            activations (list of str): Activations of the dense layers.
            dropout_rates (list of float, optional): Dropout rate (first value used).
            kernel_regularizer (str, optional): Any non-empty value adds the regularised first dense layer.
            num_conv_layers (int, optional): Number of convolutional blocks.
            num_filters (int, optional): Filters per convolutional layer.
            pool_size (int, optional): Max-pooling window.
            kernel_size (int, optional): Convolution kernel size.
            use_batch_norm (bool, optional): Batch normalisation.
            use_initializer (bool, optional): He-normal initialisation of the dense layers.
            use_dropout (bool, optional): Dropout after Flatten.
            use_init_skip_connections (bool, optional): Skip connection from the input.
            use_inter_skip_connections (bool, optional): Skip connections between dense layers.
            one_output (bool, optional): Unused, kept for compatibility.
            learning_rate (float, optional): Adam learning rate.
            epochs (int, optional): Maximum number of epochs.
            random_seed (int, optional): Seed of python, numpy and TensorFlow.
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
        self.use_init_skip_connections = use_init_skip_connections
        self.use_inter_skip_connections = use_inter_skip_connections
        self.one_output = one_output
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_seed = random_seed

    def create_model(self, outputs_path=None, best_model=False):
        """
        Build the Keras model described in the class docstring (not compiled).

        Args:
            outputs_path (str, optional): Unused, kept for compatibility.
            best_model (bool, optional): Unused, kept for compatibility.

        Returns:
            tf.keras.Model: The model.
        """
        inputs = tf.keras.layers.Input(shape=self.input_shape, name="input_layer")
        x = inputs
        skip_connections = [x]

        for i in range(self.num_conv_layers):
            x = tf.keras.layers.Conv2D(self.num_filters, (self.kernel_size, self.kernel_size), padding='same',
                                       kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.L2(0.05),
                                       name=f"conv_layer_{i+1}")(x)
            if self.use_batch_norm:
                x = tf.keras.layers.BatchNormalization(name=f"batch_norm_{i+1}")(x)
            x = tf.keras.layers.Activation('relu', name=f'activation_{i+1}')(x)
            x = tf.keras.layers.MaxPooling2D(pool_size=(self.pool_size, self.pool_size), strides=1, padding='same',
                                             name=f"max_pooling_{i+1}")(x)

        x = tf.keras.layers.Flatten()(x)
        if self.use_dropout:
            x = tf.keras.layers.Dropout(self.dropout_rates[0], name=f"dropout_{1}")(x)

        if self.kernel_regularizer:
            x = tf.keras.layers.Dense(self.layer_sizes[0], activation='relu',
                                      kernel_regularizer=tf.keras.regularizers.L2(0.01),
                                      kernel_initializer='he_normal', name="dense_wit_reg")(x)

        for i in range(len(self.layer_sizes) - 1):
            skip_connections.append(x)
            if self.use_initializer:
                x = tf.keras.layers.Dense(self.layer_sizes[i], kernel_initializer='he_normal', name=f"dense_{i+1}")(x)
            else:
                x = tf.keras.layers.Dense(self.layer_sizes[i], name=f"dense_{i+1}")(x)

            if self.use_batch_norm:
                x = tf.keras.layers.BatchNormalization(name=f"batch_norm_{i+1+self.num_conv_layers}")(x)

            x = tf.keras.layers.Activation(self.activations[i], name=f'activation_{i+1+self.num_conv_layers}')(x)

            if self.use_inter_skip_connections:
                skip_last = tf.keras.layers.Dense(self.layer_sizes[i], kernel_initializer='he_normal', name=f"dense_skip_connect_{i+1}")(skip_connections[-1])
                x = tf.keras.layers.Add(name=f"merge_skip_connect_{i+1}")([x, skip_last])

        if self.use_init_skip_connections:
            skip_first = tf.keras.layers.Flatten()(skip_connections[0])
            skip_first = tf.keras.layers.Dense(int(x.shape[-1]), kernel_initializer='he_normal', name="initial_skip_connect")(skip_first)
            x = tf.keras.layers.Add(name="merge_init_skip")([x, skip_first])

        if np.ndim(self.output_shape) == 0:
            outputs = tf.keras.layers.Dense(self.output_shape, kernel_initializer='he_normal', name="output_layer")(x)
        else:
            outputs = tf.keras.layers.Dense(np.prod(self.output_shape), kernel_initializer='he_normal', name="output_layer")(x)
            outputs = tf.keras.layers.Reshape(self.output_shape)(outputs)

        model = tf.keras.Model(inputs, outputs)
        return model

    def train_model(self, X_train, Y_train, X_valid, Y_valid, outputs_path=None,use_weights=False, weights=None):
        """
        Seed, build, compile (Adam, mean squared error) and fit a new model.

        Early stopping monitors the validation loss (patience 75 epochs). The weights of the last epoch are kept
        (restore_best_weights is not used).

        Args:
            X_train (array-like): Training inputs.
            Y_train (array-like): Training targets (standardised).
            X_valid (array-like): Validation inputs (monitored by early stopping).
            Y_valid (array-like): Validation targets.
            outputs_path (str, optional): Unused, kept for compatibility.
            use_weights (bool, optional): Use per-sample weights. Default is False.
            weights (array-like, optional): Per-sample weights.

        Returns:
            tf.keras.Model: Trained model.
            dict: Training history (loss and val_loss per epoch).
        """
        # python, numpy and TensorFlow seeds (weights initialisation, dropout, shuffling) -> reproducible training
        tf.keras.utils.set_random_seed(self.random_seed)

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=75)
        model = self.create_model(outputs_path)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss=tf.keras.losses.MeanSquaredError())
        if use_weights==True:
            history = model.fit(X_train, Y_train, epochs=self.epochs, validation_data=(X_valid, Y_valid), callbacks=[callback], verbose=0, sample_weight=weights)
        else:
            history = model.fit(X_train, Y_train, epochs=self.epochs, validation_data=(X_valid, Y_valid), callbacks=[callback], verbose=0)

        return model, history.history

    def performance_plot(self, history):
        """
        Plot the training and validation loss against the epochs.

        Args:
            history (dict): Training history returned by `train_model`.

        Returns:
            matplotlib.figure.Figure: The figure.
        """
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        fig.suptitle('Model Performance')

        ax.plot(history['loss'], label='Training Loss')
        ax.plot(history['val_loss'], label='Validation Loss')

        ax.set_title('Model 1')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend(['Training', 'Validation'])
        plt.text(0.6, 0.7, f"Loss training: {history['loss'][-1]:.2f} and validation {history['val_loss'][-1]:.2f}",
                 transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))

        plt.show()
        return fig

class ClimateDataEvaluation:
    """
    Evaluate a trained model: predictions on a test set, cross-validation, skill maps and attributions.

    Units: the network works with standardised anomalies. Predictions, observations and attributions are returned in
    the units of the predictand by multiplying by the SAME standard deviation used to standardise the targets: `std_y`
    (from `preprocess_data`) for the single train/test evaluation, and the standard deviation of the training years of
    each fold in `cross_validation`.

    Attributes:
        X (xarray.DataArray): Predictor. For `cross_validation`: seasonal aggregates (year, latitude, longitude) before
            anomalies (`preprocess['input']['data']`).
        X_train (array-like): Training inputs (single evaluation).
        X_test (array-like): Testing inputs (single evaluation).
        Y (xarray.DataArray): Predictand. For `cross_validation`: seasonal aggregates (year, latitude, longitude)
            (`preprocess['output']['data']`); an already stacked (year, space) field is also accepted.
        Y_train (array-like): Training targets (single evaluation).
        Y_test (array-like): Testing targets (single evaluation).
        lon_y (array-like): Longitudes of the predictand.
        lat_y (array-like): Latitudes of the predictand.
        std_y (xarray.DataArray): Standard deviation used to standardise the predictand in `preprocess_data`.
        model (tf.keras.Model): Trained model (single evaluation).
        time_lims (tuple): First and last year of the samples.
        train_years (list): Training years of the single evaluation.
        testing_years (list): Testing years of the single evaluation.
        jump_year (int): Offset (years) between predictor and predictand. Default is 0.
        map_nans (xarray.DataArray): Standardised predictand map (used for its shape).
        detrend_x, detrend_y (bool): Detrend the predictor / predictand inside each fold. Default is False.
        detrend_x_window, detrend_y_window (int): 0 for a linear trend, otherwise the trailing moving-average length.
        importances (bool): Compute Integrated-Gradients attributions in `cross_validation`. Default is False.
        region_atributted (list): [lat_lims, lon_lims] of the predicted points whose attributions are computed.

    Methods:
        plotter(...): Draw a map (pcolormesh or contourf) with coastlines and an optional colorbar.
        evaluation(...): Predict a test set and return predictions and observations in the units of the predictand.
        correlations(...): ACC and RMSE maps and their (area-weighted) yearly global values; saves a figure.
        attributions(...): Integrated-Gradients attributions of the predicted points of a region.
        cross_validation(n_folds, model_class, validation_fraction=0.0): K-fold cross-validated predictions,
            observations and (optionally) attributions.
        correlations_pannel(...): ACC map of each cross-validation fold.
    """

    def __init__(self, X, X_train, X_test, Y, Y_train, Y_test, lon_y, lat_y, std_y, model, time_lims, train_years, testing_years, map_nans, jump_year=0, detrend_x=False, detrend_x_window=10, detrend_y=False, detrend_y_window=10, importances=False, region_atributted=None):
        """
        Initialize the ClimateDataEvaluation class (see the class docstring for the meaning of each argument).

        Args:
            X (xarray.DataArray): Predictor (seasonal aggregates for cross-validation).
            X_train (array-like): Training inputs.
            X_test (array-like): Testing inputs.
            Y (xarray.DataArray): Predictand (seasonal aggregates for cross-validation).
            Y_train (array-like): Training targets.
            Y_test (array-like): Testing targets.
            lon_y (array-like): Longitudes of the predictand.
            lat_y (array-like): Latitudes of the predictand.
            std_y (xarray.DataArray): Standard deviation of the predictand from `preprocess_data`.
            model (tf.keras.Model): Trained model.
            time_lims (tuple): First and last year of the samples.
            train_years (list): Training years.
            testing_years (list): Testing years.
            map_nans (xarray.DataArray): Standardised predictand map (used for its shape).
            jump_year (int, optional): Offset between predictor and predictand. Default is 0.
            detrend_x (bool, optional): Detrend the predictor in each fold. Default is False.
            detrend_x_window (int, optional): 0 for a linear trend, otherwise the moving-average length. Default is 10.
            detrend_y (bool, optional): Detrend the predictand in each fold. Default is False.
            detrend_y_window (int, optional): 0 for a linear trend, otherwise the moving-average length. Default is 10.
            importances (bool, optional): Compute attributions in `cross_validation`. Default is False.
            region_atributted (list, optional): [lat_lims, lon_lims] of the attributed predicted points.
        """
        self.X = X
        self.X_train = X_train
        self.X_test = X_test
        self.Y = Y
        self.Y_train = Y_train
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
        Draw a map of a predictand-grid field with Cartopy.

        Args:
            data (array-like): Field (latitude, longitude) on the predictand grid.
            levs (list of float): Colour levels.
            cmap1 (str or Colormap): Colormap.
            l1 (str): Colorbar label.
            titulo (str): Panel title.
            ax (cartopy GeoAxes): Axes to draw on.
            pixel_style (bool, optional): pcolormesh (True) or contourf (False). Default is False.
            plot_colorbar (bool, optional): Add a vertical colorbar. Default is True.
            acc_norm (matplotlib.colors.BoundaryNorm, optional): Norm to use instead of BoundaryNorm(levs).
            extend (str, optional): Colorbar extension ('neither', 'both', 'min', 'max'). Default is 'neither'.

        Returns:
            The mappable created by pcolormesh or contourf.
        """
        # Create a filled contour plot or pixel-based colormap plot on the given axis
        cmap1 = plt.cm.get_cmap(cmap1)
        norm = colors.BoundaryNorm(levs, ncolors=cmap1.N, clip=True)
        if acc_norm:
            norm = acc_norm

        if pixel_style:
            im = ax.pcolormesh(self.lon_y, self.lat_y, data, cmap=cmap1, transform=ccrs.PlateCarree(), norm=norm, zorder=1)
        else:
            im = ax.contourf(self.lon_y, self.lat_y, data, cmap=cmap1, levels=levs, extend=extend, transform=ccrs.PlateCarree(), norm=norm, zorder=1)

        ax.coastlines(linewidth=0.75, zorder=3)
        ax.set_title(titulo, fontsize=18)
        gl = ax.gridlines(draw_labels=True, zorder=4)
        gl.xlines = False
        gl.ylines = False
        gl.top_labels = False
        gl.right_labels = False

        if plot_colorbar:
            cbar = plt.colorbar(im, extend=extend, orientation='vertical', shrink=0.9, format="%2.1f")
            cbar.set_label(l1, size=15)
            cbar.ax.tick_params(labelsize=12)
            if acc_norm:
                cbar.set_ticks(ticks=levs, labels=levs)

        return im

    def evaluation(self, X_test_other=None, Y_test_other=None, model_other=None, years_other=None, std_y=None):
        """
        Predict a test set and return predictions and observations in the units of the predictand.

        Args:
            X_test_other (array-like, optional): Test inputs (default: self.X_test).
            Y_test_other (array-like, optional): Standardised test targets (default: self.Y_test).
            model_other (tf.keras.Model, optional): Model (default: self.model).
            years_other (list, optional): [first_year, last_year] of the test samples (default: self.testing_years).
            std_y (xarray.DataArray, optional): Standard deviation used to standardise these targets (latitude,
                longitude) or (space); it must be the one used for the targets. Default: self.std_y.

        Returns:
            predicted (xarray.DataArray): Predicted anomalies (time, latitude, longitude).
            correct_value (xarray.DataArray): Observed anomalies (year, latitude, longitude).
        """
        if X_test_other is not None:
            X_test, Y_test, model, testing_years = X_test_other, Y_test_other, model_other, years_other
        else:
            X_test, Y_test, model, testing_years = self.X_test, self.Y_test, self.model, self.testing_years
        std_y = _to_lat_lon(self.std_y if std_y is None else std_y, self.lat_y, self.lon_y)

        predicted = model.predict(np.array(X_test))
        test_years = np.arange(testing_years[0] + self.jump_year, testing_years[1] + self.jump_year + 1, 1)
        if np.ndim(predicted) <= 2:
            nt = predicted.shape[0]
            predicted = np.reshape(predicted, (nt, len(np.array(self.lat_y)), len(np.array(self.lon_y))))
            Y_test = np.reshape(np.array(Y_test), (nt, len(np.array(self.lat_y)), len(np.array(self.lon_y))))

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

        predicted = predicted * std_y
        correct_value = correct_value * std_y

        return predicted, correct_value

    def correlations(self, predicted, correct_value, outputs_path, threshold, units, months_x, months_y, var_x, var_y, predictor_region, best_model=False):
        """
        ACC and RMSE maps and their yearly global values; the figure is saved in `outputs_path`.

        Panels: ACC map (hatched where the two-sided p-value > threshold), yearly area-weighted (cos latitude) pattern
        correlation with the one-sided critical value as a dashed line, RMSE map, and yearly area-weighted RMSE.

        Args:
            predicted (xarray.DataArray): Predictions (time, latitude, longitude).
            correct_value (xarray.DataArray): Observations (year, latitude, longitude).
            outputs_path (str): Directory of the saved figure.
            threshold (float): p-value threshold.
            units (str): Units of the predictand (RMSE axis label).
            months_x, months_y (list): Months of the predictor and predictand (title).
            var_x, var_y (str): Names of the predictor and predictand (title).
            predictor_region (str): Name of the predictor region (title).
            best_model (bool, optional): Save as 'correlations_best_model.png'. Default is False.

        Returns:
            matplotlib.figure.Figure: The figure.
        """
        predictions = predicted
        observations = correct_value.rename({'year': 'time'})
        spatial_correlation = xr.corr(predictions, observations, dim='time')
        p_value = xs.pearson_r_p_value(predictions, observations, dim='time')
        spatial_rmse = np.sqrt(((predictions - observations) ** 2).mean(dim='time'))
        # yearly global values, area weighted (cos latitude)
        weights = np.cos(np.deg2rad(observations.latitude))
        wmean = lambda da: da.weighted(weights).mean(dim=('latitude', 'longitude'))
        pred_a, obs_a = predictions - wmean(predictions), observations - wmean(observations)
        temporal_correlation = wmean(pred_a * obs_a) / np.sqrt(wmean(pred_a ** 2) * wmean(obs_a ** 2))
        temporal_rmse = np.sqrt(wmean((predictions - observations) ** 2))

        fig = plt.figure(figsize=(15, 7))

        ax = fig.add_subplot(221, projection=ccrs.PlateCarree())
        data = spatial_correlation
        acc_clevs = [-1, -0.9, -0.8, -0.7, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1]
        colors = np.array([[0,0,255], [0,102,201], [119,153,255], [119,187,255], [170,221,255], [170,255,255], [170,170,170], [255,255,0], [255,204,0], [255,170,0], [255,119,0], [255,0,0], [119,0,34]], np.float32) / 255.0
        acc_map, acc_norm = from_levels_and_colors(acc_clevs, colors)
        self.plotter(data, acc_clevs, acc_map, 'Correlation', 'ACC map', ax, pixel_style=True, acc_norm=acc_norm)
        hatch_mask = p_value > threshold                       # hatched: not significant
        ax.contourf(spatial_correlation.longitude, spatial_correlation.latitude, hatch_mask, levels=[0, 0.5, 1], hatches=['', '//'], alpha=0, transform=ccrs.PlateCarree(), zorder=2)

        ax1 = fig.add_subplot(222)
        data = {'time': temporal_correlation.time, 'Predictions correlation': temporal_correlation}
        df = pd.DataFrame(data)
        df.set_index('time', inplace=True)
        color_dict = {'Predictions correlation': 'blue'}
        width = 0.8
        for i, col in enumerate(df.columns):
            ax1.bar(df.index, df[col], width=width, color=color_dict[col], label=col)
        dof = len(predictions.time) - 2
        t_crit = np.abs(t.ppf(threshold, dof))
        critical_corr = t_crit / np.sqrt(dof + t_crit**2)
        ax1.axhline(y=critical_corr, color='black', linestyle='--', linewidth=1)
        ax1.set_ylim(ymin=-.75, ymax=+1)
        ax1.set_title('Time series of global ACC', fontsize=18)
        ax1.legend(loc='lower right')
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax4 = fig.add_subplot(223, projection=ccrs.PlateCarree())
        data = spatial_rmse
        rango = int(np.nanmax(np.array(data)))
        self.plotter(data, np.linspace(0, rango+1, 10), 'OrRd', 'RMSE', 'RMSE map', ax4, pixel_style=True)

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
        ax5.xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.suptitle(f'Comparison of metrics of {var_y} from months "{months_y}" when predicting with {predictor_region} {var_x} from months "{months_x}"', fontsize=20)
        plt.tight_layout()
        if best_model:
            plt.savefig(outputs_path + 'correlations_best_model.png')
        else:
            plt.savefig(outputs_path + 'correlations.png')

        return fig

    def attributions(self, model, X_test, lat_lims, lon_lims, std_y, test_years):
        """
        Integrated-Gradients attributions of the predicted points of a region.

        For every test sample and every predicted grid point inside [lat_lims] x [lon_lims], the attribution of each
        input grid point is computed with Integrated Gradients (alibi; Gauss-Legendre, 25 steps) along the path from a
        zero baseline (the climatology in standardised units) to the sample. Attributions are multiplied by the
        standard deviation of the predicted point, the same one used to de-normalise the predictions, so that the
        attributions of a sample sum (approximately, up to the integration error) to the prediction minus the
        prediction of the baseline, in the units of the predictand.

        Args:
            model (tf.keras.Model): Trained model.
            X_test (xarray.DataArray): Test inputs (time, latitude, longitude), standardised.
            lat_lims (tuple): Latitude limits of the attributed predicted points (either order).
            lon_lims (tuple): Longitude limits of the attributed predicted points (either order).
            std_y (xarray.DataArray or np.ndarray): Standard deviation of the predictand, (latitude, longitude) or
                (space).
            test_years (array-like): Years of the test samples.

        Returns:
            xarray.DataArray: Attributions (latitude_pred, longitude_pred, latitude_input, longitude_input, time).
        """
        std_y = np.asarray(_to_lat_lon(std_y, self.lat_y, self.lon_y) if isinstance(std_y, xr.DataArray) else std_y)
        if std_y.ndim == 1:
            std_y = std_y.reshape(len(np.array(self.lat_y)), len(np.array(self.lon_y)))

        ig = IntegratedGradients(model, layer=None, method="gausslegendre", n_steps=25, internal_batch_size=32)

        nt, nlat_in, nlon_in = X_test.shape
        latitudes_out = np.array(self.lat_y)
        longitudes_out = np.array(self.lon_y)

        # Create 2D grid and find indices within lat/lon limits
        lon_grid, lat_grid = np.meshgrid(longitudes_out, latitudes_out)
        mask = (lat_grid >= min(lat_lims)) & (lat_grid <= max(lat_lims)) & \
            (lon_grid >= min(lon_lims)) & (lon_grid <= max(lon_lims))
        idx_i, idx_j = np.where(mask)

        if len(idx_i) == 0:
            raise ValueError("No output points found within the specified lat/lon limits.")

        nlat_pred = len(np.unique(lat_grid[mask]))
        nlon_pred = len(np.unique(lon_grid[mask]))

        print(f"Computing attributions for {len(idx_i)} predicted points in region [{lat_lims[0]},{lat_lims[1]}], [{lon_lims[0]},{lon_lims[1]}]...")

        # Initialize attribution array
        all_importances = np.zeros((nlat_pred, nlon_pred, nlat_in, nlon_in, nt))
        lat_pred_vals = np.sort(np.unique(lat_grid[mask]))
        lon_pred_vals = np.sort(np.unique(lon_grid[mask]))

        # Map from (i,j) in full grid to local (i',j') in regional grid
        lat_idx_map = {lat: i for i, lat in enumerate(lat_pred_vals)}
        lon_idx_map = {lon: j for j, lon in enumerate(lon_pred_vals)}

        for t in range(nt):
            X_sample = X_test[t:t + 1]
            baseline = X_sample * 0

            for i_global, j_global in zip(idx_i, idx_j):
                target_idx = i_global * len(longitudes_out) + j_global
                explanation = ig.explain(np.array(X_sample), baselines=np.array(baseline), target=np.array([target_idx]))
                attr = explanation.attributions[0]  # shape: (1, lat_in, lon_in)
                attr_scaled = attr * std_y[i_global, j_global]

                # Map to regional (i, j)
                i_local = lat_idx_map[latitudes_out[i_global]]
                j_local = lon_idx_map[longitudes_out[j_global]]

                all_importances[i_local, j_local, :, :, t] = attr_scaled.squeeze()

        # Build xarray DataArray
        importances = xr.DataArray(
            data=all_importances,
            dims=["latitude_pred", "longitude_pred", "latitude_input", "longitude_input", "time"],
            coords=dict(
                latitude_pred=lat_pred_vals,
                longitude_pred=lon_pred_vals,
                latitude_input=np.array(X_test.latitude),
                longitude_input=np.array(X_test.longitude),
                time=np.array(test_years)
            ),
            name="attributions"
        )

        return importances

    def cross_validation(self, n_folds, model_class, validation_fraction=0.0):
        """
        K-fold cross-validation (consecutive folds, no shuffling).

        For each fold, using only the training years of the fold:
            1. anomalies of the predictor and predictand with respect to the training-years mean;
            2. detrending if requested (linear trend fitted on the training years and removed from all years, or a
               trailing moving average, which only uses past years);
            3. standardisation with the training-years standard deviation (after detrending); missing values -> 0;
            4. a new model is trained (`model_class.train_model`, same seed in every fold) and the test years are
               predicted;
            5. predictions, observations and attributions are converted back to the units of the predictand with the
               standard deviation of step 3, so the observations are the fold anomalies exactly.

        Args:
            n_folds (int): Number of folds (n_folds = number of years gives leave-one-year-out).
            model_class (NeuralNetworkModel): Model definition (architecture, learning rate, epochs, seed).
            validation_fraction (float, optional): Fraction of the training years held out (random, fixed seed per
                fold) to monitor early stopping. 0 (default) keeps the previous behaviour: early stopping monitors the
                training years themselves.

        Returns:
            predicted_global (xarray.DataArray): Cross-validated predictions (time, latitude, longitude).
            correct_value (xarray.DataArray): Corresponding anomalies (year, latitude, longitude).
            years_division_list (list): Test years of each fold.
            importances_global (xarray.DataArray, only if self.importances): Attributions of all folds.
        """
        # Define the KFold object
        kf = KFold(n_splits=n_folds, shuffle=False)

        # Create lists to store results
        predicted_list = []
        correct_value_list = []
        importances_list = []
        years = np.arange(self.time_lims[0], self.time_lims[-1] + 1, 1)
        years_division_list = []
        stacked_y = 'space' in self.Y.dims

        # Loop over the folds
        for i, (train_index, testing_index) in enumerate(kf.split(self.X)):
            # 1. anomalies with respect to the training years of the fold
            X = self.X - self.X.isel(year=train_index).mean('year')
            Y = self.Y - self.Y.isel(year=train_index).mean('year')

            # 2. detrending (trend fitted on the training years only)
            if self.detrend_x:
                X = _detrend(X, self.detrend_x_window, fit_years=self.X.year.values[train_index])
            if self.detrend_y:
                Y = _detrend(Y, self.detrend_y_window, fit_years=self.Y.year.values[train_index])

            # 3. standardisation with the training-years standard deviation (after detrending)
            std_reference_x = X.isel(year=train_index).std('year')
            std_reference_y = Y.isel(year=train_index).std('year')
            X, Y = X / std_reference_x, Y / std_reference_y
            X, Y = X.where(np.isfinite(X), 0), Y.where(np.isfinite(Y), 0)
            if not stacked_y:
                Y = Y.stack(space=('latitude', 'longitude')).reset_index('space')  # Convert to (time, space) matrix

            # training / validation years (validation_fraction = 0: validation = training, as before)
            if validation_fraction > 0:
                n_valid = max(1, int(round(validation_fraction * len(train_index))))
                valid_index = np.sort(np.random.default_rng(i).choice(train_index, size=n_valid, replace=False))
                fit_index = np.setdiff1d(train_index, valid_index)
            else:
                valid_index, fit_index = train_index, train_index

            print(f'Fold {i + 1}/{n_folds}')
            print('Training on:', years[fit_index])
            if validation_fraction > 0:
                print('Validating on:', years[valid_index])
            print('Testing on:', years[testing_index])

            X_train_fold = X[fit_index, :]
            Y_train_fold = Y[fit_index, :]
            X_validation_fold = X[valid_index, :]
            Y_validation_fold = Y[valid_index, :]
            X_testing_fold = X[testing_index]
            Y_testing_fold = Y[testing_index]

            # 4. train a new model and predict the test years
            model_cv, record = model_class.train_model(X_train=X_train_fold, Y_train=Y_train_fold, X_valid=X_validation_fold, Y_valid=Y_validation_fold,use_weights=False,weights=None)

            # 5. back to the units of the predictand with the standard deviation used for the targets of this fold
            predicted_value, observed_value = ClimateDataEvaluation.evaluation(self, X_test_other=X_testing_fold, Y_test_other=Y_testing_fold, model_other=model_cv, years_other=[(years[testing_index])[0], (years[testing_index])[-1]], std_y=std_reference_y)
            predicted_list.append(predicted_value)
            correct_value_list.append(observed_value)
            years_division_list.append(years[testing_index])
            if self.importances:
                imp_fold = ClimateDataEvaluation.attributions(self, model_cv, X_testing_fold, lat_lims=self.region_atributted[0], lon_lims=self.region_atributted[1], std_y=std_reference_y, test_years=years[testing_index])
                importances_list.append(imp_fold)

        # Concatenate all the predicted values in the list into one global dataarray
        predicted_global = xr.concat(predicted_list, dim='time')
        correct_value = xr.concat(correct_value_list, dim='year')
        if self.importances:
            importances_global = xr.concat(importances_list, dim='time')
            return predicted_global, correct_value, years_division_list, importances_global
        else:
            return predicted_global, correct_value, years_division_list

    def correlations_pannel(self, n_folds, predicted_global, correct_value, threshold, years_division, outputs_path, months_x, months_y, var_x, var_y, predictor_region, best_model=False, plot_differences=False):
        """
        ACC map of each cross-validation fold (computed over its test years) and of all the folds together.

        With one test year per fold the per-fold correlation is not defined; use it with folds of several years.

        Args:
            n_folds (int): Number of folds.
            predicted_global (xarray.DataArray): Cross-validated predictions (time, latitude, longitude).
            correct_value (xarray.DataArray): Observations (year, latitude, longitude).
            threshold (float): p-value threshold (hatching: not significant).
            years_division (list): Test years of each fold.
            outputs_path (str): Directory of the saved figure.
            months_x, months_y (list): Months of the predictor and predictand (title).
            var_x, var_y (str): Names of the predictor and predictand (title).
            predictor_region (str): Name of the predictor region (title).
            best_model (bool, optional): Save as 'correlations_pannel_best_model.png'. Default is False.
            plot_differences (bool, optional): Plot each fold's ACC minus the global ACC. Default is False.

        Returns:
            matplotlib.figure.Figure: The figure.
        """
        # Create a list of ensemble members
        years = np.arange(self.time_lims[0], self.time_lims[-1] + 1, 1)

        # Calculate correlation for each ensemble member
        fig, axes = plt.subplots(nrows=(n_folds // 3 + 1), ncols=3, figsize=(15, 5),
                                subplot_kw={'projection': ccrs.PlateCarree()})

        # Flatten the 2D array of subplots to simplify indexing
        axes = axes.flatten()

        predictions_member = predicted_global
        correct_value = correct_value.rename({'year': 'time'})
        spatial_correlation_global = xr.corr(predicted_global, correct_value, dim='time')
        p_value_global = xs.pearson_r_p_value(predicted_global, correct_value, dim='time')
        acc_clevs = [-1, -0.9, -0.8, -0.7, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1]
        colors = np.array([[0, 0, 255], [0, 102, 201], [119, 153, 255], [119, 187, 255], [170, 221, 255],
                           [170, 255, 255], [170, 170, 170], [255, 255, 0], [255, 204, 0], [255, 170, 0],
                           [255, 119, 0], [255, 0, 0], [119, 0, 34]], np.float32) / 255.0

        acc_map, acc_norm = from_levels_and_colors(acc_clevs, colors)

        for i in range(0, n_folds):
            if i < len(axes):  # Only proceed if there are available subplots
                years_fold_div = years_division[i]

                predictions_loop = predictions_member.sel(time=slice(years_fold_div[0], years_fold_div[-1]))
                spatial_correlation_member = xr.corr(predictions_loop, correct_value.sel(time=slice(years_fold_div[0], years_fold_div[-1])), dim='time')
                p_value = xs.pearson_r_p_value(predictions_loop, correct_value.sel(time=slice(years_fold_div[0], years_fold_div[-1])), dim='time')

                # Plot the correlation map
                ax = axes[i]
                rango = 1
                if plot_differences:
                    data_member = spatial_correlation_member - spatial_correlation_global
                    im = ClimateDataEvaluation.plotter(self, data=data_member, levs=acc_clevs, cmap1='PiYG_r', l1='Correlation', titulo='Model tested in ' + str(i + 1) + ': ' + str(years_fold_div[0]) + '-' + str(years_fold_div[-1]), ax=ax, plot_colorbar=False)
                    hatch_mask = p_value > threshold
                    ax.contourf(data_member.longitude, data_member.latitude, hatch_mask, levels=[0, 0.5, 1], hatches=['', '//'], alpha=0, transform=ccrs.PlateCarree())

                else:
                    data_member = spatial_correlation_member
                    im = ClimateDataEvaluation.plotter(self, data=data_member, levs=acc_clevs, cmap1=acc_map, l1='Correlation', titulo='Model tested in ' + str(i + 1) + ': ' + str(years_fold_div[0]) + '-' + str(years_fold_div[-1]), ax=ax, plot_colorbar=False, acc_norm=acc_norm)
                    hatch_mask = p_value > threshold
                    ax.contourf(data_member.longitude, data_member.latitude, hatch_mask, levels=[0, 0.5, 1], hatches=['', '//'], alpha=0, transform=ccrs.PlateCarree())

                if i == n_folds - 1:
                    rango = 1
                    # Plot the correlation map
                    ax = axes[i + 1]
                    data_member = spatial_correlation_global
                    im2 = ClimateDataEvaluation.plotter(self, data=data_member, levs=acc_clevs, cmap1=acc_map, l1='Correlation', titulo='ACC Global', ax=ax, plot_colorbar=False, acc_norm=acc_norm)
                    hatch_mask = p_value_global > threshold
                    ax.contourf(data_member.longitude, data_member.latitude, hatch_mask, levels=[0, 0.5, 1], hatches=['', '//'], alpha=0, transform=ccrs.PlateCarree())

        # Remove unused axes
        for ax in axes[n_folds + 1:]:
            ax.remove()

        # Add a common colorbar for all subplots
        if plot_differences:
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
        fig.suptitle(f'Correlations for predicting each time period of {var_y} months "{months_y}" \n with months "{months_x}" of {var_x} from {predictor_region}', fontsize=18)

        # Adjust layout for better spacing
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])

        if best_model:
            plt.savefig(outputs_path + 'correlations_pannel_best_model.png')
        else:
            plt.savefig(outputs_path + 'correlations_pannel.png')
        return fig

class BestModelAnalysis:
    """
    Hyperparameter search (Keras Tuner random search) and evaluation of the best model.

    Attributes:
        input_shape (tuple): Shape of one input sample.
        output_shape (int or tuple): Number of outputs (or output shape).
        X, X_train, X_valid, X_test (array-like): Predictor and its splits (see `DataSplitter`).
        Y, Y_train, Y_valid, Y_test (array-like): Predictand and its splits.
        lon_y, lat_y (array-like): Coordinates of the predictand.
        std_y (xarray.DataArray): Standard deviation of the predictand from `preprocess_data`.
        time_lims (tuple): First and last year of the samples.
        train_years, testing_years (list): Training and testing years.
        params_selection (dict): Search space (pos_number_layers, pos_layer_sizes, pos_activations, pos_dropout,
            pos_kernel_regularizer, search_skip_connections, pos_conv_layers, pos_learning_rate).
        epochs (int): Maximum number of epochs.
        outputs_path (str): Output directory.
        output_original (xarray.DataArray): Standardised predictand map (used for its shape).
        random_seed (int): Seed of python, numpy and TensorFlow. Default is 42.
        jump_year (int): Offset between predictor and predictand. Default is 0.
        threshold (float): p-value threshold of the evaluation figures. Default is 0.1.
        detrend_x, detrend_y (bool), detrend_x_window, detrend_y_window (int): Detrending used in cross-validation.

    Methods:
        build_model(hp): Compiled model for one set of hyperparameters.
        tuner_searcher(max_trials): Run the random search.
        bm_evaluation(...): Train the best model and evaluate it on the test set or by cross-validation.
    """

    def __init__(
        self, input_shape, output_shape, X, X_train, X_valid, X_test, Y, Y_train, Y_valid, Y_test, lon_y, lat_y, std_y, time_lims, train_years, testing_years, params_selection, epochs,
        outputs_path, output_original, random_seed=42, jump_year=0, threshold=0.1, detrend_x=False, detrend_x_window=10, detrend_y=False, detrend_y_window=10):
        """
        Initialize the BestModelAnalysis class (see the class docstring for the meaning of each argument).
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.X = X
        self.X_train = X_train
        self.X_valid = X_valid
        self.X_test = X_test
        self.Y = Y
        self.Y_train = Y_train
        self.Y_valid = Y_valid
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
        Build and compile (Adam, MSE) a model for one set of hyperparameters drawn by Keras Tuner.

        Searched: number of layers, their widths and activations, kernel regulariser, learning rate, dropout rate,
        skip connections (if `search_skip_connections`), and the convolutional layers (if `pos_conv_layers` > 0).
        Batch normalisation, He initialisation and dropout are always used in the built model.

        Args:
            hp (kerastuner.HyperParameters): Hyperparameters object.

        Returns:
            tf.keras.Model: Compiled model.
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
        use_batch_norm = hp.Choice('batch_normalization', ['True', 'False'])
        print("Use batch normalization:", use_batch_norm)

        # Choose whether to use He initialization
        use_initializer = hp.Choice('he_initialization', ['True', 'False'])
        print("Use He initialization:", use_initializer)

        # Define the learning rate
        learning_rate = hp.Choice('learning_rate', pos_learning_rate)

        # If dropout is chosen, define dropout rate
        dropout_rate = [hp.Choice('dropout', pos_dropout)]
        print("Dropout rate:", dropout_rate)

        # Choose whether to use skip connections
        if search_skip_connections == 'True':
            use_init_skip_connections = hp.Choice('initial_skip_connection', ['True', 'False'])
            print("Use initial skip connections:", use_init_skip_connections)
            use_inter_skip_connections = hp.Choice('intermediate_skip_connections', ['True', 'False'])
            print("Use intermediate skip connections:", use_inter_skip_connections)
        else:
            use_init_skip_connections, use_inter_skip_connections = False, False

        # Define hyperparameters related to convolutional layers
        if pos_conv_layers > 0:
            num_conv_layers = hp.Int('#_convolutional_layers', 0, pos_conv_layers)
            if num_conv_layers > 0:
                print("Number of Convolutional Layers:", num_conv_layers)
                num_filters = hp.Choice('number_of_filters_per_conv', [2, 4, 16])
                print("Number of filters:", num_filters)
                pool_size = hp.Choice('pool size', [2, 4])
                print("Pool size:", pool_size)
                kernel_size = hp.Choice('kernel size', [3, 6])
                print("Kernel size:", kernel_size)
            else:
                num_filters, pool_size, kernel_size = 0, 0, 0
        else:
            num_conv_layers, num_filters, pool_size, kernel_size = 0, 0, 0, 0

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

        # python, numpy and TensorFlow seeds for reproducibility
        tf.keras.utils.set_random_seed(self.random_seed)

        # Create the model with specified hyperparameters
        neural_network_cv = NeuralNetworkModel(
            input_shape=self.input_shape, output_shape=self.output_shape, layer_sizes=layer_sizes,
            activations=activations, dropout_rates=dropout_rate, kernel_regularizer=reg_function,
            num_conv_layers=num_conv_layers, num_filters=num_filters, pool_size=pool_size,
            kernel_size=kernel_size, use_batch_norm=True, use_initializer=True, use_dropout=True,
            use_init_skip_connections=False, use_inter_skip_connections=False, learning_rate=learning_rate,
            epochs=self.epochs
        )
        model = neural_network_cv.create_model()

        # Define the optimizer with a learning rate hyperparameter
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Compile the model with the specified optimizer and loss function
        model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())
        return model

    def tuner_searcher(self, max_trials):
        """
        Random search of the hyperparameters with Keras Tuner (objective: validation loss, 2 executions per trial,
        early stopping with patience 50). Previous trials in `outputs_path`/trials_bm_search are deleted.

        Args:
            max_trials (int): Number of hyperparameter sets tried.

        Returns:
            kerastuner.Tuner: The tuner after the search.
        """
        # Directory for storing trial information
        trial_dir = self.outputs_path + 'trials_bm_search'

        # Check if the trial directory already exists, if so, it will be deleted
        if os.path.exists(trial_dir):
            shutil.rmtree(trial_dir)

        tuner = RandomSearch(
            lambda hp: self.build_model(hp),  # Pass the dictionary as an argument
            objective='val_loss',
            max_trials=max_trials,
            executions_per_trial=2,
            directory=trial_dir
        )

        tuner.search(
            np.array(self.X_train), np.array(self.Y_train),
            epochs=self.epochs,
            validation_data=(np.array(self.X_valid), np.array(self.Y_valid)),
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)]
        )

        return tuner

    def bm_evaluation(self, tuner, units, var_x, var_y, months_x, months_y, predictor_region, n_cv_folds=0, cross_validation=False, threshold=0.1):
        """
        Rebuild the best model of the search, train it, and evaluate it on the test set or by cross-validation.

        Args:
            tuner (kerastuner.Tuner): Tuner after the search.
            units (str): Units of the predictand.
            var_x, var_y (str): Names of the predictor and predictand.
            months_x, months_y (list): Months of the predictor and predictand.
            predictor_region (str): Name of the predictor region.
            n_cv_folds (int, optional): Number of folds for cross-validation. Default is 0.
            cross_validation (bool, optional): Cross-validate instead of the single test evaluation. Default is False.
            threshold (float, optional): p-value threshold of the figures. Default is 0.1.

        Returns:
            (predicted, observed): Predictions and observations in the units of the predictand, for the test set or
            from the cross-validation. For cross-validation, `self.X` and `self.Y` must be the seasonal aggregates
            (see `ClimateDataEvaluation.cross_validation`).
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

        if not self.params_selection['pos_conv_layers'] == 0:
            num_conv_layers = best_hparams['#_convolutional_layers']
            number_of_filters_per_conv = best_hparams['number_of_filters_per_conv']
            pool_size = best_hparams['pool size']
            kernel_size = best_hparams['kernel size']
        else:
            num_conv_layers, number_of_filters_per_conv, pool_size, kernel_size = 0, 0, 0, 0

        print('Now creating and training the best model')
        start_time = time.time()
        neural_network_bm = NeuralNetworkModel(
            input_shape=self.input_shape, output_shape=self.output_shape, layer_sizes=units_list,
            activations=activations_list, dropout_rates=dropout_list, kernel_regularizer=kernel_regularizer,
            num_conv_layers=num_conv_layers, num_filters=number_of_filters_per_conv, pool_size=pool_size,
            kernel_size=kernel_size, use_batch_norm=batch_normalization, use_initializer=he_initialization,
            use_dropout=dropout, use_init_skip_connections=False, use_inter_skip_connections=False,
            learning_rate=learning_rate, epochs=self.epochs, random_seed=self.random_seed
        )

        if not cross_validation:
            model_bm, record = neural_network_bm.train_model(self.X_train, self.Y_train, self.X_valid, self.Y_valid, self.outputs_path)
            neural_network_bm.performance_plot(record)
            end_time = time.time()
            time_taken = end_time - start_time
            print(f'Training done (Time taken: {time_taken:.2f} seconds)')
            print('Now evaluating the best model on the test set')
            evaluations_toolkit_bm = ClimateDataEvaluation(
                self.X, self.X_train, self.X_test, self.Y, self.Y_train, self.Y_test, self.lon_y,
                self.lat_y, self.std_y, model_bm, self.time_lims, self.train_years, self.testing_years,
                self.output_original, jump_year=self.jump_year, detrend_x=self.detrend_x,
                detrend_x_window=self.detrend_x_window, detrend_y=self.detrend_y,
                detrend_y_window=self.detrend_y_window
            )
            predicted_value, observed_value = evaluations_toolkit_bm.evaluation()
            fig1 = evaluations_toolkit_bm.correlations(
                predicted_value, observed_value, self.outputs_path, threshold=threshold, units=units,
                months_x=months_x, months_y=months_y, var_x=var_x, var_y=var_y, predictor_region=predictor_region,
                best_model=True
            )
            return predicted_value, observed_value
        else:
            print('Now evaluating the best model via Cross Validation')
            evaluations_toolkit_bm = ClimateDataEvaluation(
                self.X, self.X_train, self.X_test, self.Y, self.Y_train, self.Y_test, self.lon_y,
                self.lat_y, self.std_y, None, self.time_lims, self.train_years, self.testing_years,
                self.output_original, jump_year=self.jump_year, detrend_x=self.detrend_x,
                detrend_x_window=self.detrend_x_window, detrend_y=self.detrend_y,
                detrend_y_window=self.detrend_y_window
            )
            predicted_global, correct_value, years_division_list = evaluations_toolkit_bm.cross_validation(
                n_folds=n_cv_folds, model_class=neural_network_bm
            )
            fig2 = evaluations_toolkit_bm.correlations(
                predicted_global, correct_value, self.outputs_path, threshold=threshold, units=units,
                months_x=months_x, months_y=months_y, var_x=var_x, var_y=var_y, predictor_region=predictor_region,
                best_model=True
            )
            fig3 = evaluations_toolkit_bm.correlations_pannel(
                n_folds=n_cv_folds, predicted_global=predicted_global, correct_value=correct_value,
                years_division=years_division_list, threshold=self.threshold, outputs_path=self.outputs_path,
                months_x=months_x, months_y=months_y, predictor_region=predictor_region, var_x=var_x,
                var_y=var_y, best_model=True
            )
            return predicted_global, correct_value

def Dictionary_saver(dictionary):
    """
    Save the hyperparameter dictionary as `<outputs_path>dict_hyperparms.yaml`, asking before overwriting.

    Args:
        dictionary (dict): Hyperparameters; must contain 'outputs_path'.

    Returns:
        None
    """
    output_file_path = dictionary['outputs_path'] + 'dict_hyperparms.yaml'

    # Check if the file already exists
    print('Checking if the file already exists in the current directory...')
    if os.path.isfile(output_file_path):
        overwrite_confirmation = input(f'The file {output_file_path} already exists. Do you want to overwrite it? (yes/no): ')
        if overwrite_confirmation.lower() != 'yes':
            print('Operation aborted. The existing file was not overwritten.')
            # Add further handling or exit the script if needed
            return

    # Write the dictionary to the YAML file
    with open(output_file_path, 'w') as yaml_file:
        yaml.dump(dictionary, yaml_file, default_flow_style=False)

    print(f'Dictionary saved to {output_file_path}')

def Preprocess(dictionary_hyperparams):
    """
    Preprocess the predictor and predictand and split them for the single train/validation/test evaluation.

    The predictor covers `time_lims` minus `jump_year` at the end and the predictand `time_lims` plus `jump_year` at
    the start. When any of the two variables drops months (a season crossing the year boundary), the aggregated
    series loses its last year. Both variables use `train_years` as reference period for their mean and standard
    deviation (see `ClimateDataPreprocessing.preprocess_data`).

    Args:
        dictionary_hyperparams (dict): Hyperparameters, with keys path_x/path_y, name_x/name_y, lat_lims_x/_y,
            lon_lims_x/_y, scale_x/_y, regrid_degree_x/_y, months_x/_y, months_skip_x/_y, detrend_x/_y,
            detrend_x_window/_y_window, mean_seasonal_method_x/_y, time_lims, jump_year, train_years,
            validation_years and testing_years.

    Returns:
        dict: {'input': {...}, 'output': {...}, 'data_split': {...}}. 'input' and 'output' contain lat, lon, data
        (seasonal aggregates), anomaly, normalized, mean and std; 'data_split' contains X, X_train, X_valid, X_test,
        Y, Y_train, Y_valid, Y_test, input_shape and output_shape (see `DataSplitter.prepare_data`).
    """
    print('Preprocessing the data')
    start_time = time.time()
    time_lims_x = [dictionary_hyperparams['time_lims'][0],dictionary_hyperparams['time_lims'][1]-dictionary_hyperparams['jump_year']]
    time_lims_y = [dictionary_hyperparams['time_lims'][0]+dictionary_hyperparams['jump_year'],dictionary_hyperparams['time_lims'][1]]

    # Output years: one year less if any of the two variables drops months (season crossing the year boundary)
    if dictionary_hyperparams['months_skip_x'] == ['None'] and dictionary_hyperparams['months_skip_y'] == ['None']:
        years_out = [dictionary_hyperparams['time_lims'][0], dictionary_hyperparams['time_lims'][1] - dictionary_hyperparams['jump_year']]
    else:
        years_out = [dictionary_hyperparams['time_lims'][0], dictionary_hyperparams['time_lims'][1] - 1 - dictionary_hyperparams['jump_year']]

    # Initialize ClimateDataPreprocessing for input and output data
    data_mining_x = ClimateDataPreprocessing(
        relative_path=dictionary_hyperparams['path_x'],
        lat_lims=dictionary_hyperparams['lat_lims_x'],
        lon_lims=dictionary_hyperparams['lon_lims_x'],
        time_lims=time_lims_x,
        scale=dictionary_hyperparams['scale_x'],
        regrid_degree=dictionary_hyperparams['regrid_degree_x'],
        variable_name=dictionary_hyperparams['name_x'],
        months=dictionary_hyperparams['months_x'],
        months_to_drop=dictionary_hyperparams['months_skip_x'],
        years_out=years_out,
        detrend=dictionary_hyperparams['detrend_x'],
        detrend_window=dictionary_hyperparams['detrend_x_window'],
        mean_seasonal_method=dictionary_hyperparams['mean_seasonal_method_x'],
        train_years=dictionary_hyperparams['train_years']
    )

    data_mining_y = ClimateDataPreprocessing(
        relative_path=dictionary_hyperparams['path_y'],
        lat_lims=dictionary_hyperparams['lat_lims_y'],
        lon_lims=dictionary_hyperparams['lon_lims_y'],
        time_lims=time_lims_y,
        scale=dictionary_hyperparams['scale_y'],
        regrid_degree=dictionary_hyperparams['regrid_degree_y'],
        variable_name=dictionary_hyperparams['name_y'],
        months=dictionary_hyperparams['months_y'],
        months_to_drop=dictionary_hyperparams['months_skip_y'],
        years_out=years_out,
        detrend=dictionary_hyperparams['detrend_y'],
        detrend_window=dictionary_hyperparams['detrend_y_window'],
        jump_year=dictionary_hyperparams['jump_year'],
        mean_seasonal_method=dictionary_hyperparams['mean_seasonal_method_y'],
        train_years=dictionary_hyperparams['train_years']
    )

    # Preprocess data
    lat_x, lon_x, data_x, anom_x, norm_x, mean_x, std_x = data_mining_x.preprocess_data()
    lat_y, lon_y, data_y, anom_y, norm_y, mean_y, std_y = data_mining_y.preprocess_data()

    # Split data into training, validation, and test sets
    data_splitter = DataSplitter(
        train_years=dictionary_hyperparams['train_years'],
        validation_years=dictionary_hyperparams['validation_years'],
        testing_years=dictionary_hyperparams['testing_years'],
        predictor=norm_x,
        predictant=norm_y,
        jump_year=dictionary_hyperparams['jump_year']
    )
    X, X_train, X_valid, X_test, Y, Y_train, Y_valid, Y_test, input_shape, output_shape = data_splitter.prepare_data()

    end_time = time.time()
    time_taken = end_time - start_time
    print(f'Preprocessing done (Time taken: {time_taken:.2f} seconds)')

    # Prepare results dictionary
    preprocessing_results = {
        'input': {
            'lat': lat_x,
            'lon': lon_x,
            'data': data_x,
            'anomaly': anom_x,
            'normalized': norm_x,
            'mean': mean_x,
            'std': std_x
        },
        'output': {
            'lat': lat_y,
            'lon': lon_y,
            'data': data_y,
            'anomaly': anom_y,
            'normalized': norm_y,
            'mean': mean_y,
            'std': std_y
        },
        'data_split': {
            'X': X,
            'X_train': X_train,
            'X_valid': X_valid,
            'X_test': X_test,
            'Y': Y,
            'Y_train': Y_train,
            'Y_valid': Y_valid,
            'Y_test': Y_test,
            'input_shape': input_shape,
            'output_shape': output_shape
        }
    }

    return preprocessing_results

def Model_build_and_test(dictionary_hyperparams, dictionary_preprocess, cross_validation=False, n_cv_folds=0, plot_differences=False, importances=False, region_importances=None, validation_fraction=0.0):
    """
    Build, train and evaluate a model, save figures and NetCDF outputs in `<outputs_path>data_outputs/`.

    Without cross-validation, one model is trained on `train_years` (early stopping on `validation_years`) and
    evaluated on `testing_years` ('predicted_test_period.nc', 'observed_test_period.nc'). With cross-validation, a
    model is trained in every fold (see `ClimateDataEvaluation.cross_validation`) and the cross-validated
    predictions, observations and (optionally) attributions are saved ('predicted_global_cv.nc' with dimension
    'time', 'observed_global_cv.nc' with dimension 'year', 'importances_region_cv.nc'). 'predictor_anomalies.nc'
    contains the predictor anomalies of `preprocess_data` (reference period `train_years`). All outputs are in the
    units of the variables.

    Args:
        dictionary_hyperparams (dict): Hyperparameters (layer_sizes, activations, dropout_rates, kernel_regularizer,
            num_conv_layers, use_batch_norm, use_initializer, use_dropout, use_init_skip_connections,
            use_inter_skip_connections, learning_rate, epochs, outputs_path, jump_year, p_value, units_y, name_x,
            name_y, months_x, months_y, region_predictor, time_lims, train_years, testing_years, detrend_x,
            detrend_x_window, detrend_y, detrend_y_window).
        dictionary_preprocess (dict): Output of `Preprocess`.
        cross_validation (bool, optional): Cross-validate instead of the single test evaluation. Default is False.
        n_cv_folds (int, optional): Number of folds. Default is 0.
        plot_differences (bool, optional): Per-fold ACC minus global ACC in the panel figure. Default is False.
        importances (bool, optional): Compute Integrated-Gradients attributions (cross-validation only). Default False.
        region_importances (list, optional): [lat_lims, lon_lims] of the attributed predicted points.
        validation_fraction (float, optional): Fraction of the training years of each fold held out for early stopping
            (cross-validation only). Default is 0 (previous behaviour).

    Returns:
        dict: {'predictions', 'observations'} and, with importances, {'importances', 'region_attributed'}.
    """
    print('Now creating and training the model')
    start_time = time.time()

    # Initialize and create the neural network model
    neural_network = NeuralNetworkModel(
        input_shape=dictionary_preprocess['data_split']['input_shape'],
        output_shape=dictionary_preprocess['data_split']['output_shape'],
        layer_sizes=dictionary_hyperparams['layer_sizes'],
        activations=dictionary_hyperparams['activations'],
        dropout_rates=dictionary_hyperparams['dropout_rates'],
        kernel_regularizer=dictionary_hyperparams['kernel_regularizer'],
        num_conv_layers=dictionary_hyperparams['num_conv_layers'],
        use_batch_norm=dictionary_hyperparams['use_batch_norm'],
        use_initializer=dictionary_hyperparams['use_initializer'],
        use_dropout=dictionary_hyperparams['use_dropout'],
        use_init_skip_connections=dictionary_hyperparams['use_init_skip_connections'],
        use_inter_skip_connections=dictionary_hyperparams['use_inter_skip_connections'],
        learning_rate=dictionary_hyperparams['learning_rate'],
        epochs=dictionary_hyperparams['epochs']
    )

    model, record = neural_network.train_model(
        dictionary_preprocess['data_split']['X_train'],
        dictionary_preprocess['data_split']['Y_train'],
        dictionary_preprocess['data_split']['X_valid'],
        dictionary_preprocess['data_split']['Y_valid'],
        dictionary_hyperparams['outputs_path']
    )

    # Initialize evaluation toolkit and perform evaluation
    evaluations_toolkit = ClimateDataEvaluation(
        dictionary_preprocess['input']['data'],
        dictionary_preprocess['data_split']['X_train'],
        dictionary_preprocess['data_split']['X_test'],
        dictionary_preprocess['output']['data'],
        dictionary_preprocess['data_split']['Y_train'],
        dictionary_preprocess['data_split']['Y_test'],
        dictionary_preprocess['output']['lon'],
        dictionary_preprocess['output']['lat'],
        dictionary_preprocess['output']['std'],
        model,
        dictionary_hyperparams['time_lims'],
        dictionary_hyperparams['train_years'],
        dictionary_hyperparams['testing_years'],
        dictionary_preprocess['output']['normalized'],
        jump_year=dictionary_hyperparams['jump_year'],
        detrend_x=dictionary_hyperparams['detrend_x'],
        detrend_x_window=dictionary_hyperparams['detrend_x_window'],
        detrend_y=dictionary_hyperparams['detrend_y'],
        detrend_y_window=dictionary_hyperparams['detrend_y_window'],
        importances=importances,
        region_atributted=region_importances
    )

    # Create output directory and save anomaly data
    output_directory = os.path.join(dictionary_hyperparams['outputs_path'], 'data_outputs')
    os.makedirs(output_directory, exist_ok=True)
    dictionary_preprocess['input']['anomaly'].to_netcdf(
        os.path.join(output_directory, 'predictor_anomalies.nc'),
        format='NETCDF4',
        mode='w',
        group='/',
        engine='netcdf4'
    )

    if not cross_validation:
        # Plot performance and save results for non-cross-validation mode
        neural_network.performance_plot(record)
        predicted_value, correct_value = evaluations_toolkit.evaluation()
        fig1 = evaluations_toolkit.correlations(
            predicted_value,
            correct_value,
            outputs_path=dictionary_hyperparams['outputs_path'],
            threshold=dictionary_hyperparams['p_value'],
            units=dictionary_hyperparams['units_y'],
            var_x=dictionary_hyperparams['name_x'],
            var_y=dictionary_hyperparams['name_y'],
            months_x=dictionary_hyperparams['months_x'],
            months_y=dictionary_hyperparams['months_y'],
            predictor_region=dictionary_hyperparams['region_predictor'],
            best_model=False
        )
        datasets = [predicted_value, correct_value]
        names = ['predicted_test_period.nc', 'observed_test_period.nc']
    else:
        # Perform cross-validation and save results
        if importances:
            predicted_value, correct_value, years_division_list, importances_region = evaluations_toolkit.cross_validation(
                n_folds=n_cv_folds,
                model_class=neural_network,
                validation_fraction=validation_fraction
            )
            datasets = [predicted_value, correct_value, importances_region]
            names = ['predicted_global_cv.nc', 'observed_global_cv.nc', 'importances_region_cv.nc']
        else:
            predicted_value, correct_value, years_division_list = evaluations_toolkit.cross_validation(
                n_folds=n_cv_folds,
                model_class=neural_network,
                validation_fraction=validation_fraction
            )
            datasets = [predicted_value, correct_value]
            names = ['predicted_global_cv.nc', 'observed_global_cv.nc']

        fig1 = evaluations_toolkit.correlations(
            predicted_value,
            correct_value,
            outputs_path=dictionary_hyperparams['outputs_path'],
            threshold=dictionary_hyperparams['p_value'],
            units=dictionary_hyperparams['units_y'],
            var_x=dictionary_hyperparams['name_x'],
            var_y=dictionary_hyperparams['name_y'],
            months_x=dictionary_hyperparams['months_x'],
            months_y=dictionary_hyperparams['months_y'],
            predictor_region=dictionary_hyperparams['region_predictor'],
            best_model=False
        )
        fig2 = evaluations_toolkit.correlations_pannel(
            n_folds=n_cv_folds,
            predicted_global=predicted_value,
            correct_value=correct_value,
            years_division=years_division_list,
            threshold=dictionary_hyperparams['p_value'],
            outputs_path=dictionary_hyperparams['outputs_path'],
            months_x=dictionary_hyperparams['months_x'],
            months_y=dictionary_hyperparams['months_y'],
            predictor_region=dictionary_hyperparams['region_predictor'],
            var_x=dictionary_hyperparams['name_x'],
            var_y=dictionary_hyperparams['name_y'],
            best_model=False,
            plot_differences=plot_differences
        )

    # Save each dataset to a NetCDF file
    for i, ds in enumerate(datasets, start=1):
        ds.to_netcdf(
            os.path.join(output_directory, names[i-1]),
            format='NETCDF4',
            mode='w',
            group='/',
            engine='netcdf4'
        )

    end_time = time.time()
    time_taken = end_time - start_time
    print(f'Training done (Time taken: {time_taken:.2f} seconds)')

    # Prepare model outputs dictionary
    if importances:
        model_outputs = {
            'predictions': predicted_value,
            'observations': correct_value,
            'importances': importances_region,
            'region_attributed': region_importances
        }
    else:
        model_outputs = {
            'predictions': predicted_value,
            'observations': correct_value
        }

    return model_outputs
