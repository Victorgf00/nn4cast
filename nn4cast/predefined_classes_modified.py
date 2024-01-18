#python file with all the functions created to the TFM model
import sys

# Add the directory containing the hyperparameters module or file to the search path
sys.path.append('C:/Users/ideapad 5 15ITL05/Desktop/TFM/Predicciones_y_comparacion_ecmwf/model_library/')

# Import the necessary libraries
from my_libraries import *
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
        rename (bool, optional): Rename latitude and longitude dimensions. Default is False.
        latitude_regrid (bool, optional): Perform latitude regridding. Default is False.
        months (list): List of months to select.
        years (list): List of years to select.
        months_to_drop (list): List of months to drop.
        years_out (list): List of output years.
        reference_period (list): Reference period for normalization [start_year, end_year].
        drop_months (bool, optional): Drop specified months from the selection. Default is False.
        detrend (bool, optional): Detrend the data using linear regression. Default is False.
        one_output (bool, optional): Return one output for each year. Default is False.

    Methods:
        preprocess_data(): Preprocess the climate data based on the specified parameters.
    """

    def __init__(
        self, relative_path, lat_lims, lon_lims, time_lims, scale=1, regrid=False,
        regrid_degree=2, overlapping=False, variable_name=None, rename=False, latitude_regrid=False,
        months=None, years=None, months_to_drop=None, years_out=None, reference_period=None,
        drop_months=False, detrend=False, one_output=False
    ):
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
        self.regrid = regrid
        self.regrid_degree = regrid_degree
        self.overlapping = overlapping
        self.variable_name = variable_name
        self.rename = rename
        self.latitude_regrid = latitude_regrid
        self.months = months
        self.years = years
        self.months_to_drop = months_to_drop
        self.years_out = years_out
        self.reference_period = reference_period
        self.drop_months = drop_months
        self.detrend = detrend
        self.one_output = one_output

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

        if self.rename:
            data = data.rename({'lat': 'latitude', 'lon': 'longitude'})

        if self.latitude_regrid:
            lat_newgrid = np.arange(-89, 90, 1)
            data = data.interp(latitude=np.array(lat_newgrid))
            data = data.sortby('latitude', ascending=False)

        if self.lon_lims[1] > 180:
            data = data.assign_coords(longitude=np.where(data.longitude < 0, 360 + data.longitude, data.longitude)).sortby('longitude')
        else:
            data = data.assign_coords(longitude=(((data.longitude + 180) % 360) - 180)).sortby('longitude')

        data = data.sel(latitude=slice(self.lat_lims[0], self.lat_lims[1]), longitude=slice(self.lon_lims[0], self.lon_lims[1]), time=slice(str(self.time_lims[0]), str(self.time_lims[1])))

        if self.regrid:
            lon_regrid = np.arange(self.lon_lims[0], self.lon_lims[1], self.regrid_degree)
            lat_regrid = np.arange(self.lat_lims[0], self.lat_lims[1], -self.regrid_degree)
            data = data.interp(longitude=np.array(lon_regrid)).interp(latitude=np.array(lat_regrid))

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

        data_red = data.sel(time=slice(f"{self.years[0]}", f"{self.years[-1]}"))
        data_red = data_red.sel(time=np.isin(data_red['time.month'], self.months))

        if self.drop_months:
            if self.months_to_drop != ['None']:
                data_red = data_red.drop_sel(time=self.months_to_drop)

        data_red = data_red.groupby('time.month')
        mean_data = 0
        for i in self.months:
            mean_data += np.array(data_red[i])
        mean_data /= len(self.months)

        if self.one_output:
            data_red = xr.DataArray(data=mean_data, dims=["year"], coords=dict(year=self.years_out))
        else:
            data_red = xr.DataArray(
                data=mean_data, dims=["year", "latitude", "longitude"],
                coords=dict(
                    longitude=(["longitude"], np.array(data.longitude)),
                    latitude=(["latitude"], np.array(data.latitude)), year=self.years_out
                )
            )

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

    def create_model(self):
        """
        Create a customizable deep neural network model.

        Returns:
            model (tf.keras.Model): Customizable deep neural network model.
            fig (matplotlib.figure.Figure): Figure showing the model architecture.
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
        fig= tf.keras.utils.plot_model(model,show_shapes=True,show_dtype=False,show_layer_names=True,rankdir='LR',expand_nested=False,dpi=50,layer_range=None,show_layer_activations=True)
        return model, fig

    def train_model(self, X_train, Y_train, X_valid, Y_valid):
        """
        Train the created model.

        Args:
            - X_train (numpy.ndarray): Training input data.
            - Y_train (numpy.ndarray): Training output data.
            - X_valid (numpy.ndarray): Validation input data.
            - Y_valid (numpy.ndarray): Validation output data.

        Returns:
            history (tf.keras.callbacks.History): Training history of the model.
        """
        # Set random seeds for reproducibility
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        tf.compat.v1.random.set_random_seed(self.random_seed)

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
        model, fig = self.create_model()
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
        - years (numpy.ndarray): Years for the climate data.
        - train_years (tuple): Years for training data.
        - testing_years (tuple): Years for testing data.
        - jump_year (int): Year offset for the predictions (default: 0).

    Methods:
        - plotter(): Tool for plotting model outputs.
        - prediction_vs_observation1D(): Plot the model's predictions vs. observations for both training and testing data.
        - evaluation(): Evaluate a trained model's predictions against test data.
        - correlations(): Calculate and visualize various correlation and RMSE metrics for climate predictions.
        - cross_validation(): Perform cross-validation for the climate prediction model.
        - correlations_pannel(): Visualize correlations for each ensemble member in a panel plot.
    """
    def __init__(
        self, X, X_train, X_test, Y, Y_train, Y_test, lon_y, lat_y, std_y, model, years, train_years, testing_years, jump_year=0):
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
        self.years = years
        self.train_years = train_years
        self.testing_years = testing_years
        self.jump_year = jump_year

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
        nt, nm= predicted.shape
        test_years = np.arange(testing_years[0]+self.jump_year, testing_years[1]+self.jump_year+1)
        if np.ndim(predicted.shape)<=2:
            predicted= np.reshape(predicted, (nt, len(np.array(self.lat_y)), len(np.array(self.lon_y))))
        
        # Create xarray DataArray for predicted values
        predicted = xr.DataArray(
            data=predicted,
            dims=["time", "latitude", "longitude"],
            coords=dict(
                longitude=(["longitude"], np.array(self.lon_y)),
                latitude=(["latitude"], np.array(self.lat_y)),
                time=test_years))

        correct_value = xr.DataArray(
            data=np.reshape(Y_test, (nt, len(np.array(self.lat_y)), len(np.array(self.lon_y)))),
            dims=["year", "latitude", "longitude"],
            coords=dict(
                longitude=(["longitude"], np.array(self.lon_y)),
                latitude=(["latitude"], np.array(self.lat_y)),
                year=test_years))

        # De-standardize the predicted and true values to obtain anomalies
        predicted = predicted * self.std_y
        correct_value = correct_value * self.std_y
        
        return predicted, correct_value

    def correlations(self, predicted, correct_value, outputs_path, threshold, units, titulo1, titulo2, titulo_corr, periodo):
        """
        Calculate and visualize various correlation and RMSE metrics for climate predictions.

        Args:
            - predicted (xr.DataArray): Predicted sea-level pressure anomalies.
            - correct_value (xr.DataArray): True sea-level pressure anomalies.
            - outputs_path (str): Output path for saving the plot.
            - threshold (float): Threshold for significance in correlation.
            - units (str): Units for the plot.
            - titulo1 (str): Title for the first part of the plots.
            - titulo2 (str): Title for the second part of the plots.
            - titulo_corr (str): Title specifying the correlation type.
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
        ax.scatter(lon_sig, lat_sig, s=5, c='k', marker='.', alpha=0.5, transform=ccrs.PlateCarree(), label='Significant')

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
        rango= int(np.max(np.array(data)))
        ClimateDataEvaluation.plotter(self, data, np.arange(0, rango+1, 1), 'OrRd','RMSE', 'Spatial RMSE', ax4, pixel_style=True)
        
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
        fig.suptitle(f'Metrics {titulo1} when predicting with {titulo_corr} {periodo} {titulo2}', fontsize=20)
        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig(outputs_path + 'correlations')
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

        # Loop over the folds
        for i, (train_index, testing_index) in enumerate(kf.split(self.X)):
            print(f'Fold {i+1}/{n_folds}')
            print('Training on:',self.years[train_index])
            print('Testing on:', self.years[testing_index])

            # Get the training and validation data for this fold
            X_train_fold = self.X[train_index]
            Y_train_fold = self.Y[train_index]
            X_testing_fold = self.X[testing_index]
            Y_testing_fold = self.Y[testing_index]

            model_cv, fig = model_class.create_model()
            model_cv, record= model_class.train_model(X_train=X_train_fold, Y_train=Y_train_fold, X_valid=X_train_fold, Y_valid=Y_train_fold)
            predicted_value,observed_value= ClimateDataEvaluation.evaluation(self, X_test_other=X_testing_fold, Y_test_other=Y_testing_fold, model_other=model_cv, years_other=[(self.years[testing_index])[0],(self.years[testing_index])[-1]])
            predicted_list.append(predicted_value)
            correct_value_list.append(observed_value)


        # concatenate all the predicted values in the list into one global dataarray
        predicted_global = xr.concat(predicted_list, dim='time')
        correct_value = xr.concat(correct_value_list, dim='year')
        return predicted_global,correct_value

    def correlations_pannel(self, n_folds, predicted_global, correct_value, months_x, months_y, titulo_corr):
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
        time_range = int((len(self.years)/n_folds))

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
                predictions_loop = predictions_member.sel(time=slice(self.years[0]+i*time_range,self.years[time_range]+i*time_range))
                spatial_correlation_member = xr.corr(predictions_loop, correct_value, dim='time')

                # Plot the correlation map
                ax = axes[i]
                data_member = spatial_correlation_member-spatial_correlation_global
                rango=1
                im= ClimateDataEvaluation.plotter(self, data=data_member,levs=np.linspace(-rango, +rango, 11), cmap1='PiYG_r', l1='Correlation', titulo='Model tested in '+str(i+1)+': '+str(self.years[0]+i*time_range)+'-'+str(self.years[time_range]+i*time_range), ax=ax, plot_colorbar=False)

                if i==n_folds-1:
                    rango=1
                    # Plot the correlation map
                    ax = axes[i+1]
                    data_member = spatial_correlation_global
                    im2= ClimateDataEvaluation.plotter(self, data=data_member,levs=np.linspace(-rango, +rango, 11), cmap1='RdBu_r', l1='Correlation', titulo='ACC Global', ax=ax, plot_colorbar=False)

        # Add a common colorbar for all subplots

        cbar_ax = fig.add_axes([0.92, 0.35, 0.02, 0.5])  # Adjust the position for your preference
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical', label='Correlation difference: ACC_member-ACC_global', format="%2.1f")

        cbar_ax = fig.add_axes([0.92, 0.025, 0.02, 0.3])  # Adjust the position for your preference
        cbar = fig.colorbar(im2, cax=cbar_ax, orientation='vertical', label='Correlation', format="%2.1f")

        # Add a common title for the entire figure
        fig.suptitle(f'Correlations for predicting each time period for SLP months "{months_y}" \n with months "{months_x}" {titulo_corr} SST.',fontsize=18)

        # Adjust layout for better spacing
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])

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
        - years (array): Full set of years.
        - train_years (list): Years used for training.
        - testing_years (list): Years used for testing.
        - jump_year (int, optional): Time interval, default is zero.
        - params_selection (dictionary): Dictionary containing hyperparameter search space.
        - epochs (int): Number of training epochs.
        - random_seed (int, optional): Seed for random number generation, default is 42.
        - outputs_path (directory): Path to store the outputs.

    Methods:
        - build_model(): Build a deep learning model with hyperparameters defined by the Keras Tuner.
        - tuner_searcher(): Perform hyperparameter search using Keras Tuner.
        - bm_evaluation(): Evaluate the best model though different methods.
    """

    def __init__(
        self, input_shape, output_shape, X, X_train,X_valid, X_test, Y, Y_train, Y_valid, Y_test, lon_y, lat_y, std_y, years, train_years, testing_years, params_selection, epochs, 
        outputs_path, random_seed=42, jump_year=0):
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
        self.years = years
        self.train_years = train_years
        self.testing_years = testing_years
        self.jump_year = jump_year
        self.params_selection = params_selection
        self.epochs = epochs
        self.random_seed = random_seed
        self.outputs_path = outputs_path

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

        # Choose whether to use dropout
        use_dropout = hp.Choice('dropout', ['True','False'])
        print("Use dropout:", use_dropout)

        #Define the learning rate
        learning_rate=hp.Choice('learning_rate', pos_learning_rate)

        # If dropout is chosen, define dropout rates for hidden layers
        if use_dropout=='True':
            dropout_rates = [hp.Choice('dropout_' + str(i), pos_dropout) for i in range(len(layer_sizes)-1)]
            print("Dropout rates:", dropout_rates)
        else:
            dropout_rates = [0.0]

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

        if not dropout_rates:
            dropout_rates = [0.0]

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
        neural_network_cv = NeuralNetworkModel(input_shape=self.input_shape, output_shape=self.output_shape, layer_sizes=layer_sizes, activations=activations, dropout_rates=dropout_rates, kernel_regularizer=reg_function,
                                        num_conv_layers=num_conv_layers, num_filters=num_filters, pool_size=pool_size, kernel_size=kernel_size, use_batch_norm=True, use_initializer=True, use_dropout=True, use_initial_skip_connections=False, use_intermediate_skip_connections=False, learning_rate=learning_rate,
                                        epochs=self.epochs)
        model, fig = neural_network_cv.create_model()
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

        # Save the best model found by the tuner
        best_model = tuner.get_best_models(num_models=1)[0]
        best_model.save(self.outputs_path+'best_model.h5')  # Save the model in HDF5 format
        fig= tf.keras.utils.plot_model(best_model,show_shapes=True,show_dtype=False,show_layer_names=True,rankdir='LR',expand_nested=False,dpi=50,layer_range=None,show_layer_activations=True)
        return tuner, best_model, fig
    
    def bm_evaluation(self, tuner, units, titulo1, titulo2, titulo_corr, periodo, months_x, months_y, n_cv_folds=0, cross_validation=False):
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
        if dropout==True:
            dropout_list = [best_hparams[f'dropout_{i}'] for i in range(num_layers)]
        else:
            dropout_list=[]
            
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
        model_bm, fig = neural_network_bm.create_model()
        if cross_validation==False:
            model_bm, record= neural_network_bm.train_model(self.X_train, self.Y_train, self.X_valid, self.Y_valid)
            neural_network_bm.performance_plot(record)

            end_time = time.time()
            time_taken = end_time - start_time
            print(f'Training done (Time taken: {time_taken:.2f} seconds)')

            print('Now evalutating the best model on the test set')
            evaluations_toolkit_bm= ClimateDataEvaluation(self.X, self.X_train, self.X_test, self.Y, self.Y_train, self.Y_test, self.lon_y, self.lat_y, self.std_y, model_bm, self.years, self.train_years, self.testing_years, jump_year=self.jump_year)
            predicted_value,observed_value= evaluations_toolkit_bm.evaluation()
            fig1= evaluations_toolkit_bm.correlations(predicted_value,observed_value,self.outputs_path, threshold=0.1, units=units, titulo1=titulo1, titulo2=titulo2, periodo=periodo, titulo_corr=titulo_corr)
            return predicted_value, observed_value, fig1
        else:
            print('Now evalutating the best model via Cross Validation')
            evaluations_toolkit_bm= ClimateDataEvaluation(self.X, self.X_train, self.X_test, self.Y, self.Y_train, self.Y_test, self.lon_y, self.lat_y, self.std_y, model_bm, self.years, self.train_years, self.testing_years, jump_year=0)
            predicted_global,correct_value= evaluations_toolkit_bm.cross_validation(n_folds=n_cv_folds, model_class=neural_network_bm)
            fig2= evaluations_toolkit_bm.correlations(predicted_global,correct_value,self.outputs_path, threshold=0.1, units=units, titulo1=titulo1, titulo2=titulo2, periodo=periodo, titulo_corr=titulo_corr)
            fig3= evaluations_toolkit_bm.correlations_pannel(n_folds=n_cv_folds,predicted_global=predicted_global, correct_value=correct_value, months_x=months_x, months_y=months_y, titulo_corr=titulo_corr)
            return predicted_global, correct_value, fig2, fig3



