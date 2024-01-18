#python file with all the functions created to the TFM model
import sys

# Add the directory containing the hyperparameters module or file to the search path
sys.path.append('C:/Users/ideapad 5 15ITL05/Desktop/TFM/Predicciones_y_comparacion_ecmwf/model_library/')

# Import the necessary libraries
from my_libraries import *
plt.style.use('seaborn-v0_8')

def data_mining(relative_path, lat_lims, lon_lims, time_lims, scale=1, regrid=False, regrid_degree=2, overlapping=False, variable_name=None, rename=False, latitude_regrid=False):
    """
    Perform data preprocessing and manipulation on climate data.

    Args:
    - relative_path (str): The relative path to the NetCDF climate data file.
    - lat_lims (tuple): A tuple containing the latitude limits (min, max) for the selected region.
    - lon_lims (tuple): A tuple containing the longitude limits (min, max) for the selected region.
    - time_lims (tuple): A tuple containing the time limits (start, end) for the selected time period.
    - scale (float, optional): A scaling factor to apply to the data. Default is 1.
    - regrid (bool, optional): Perform grid regridding. Default is False.
    - regrid_degree (int, optional): The degree of grid regridding. Default is 2.
    - overlapping (bool, optional): Create a cyclic point for overlapping data. Default is False.
    - variable_name (str, optional): The name of the variable to extract. Default is None.
    - rename (bool, optional): Rename latitude and longitude dimensions. Default is False.
    - latitude_regrid (bool, optional): Perform latitude regridding. Default is False.

    Returns:
    - data (xarray.DataArray): The preprocessed climate data.
    - latitude (xarray.DataArray): The latitude coordinates.
    - longitude (xarray.DataArray): The longitude coordinates.
    """
    data = xr.open_dataset(relative_path) / scale
    time = data['time'].astype('datetime64[M]')
    data = data.assign_coords(time=time)
    
    if rename:
        data = data.rename({'lat': 'latitude', 'lon': 'longitude'})
    
    if latitude_regrid:
        lat_newgrid = np.arange(-89, 90, 1)
        data = data.interp(latitude=np.array(lat_newgrid))
        data = data.sortby('latitude', ascending=False)
    
    # Automatically adjust longitude handling based on lon_lims
    if lon_lims[1] > 180:
        data = data.assign_coords(longitude=np.where(data.longitude < 0, 360 + data.longitude, data.longitude)).sortby('longitude')
    else:
        data = data.assign_coords(longitude=(((data.longitude + 180) % 360) - 180)).sortby('longitude')

    data = data.sel(latitude=slice(lat_lims[0], lat_lims[1]), longitude=slice(lon_lims[0], lon_lims[1]), time=slice(str(time_lims[0]), str(time_lims[1])))
    
    if regrid:
        lon_regrid = np.arange(lon_lims[0], lon_lims[1], regrid_degree)
        lat_regrid = np.arange(lat_lims[0], lat_lims[1], -regrid_degree) 
        data = data.interp(longitude=np.array(lon_regrid)).interp(latitude=np.array(lat_regrid))
    
    latitude = data.latitude
    longitude = data.longitude
    data = data[str(variable_name)]
    
    if overlapping:
        creg, longitude = add_cyclic_point(np.array(data), coord=longitude)
        data = xr.DataArray(
            data=creg,
            dims=["time", "latitude", "longitude"],
            coords=dict(
                longitude=(["longitude"], np.array(longitude)),
                latitude=(["latitude"], np.array(latitude)),
                time=np.array(data.time)))
    
    return data, latitude, longitude

def data_selection(data, months, years, months_to_drop, years_out, reference_period, variable_name= None,
                   drop_months=False, detrend=False, one_output=False ):
    """
    Select, preprocess, and normalize data for specific months and years.

    Parameters:
    - data (xarray.DataArray): Input data.
    - months (list): List of months to select.
    - years (list): List of years to select.
    - months_to_drop (list): List of months to drop.
    - years_out (list): List of output years.
    - reference_period (list): Reference period for normalization [start_year, end_year].
    - drop_months (bool, optional): Drop specified months from the selection (default=False).
    - detrend (bool, optional): Detrend the data using linear regression (default=False).

    Returns:
    - data_red (xarray.DataArray): Selected and preprocessed data.
    - anomaly (xarray.DataArray): Anomaly data (detrended and normalized).
    - normalization (xarray.DataArray): Normalized data.
    - mean_reference (xarray.DataArray): Mean reference data for the reference period.
    - std_reference (xarray.DataArray): Standard deviation reference data for the reference period.
    """
    # Select years and months
    data_red = data.sel(time=slice(f"{years[0]}", f"{years[-1]}"))
    data_red = data_red.sel(time=np.isin(data_red['time.month'], months))

    # Drop specified months if requested
    if drop_months:
        if months_to_drop!=['None']: #this is done as a double check to make sure there are months to drop and not only the boolean selected as True
            data_red = data_red.drop_sel(time=months_to_drop)

    # Calculate monthly means
    data_red = data_red.groupby('time.month')
    mean_data = 0
    for i in months:
        mean_data += np.array(data_red[i])
    mean_data /= len(months)
    # Create a DataArray for the mean data
    if one_output:
        data_red = xr.DataArray(data=mean_data, dims=["year"],
                            coords=dict(year=years_out))
    else:
        data_red = xr.DataArray(data=mean_data, dims=["year", "latitude", "longitude"],
                            coords=dict(longitude=(["longitude"], np.array(data.longitude)),
                                        latitude=(["latitude"], np.array(data.latitude)), year=years_out))

    # Detrend data if requested
    if detrend:
        print('Detrending '+str(variable_name)+' data...')
        adjust = (data_red.polyfit(dim='year', deg=1)).sel(degree=1).polyfit_coefficients
        time_step = np.arange(len(data_red['year']))
        adjust_expanded = adjust.expand_dims(year=data_red['year'])
        if one_output:
            data_red = data_red - adjust_expanded * time_step
        else:
            time_step_expanded = np.expand_dims(time_step, axis=(1, 2))
            data_red = data_red - adjust_expanded * time_step_expanded

    # Calculate mean and standard deviation reference for normalization
    mean_reference = (data_red.sel(year=slice(str(reference_period[0]), str(reference_period[1])))).mean(dim='year')
    std_reference = (data_red.sel(year=slice(str(reference_period[0]), str(reference_period[1])))).std(dim='year')

    # Calculate anomaly and normalization
    anomaly = data_red - mean_reference
    normalization = anomaly / std_reference

    return data_red, anomaly, normalization, mean_reference, std_reference

def plotter(A, lon, lat, levs, cmap1, l1, titulo, ax, pixel_style=False):
    """
    Create a filled contour map using Cartopy.

    Parameters:
    - A (2D array-like): Data values to be plotted.
    - lon (1D array-like): Longitudes for the data grid.
    - lat (1D array-like): Latitudes for the data grid.
    - levs (list): Contour levels.
    - cmap1 (Colormap): Matplotlib colormap for the plot.
    - l1 (str): Label for the colorbar.
    - titulo (str): Title for the plot.
    - ax (matplotlib.axes.Axes): Matplotlib axis to draw the plot on.
    - pixel_style (bool): Whether to use a pcolormesh to plot the map.
    Returns:
    - fig

    This function creates a filled contour plot using Cartopy, displaying geographical data.
    """
    # Create a filled contour plot on the given axis
    if pixel_style==True:
        # Create a BoundaryNorm object
        cmap1 = plt.cm.get_cmap(cmap1)
        norm = colors.BoundaryNorm(levs, ncolors=cmap1.N, clip=True)

        # Create a pixel-based colormap plot on the given axis
        im = ax.pcolormesh(lon, lat, A, cmap=cmap1, transform=ccrs.PlateCarree(), norm=norm)  # 'norm' defines custom levels
    else:
        im = ax.contourf(lon, lat, A, cmap=cmap1, levels=levs, extend='both', transform=ccrs.PlateCarree())

    # Add coastlines to the plot
    ax.coastlines(linewidth=0.75)

    # Set the title for the plot
    ax.set_title(titulo, fontsize=18)

    # Create a colorbar for the plot
    cbar = plt.colorbar(im, extend='neither', spacing='proportional', orientation='vertical', shrink=0.7, format="%2.1f")
    cbar.set_label(l1, size=15)
    cbar.ax.tick_params(labelsize=15)
    gl = ax.gridlines(draw_labels=True)
    gl.xlines = False
    gl.ylines = False
    gl.top_labels = False  # Disable top latitude labels
    gl.right_labels = False  # Disable right longitude labels

def data_split(train_years,validation_years,testing_years,predictor,predictant,jump_year=0, replace_nans_with_0_predictor=False,replace_nans_with_0_predictant=False):
    """
    Prepares the data for training the model.

    Parameters:
    -----------
    train_years : list
        list containing the start and end years of the training data
    validation_years : list
        list containing the start and end years of the validation data
    testing_years : list
        list containing the start and end years of the test data
    predictor : xarray datarray
        dataarray containing the predictor data
    predictant : xarray datarray
        dataarray containing the predictant data
    jump_year (int, optional) : This integer specifies the number of years of difference from the predictor and the predictand in each sample, the default is 0 (ie, same year for both). 
    replace_nans_with_0_predictor (bool, optional): Substitute nan values with 0 from the predictor. Default is False, which delete every nan (value along time dimension) and reshapes to 2D (time, space)
    replace_nans_with_0_predictant (bool, optional): Substitute nan values with 0 from the predictant. Default is False, which delete every nan (value along time dimension) and reshapes to 2D (time, space)


    Returns:
    --------
    X_train : numpy array
        numpy array containing the cleaned SST data for the training data
    X_valid : numpy array
        numpy array containing the cleaned SST data for the validation data
    X_test : numpy array
        numpy array containing the cleaned SST data for the test data
    Y_train : xarray DataArray
        xarray DataArray containing the SLP data for the training data
    Y_valid : xarray DataArray
        xarray DataArray containing the SLP data for the validation data
    Y_test : xarray DataArray
        xarray DataArray containing the SLP data for the test data
    """
    X_train= predictor.sel(year=slice(train_years[0],train_years[1]))
    X_valid= predictor.sel(year=slice(validation_years[0],validation_years[1]))
    X_test= predictor.sel(year=slice(testing_years[0],testing_years[1]))

    Y_train= predictant.sel(year=slice(train_years[0]+jump_year,train_years[1]+jump_year))
    Y_valid= predictant.sel(year=slice(validation_years[0]+jump_year,validation_years[1]+jump_year))
    Y_test= predictant.sel(year=slice(testing_years[0]+jump_year,testing_years[1]+jump_year))

    #if our data has nans, we must remove it to avoid problems when training the model
    def quitonans(mat, reference):
        out = mat[:,~np.isnan(reference.mean(axis = 0))]
        return out
    
    if replace_nans_with_0_predictor==True:
            predictor= predictor.fillna(value=0)
            predictor= predictor.where(np.isfinite(predictor), 0)
            X_train= predictor.sel(year=slice(train_years[0],train_years[1]))
            X_valid= predictor.sel(year=slice(validation_years[0],validation_years[1]))
            X_test= predictor.sel(year=slice(testing_years[0],testing_years[1]))
    else:
            nt,nlat,nlon= predictor.shape
            nt_train,nlat,nlon= X_train.shape
            nt_valid,nlat,nlon= X_valid.shape
            nt_test,nlat,nlon= X_test.shape

            X= np.reshape(np.array(predictor), (nt, nlat*nlon))
            X_train_reshape= np.reshape(np.array(X_train), (nt_train, nlat*nlon))
            X_valid_reshape= np.reshape(np.array(X_valid), (nt_valid, nlat*nlon))
            X_test_reshape= np.reshape(np.array(X_test), (nt_test, nlat*nlon))

            X_train= quitonans(X_train_reshape,X)
            X_valid= quitonans(X_valid_reshape,X)
            X_test= quitonans(X_test_reshape,X)

    if replace_nans_with_0_predictant==True:
            predictant= predictant.fillna(value=0)
            Y_train= predictant.sel(year=slice(train_years[0]+jump_year,train_years[1]+jump_year))
            Y_valid= predictant.sel(year=slice(validation_years[0]+jump_year,validation_years[1]+jump_year))
            Y_test= predictant.sel(year=slice(testing_years[0]+jump_year,testing_years[1]+jump_year))

    else:
            nt,nlat,nlon= predictant.shape
            nt_train,nlat,nlon= Y_train.shape
            nt_valid,nlat,nlon= Y_valid.shape
            nt_test,nlat,nlon= Y_test.shape

            Y= np.reshape(np.array(predictant), (nt, nlat*nlon))
            Y_train_reshape= np.reshape(np.array(Y_train), (nt_train, nlat*nlon))
            Y_valid_reshape= np.reshape(np.array(Y_valid), (nt_valid, nlat*nlon))
            Y_test_reshape= np.reshape(np.array(Y_test), (nt_test, nlat*nlon))

            Y_train= quitonans(Y_train_reshape,Y)
            Y_valid= quitonans(Y_valid_reshape,Y)
            Y_test= quitonans(Y_test_reshape,Y)

    return X_train,X_valid,X_test,Y_train,Y_valid,Y_test

def create_model(input_shape, output_shape, layer_sizes, activations, dropout_rates=None, kernel_regularizer=None, num_conv_layers=0, num_filters=32, pool_size=2, kernel_size=3,
                 use_batch_norm=False, use_initializer=False, use_dropout=False, use_initial_skip_connections=False, use_intermediate_skip_connections=False, one_output=False):
    """
    Create a customizable deep neural network model.

    Parameters:
    - input_shape (tuple): The shape of the input data.
    - output_shape (int): The number of output units.
    - layer_sizes (list): List of layer sizes, excluding the output layer.
    - activations (list): List of activation functions for each layer.
    - dropout_rates (list, optional): List of dropout rates for each layer (default=None).
    - kernel_regularizer (tf.keras.regularizers, optional): Regularizer for kernel weights (default=None).
    - num_conv_layers (int, optional): Number of convolutional layers (default=2).
    - num_filters (list, optional): List of filter counts for each convolutional layer (default=32).
    - pool_size (list, optional): List of pool sizes for each convolutional layer (default=2).
    - kernel_size (list, optional): List of kernel sizes for each convolutional layer (default=3).
    - use_batch_norm (bool, optional): Include batch normalization layers (default=False).
    - use_initializer (bool, optional): Use He initialization for weights (default=False).
    - use_dropout (bool, optional): Include dropout layers (default=False).
    - use_initial_skip_connections (bool, optional): Include one skip connection from the beginning (default=False).
    - use_intermediate_skip_connections (bool, optional): Include skip connections between fully connected layers (default=False).

    Returns:
    - model (tf.keras.Model): Customizable deep neural network model.
    """
    # Define the input layer
    inputs = tf.keras.layers.Input(shape=input_shape, name="input_layer")
    x = inputs
    
    # Store skip connections
    skip_connections = []
    skip_connections.append(x)
    
    # Add convolutional layers with skip connections
    for i in range(num_conv_layers):
        x = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), padding='same', kernel_initializer='he_normal', name=f"conv_layer_{i+1}")(x)
        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization(name=f"batch_norm_{i+1}")(x)
        x = tf.keras.layers.Activation('elu', name=f'activation_{i+1}')(x)
        # Store the current layer as a skip connection
        x = tf.keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size), name=f"max_pooling_{i+1}")(x)

    # Flatten the output
    x = tf.keras.layers.Flatten()(x)
    
    # Add the output layer
    if kernel_regularizer is not "None":
        #x = tf.keras.layers.Dropout(dropout_rates[0], name=f"dropout_{0}")(x)
        x = tf.keras.layers.Dense(units=layer_sizes[0], activation=activations[0], kernel_regularizer=tf.keras.regularizers.L2(0.01), kernel_initializer='he_normal', name="dense_wit_reg_"+str(kernel_regularizer))(x)
    
    # Add fully connected layers
    for i in range(len(layer_sizes) - 1):
        # Store the current layer as a skip connection
        skip_connections.append(x)
        # Add a dense layer with He initialization if specified
        if use_initializer:
            x = tf.keras.layers.Dense(units=layer_sizes[i], kernel_initializer='he_normal', name=f"dense_{i+1}")(x)
        else:
            x = tf.keras.layers.Dense(units=layer_sizes[i], name=f"dense_{i+1}")(x)
        
        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization(name=f"batch_norm_{i+1+num_conv_layers}")(x)
            
        x = tf.keras.layers.Activation(activations[i], name=f'activation_{i+1+num_conv_layers}')(x)
        
        # Add dropout if applicable
        if use_dropout and i < len(dropout_rates):
            x = tf.keras.layers.Dropout(dropout_rates[i], name=f"dropout_{i+1}")(x)
        
        # Merge with the skip connection if specified
        if use_intermediate_skip_connections:
            skip_connection_last = skip_connections[-1]
            skip_connection_last = tf.keras.layers.Dense(units=layer_sizes[i], kernel_initializer='he_normal', name=f"dense_skip_connect_{i+1}")(skip_connection_last)
            x = tf.keras.layers.Add(name=f"merge_skip_connect_{i+1}")([x, skip_connection_last])

   # Add the output layer
    if kernel_regularizer is not "None":
        x = tf.keras.layers.Dense(units=layer_sizes[-1], activation=activations[-1], kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00, l2=0.2), kernel_initializer='he_normal', name="dense_wit_reg2_"+str(kernel_regularizer))(x)

    if use_initial_skip_connections:
        skip_connection_first= tf.keras.layers.Flatten()(skip_connections[0])
        skip_connection_first = tf.keras.layers.Dense(units=layer_sizes[-1], kernel_initializer='he_normal', name=f"initial_skip_connect")(skip_connection_first)
        x = tf.keras.layers.Add(name=f"merge_skip_connect")([x, skip_connection_first])
        
    # Define the output layer
    if np.ndim(output_shape)==0:
        outputs = tf.keras.layers.Dense(units=output_shape, kernel_initializer='he_normal', name=f"output_layer")(x) 
    else: 
        outputs = tf.keras.layers.Dense(units=output_shape[0]*output_shape[1], kernel_initializer='he_normal', name=f"output_layer")(x)
        outputs = tf.keras.layers.Reshape(output_shape)(outputs)


    # Create and summarize the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    fig= tf.keras.utils.plot_model(model,show_shapes=True,show_dtype=False,show_layer_names=True,rankdir='LR',expand_nested=False,dpi=50,layer_range=None,show_layer_activations=True)
    return model, fig

def performance_plot(record1):
    """
    Plot the training and validation loss over epochs for a given model.

    Parameters:
    - record1 (tf.keras.callbacks.History): Training history of the model.

    Returns:
    - None
    """
    
    # Create a figure and subplot
    fig1 = plt.figure(figsize=(10, 5))
    ax1 = fig1.add_subplot(111)
    
    # Set the figure title
    fig1.suptitle('Models Performance')
    
    # Plot the training and validation loss
    ax1.plot(record1.history['loss'], label='Training Loss')
    ax1.plot(record1.history['val_loss'], label='Validation Loss')
    
    # Set subplot title, labels, and legend
    ax1.set_title('Model 1')
    ax1.set_xlabel('# Epochs')
    ax1.set_ylabel('Loss Magnitude')
    ax1.legend(['Training', 'Validation'])
    plt.text(0.6, 0.7, 'Loss training: %.2f and validation %.2f '%(record1.history['loss'][-1],record1.history['val_loss'][-1]), transform=plt.gca().transAxes,bbox=dict(facecolor='white', alpha=0.8))
    # Show the plot
    plt.show()

def prediction_vs_observation1D(X_train, Y_train, X_test, Y_test, model, std_y, train_years, testing_years, outputs_path):
    """
    Plot the model's predictions vs. observations for both training and testing data.

    Parameters:
        X_train (numpy.ndarray): Input data for training.
        Y_train (numpy.ndarray): Ground truth output for training.
        X_test (numpy.ndarray): Input data for testing.
        Y_test (numpy.ndarray): Ground truth output for testing.
        model1 (keras.Model): Trained machine learning model.
        std_y (float): Standardization factor for predictions.
        train_years (tuple): Range of training years (e.g., (start_year, end_year)).
        testing_years (tuple): Range of testing years (e.g., (start_year, end_year)).
        outputs_path (str): Path to save the plot.

    Returns:
        None
    """
    # Calculate predictions and observations for training data
    prediction_train, observation_train = (model.predict(X_train)) * np.array(std_y), Y_train * std_y

    # Plot training data
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(train_years[0], train_years[1] + 1, 1), prediction_train, label='Prediction train')
    plt.plot(np.arange(train_years[0], train_years[1] + 1, 1), observation_train, label='Observation train')

    # Calculate correlation and RMSE for training data
    correlation_train, rmse_train = np.corrcoef(prediction_train.ravel(), observation_train)[0, 1], np.sqrt(
        np.mean((np.array(prediction_train).ravel() - np.array(observation_train)) ** 2))
    
    # Add correlation and RMSE text to the plot
    plt.text(0.05, 0.9, f'Correlation train: {correlation_train:.2f}', transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    plt.text(0.05, 0.85, f'RMSE train: {rmse_train:.2f}', transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))

    # Calculate predictions and observations for testing data
    prediction_test, observation_test = (model.predict(X_test)) * np.array(std_y), Y_test * std_y

    # Plot testing data
    plt.plot(np.arange(testing_years[0], testing_years[1], 1), prediction_test, label='Prediction test')
    plt.plot(np.arange(testing_years[0], testing_years[1], 1), observation_test, label='Observation test')
    plt.axvline(x=train_years[1], color='black')
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

def evaluation(model, X_test, Y_test, longitude, latitude, std_y, testing_years):
    """
    Evaluate a trained model's predictions against test data.

    Parameters:
    - model (tf.keras.Model): The trained neural network model.
    - X_test (array-like): Test input data.
    - Y_test (array-like): True target values for the test data.
    - longitude (array-like): Longitude values for spatial coordinates.
    - latitude (array-like): Latitude values for spatial coordinates.
    - std_y (float): Standard deviation for output field (for de-standardization).

    Returns:
    - predicted1 (xarray.DataArray): Predicted sea-level pressure anomalies.
    - correct_value (xarray.DataArray): True sea-level pressure anomalies.
    """
    # Predict sea-level pressure using the model
    predicted = model.predict(np.array(X_test))
    nt, nm= predicted.shape

    if np.ndim(predicted.shape)<=2:
        predicted= np.reshape(predicted, (nt, len(np.array(latitude)), len(np.array(longitude))))
    
    # Evaluate the model's performance on the test data
    results = model.evaluate(np.array(X_test), np.array(Y_test), batch_size=32)

    # Create xarray DataArray for predicted values
    predicted = xr.DataArray(
        data=predicted,
        dims=["time", "latitude", "longitude"],
        coords=dict(
            longitude=(["longitude"], np.array(longitude)),
            latitude=(["latitude"], np.array(latitude)),
            time=testing_years))

    correct_value = xr.DataArray(
        data=np.reshape(Y_test, (nt, len(np.array(latitude)), len(np.array(longitude)))),
        dims=["year", "latitude", "longitude"],
        coords=dict(
            longitude=(["longitude"], np.array(longitude)),
            latitude=(["latitude"], np.array(latitude)),
            year=testing_years))

    # De-standardize the predicted and true values to obtain anomalies
    predicted = predicted * std_y
    correct_value = correct_value * std_y
    
    return predicted, correct_value

def DibujoMapados(lon, lat, lon0, lat0, var0, var1, var2, mapbar, unidades, titulo1, titulo2, subtitulo1, periodo, subtitulo2, figura, año, reference_period):
    """
    Create a map plot with multiple subplots displaying climate data.

    Parameters:
    - lon (array-like): Longitude values for the main plot.
    - lat (array-like): Latitude values for the main plot.
    - lon0 (array-like): Longitude values for one of the subplots.
    - lat0 (array-like): Latitude values for one of the subplots.
    - var0 (array-like): Climate data for one of the subplots.
    - var1 (array-like): Climate data for the main plot (subplot 1).
    - var2 (array-like): Climate data for the main plot (subplot 2).
    - mapbar (str): Colormap for the plots.
    - unidades (str): Units for the colorbar.
    - titulo1 (str): Main title for the figure.
    - titulo2 (str): Subtitle for the figure.
    - subtitulo1 (str): Subtitle for the first subplot.
    - periodo (str): Time period information.
    - subtitulo2 (str): Subtitle for the second subplot.
    - figura (str): Filename for saving the figure.
    - año (int): Year information.
    - reference_period (str): Reference period information.

    Returns:
    - None
    """
    # Create a new figure
    plt.style.use('seaborn')
    fig0 = plt.figure(figsize=(10, 7.5))
    
    # Create subplot axes
    ax1 = fig0.add_subplot(221, projection=ccrs.PlateCarree())
    ax2 = fig0.add_subplot(222, projection=ccrs.PlateCarree())
    ax0 = fig0.add_subplot(212, projection=ccrs.PlateCarree())
    axes = [ax1, ax2, ax0]
    
    # Set the main title for the figure
    fig0.suptitle(str(titulo1) + ' for ' + str(año) +
                 '\n Reference period: ' + str(reference_period), fontsize=15, weight='bold')

    # Create the first subplot (ax0)
    im0 = ax0.contourf(lon0, lat0, var0, levels=np.round(np.linspace(-1.5, +1.5, 20), decimals=1), cmap='RdBu_r', extend='both')
    ax0.coastlines(linewidth=2)
    gl = ax0.gridlines(draw_labels=True)
    gl.ylabels_right = False
    gl.xlabels_top = False
    ax0.set_title(str(titulo2) + ' for ' + str(año) + ' ' + str(periodo), fontsize=15)
    
    # Calculate data range for the main plots
    maximo1 = np.max((np.ma.masked_array(var1, np.isnan(var1))))
    maximo2 = np.max((np.ma.masked_array(var2, np.isnan(var2))))
    maximo = np.max((maximo1, maximo2))
    minimo1 = np.min((np.ma.masked_array(var1, np.isnan(var1))))
    minimo2 = np.min((np.ma.masked_array(var2, np.isnan(var2))))
    minimo = np.min((minimo1, minimo2))
    rango = np.max((np.abs(maximo), np.abs(minimo)))
    
    # Create the second subplot (ax1)
    im = ax1.pcolormesh(lon, lat, var1, cmap=mapbar, vmin=-rango, vmax=rango, transform=ccrs.PlateCarree())
    ax1.coastlines(linewidth=2)
    gl = ax1.gridlines(draw_labels=True)
    gl.ylabels_right = False
    gl.xlabels_top = False
    ax1.set_title(subtitulo1, fontsize=15)

    # Create the third subplot (ax2)
    im3 = ax2.pcolormesh(lon, lat, var2, cmap=mapbar, vmin=-rango, vmax=rango, transform=ccrs.PlateCarree())
    ax2.coastlines(linewidth=2)
    gl = ax2.gridlines(draw_labels=True)
    gl.ylabels_right = False
    gl.ylabels_left = False
    gl.xlabels_top = False

    # Adjust the figure layout
    fig0.subplots_adjust(right=1.)
    fig0.colorbar(im0, ax=ax0, shrink=0.8, label='$^\circ$C', orientation='horizontal')

    # Create a colorbar axis
    cbar_ax = fig0.add_axes([0.97, 0.475, 0.03, 0.4])  # left, bottom, width, height
    fig0.colorbar(im3, cax=cbar_ax, label=unidades)

    ax2.set_title(subtitulo2, fontsize=15)
    plt.tight_layout()
    
    # Save the figure
    fig0.savefig(figura)

def correlations(predicted, correct_value, outputs_path, threshold, units, titulo1, titulo2, titulo_corr, periodo):
    """
    Calculate and visualize various correlation and RMSE metrics for climate predictions.

    Parameters:
    - predicted (xarray.DataArray): Predicted climate data.
    - correct_value (xarray.DataArray): Observed climate data.
    - outputs_path (str): Path to save the generated plots.
    - threshold (float): Theshold for the significance test.

    Returns:
    - fig
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
    plotter(data, data.longitude, data.latitude, np.linspace(-rango, +rango, 11), 'RdBu_r',
                                'Correlation', 'Temporal Correlation', ax, pixel_style=True)
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
    ax1.set_title('Spatial Correlation', fontsize=18)
    ax1.legend(loc='upper right')

    # Subplot 3: Spatial RMSE Map
    plt.style.use('default')
    ax4 = fig.add_subplot(223, projection=ccrs.PlateCarree(0))
    data = spatial_rmse
    rango= int(np.max(np.array(data)))
    plotter(data, data.longitude, data.latitude, np.arange(0, rango+1, 1), 'OrRd',
                          'RMSE', 'Temporal RMSE', ax4, pixel_style=True)
    
    plt.style.use('seaborn')
    # Subplot 4: Temporal RMSE Plot
    ax5 = fig.add_subplot(224)
    data = {'time': temporal_rmse.time, 'Predictions RMSE': temporal_rmse}
    df = pd.DataFrame(data)
    df.set_index('time', inplace=True)
    color_dict = {'Predictions RMSE': 'orange'}
    for i, col in enumerate(df.columns):
        ax5.bar(df.index, df[col], width=width, color=color_dict[col], label=col)

    ax5.set_title('Spatial RMSE', fontsize=18)
    ax5.legend(loc='upper right')
    ax5.set_ylabel(f'[{units}]')
    fig.suptitle(f'Metrics {titulo1} when predicting with {titulo_corr} {periodo} {titulo2}', fontsize=20)
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(outputs_path + 'correlations')
    return fig

def build_model(hp, params_selection):
    """
    Build a deep learning model with hyperparameters defined by the Keras Tuner.
    You have to create a dictionary of parameters before calling this function to select the hyperparameter space where you want to search, and add it to the params_selection
    Args:
        hp (HyperParameters): The hyperparameters object for defining search spaces.
        params_selection (dict): Dictionary containing the range of the hyperparameters to search
    Returns:
        keras.models.Model: A compiled Keras model.
    """
    
    # Access hyperparameters from params_selection
    pos_number_layers = params_selection['pos_number_layers']
    pos_layer_sizes = params_selection['pos_layer_sizes']
    pos_activations = params_selection['pos_activations']
    pos_dropout = params_selection['pos_dropout']
    pos_kernel_regularizer = params_selection['pos_kernel_regularizer']
    search_skip_connections = params_selection['search_skip_connections']
    pos_conv_layers = params_selection['pos_conv_layers']
    input_shape = params_selection['input_shape']
    output_shape = params_selection['output_shape']

    # Debugging: Print generated hyperparameter values
    print("Generated hyperparameters:")
    
    # Define the number of layers within the specified range
    num_layers = hp.Int('num_layers', 1, pos_number_layers)
    print("Number of layers:", num_layers)
    
    # Define layer sizes based on the specified choices
    layer_sizes = [hp.Choice('units_' + str(i), pos_layer_sizes) for i in range(num_layers)]
    print("Layer sizes:", layer_sizes)
    
    # Define activations for hidden layers and output layer
    activations = [hp.Choice('activations' + str(i), pos_activations) for i in range(len(layer_sizes)-1)] + ['linear']
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
    if kernel_regularizer == 'l1_l2':
        reg_function = 'l1_l2'
    else:
        reg_function = None  # No regularization

    # Create the model with specified hyperparameters
    model = create_model(input_shape, output_shape, layer_sizes, activations, dropout_rates, kernel_regularizer=reg_function, num_conv_layers=num_conv_layers ,num_filters= num_filters, pool_size=pool_size, kernel_size=kernel_size, use_batch_norm=use_batch_norm, use_initializer=use_initializer, use_dropout=use_dropout, use_initial_skip_connections=use_initial_skip_connections, use_intermediate_skip_connections=use_intermediate_skip_connections)
    
    # Define the optimizer with a learning rate hyperparameter
    optimizer = tf.keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', [1e-5,1e-4,1e-3,1e-2]))

    # Compile the model with the specified optimizer and loss function
    model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())
    
    return model



