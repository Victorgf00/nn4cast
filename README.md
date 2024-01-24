# nn4cast
 Repository containing the nn4cast library and some necessary tools to work with it

## Installation
**WARNING**: The environment must be compatible with all the dependencies, and it has been tested with python 3.9.16, future versions could be compatible with newer versions of python and other packages
**NOTE**: Cartopy has to be installed with conda because pip version does not work

To get the latest version:
```console
    $ conda create -n <your-env-name> python=3.9.16
    $ conda activate <your-env-name>
    (<your-env-name>) $ conda install pip
    (<your-env-name>) $ conda install cartopy
    (<your-env-name>) $ pip install git+https://github.com/Victorgf00/nn4cast
```

## Example to predict Atlantic SLP anomalies from JF with Pacific SST ND anomalies
### I. Import the functions and define the hyperparameters as a dictionary
```python
from nn4cast.predefined_classes import Dictionary_saver,Preprocess,Model_build_and_test,Model_searcher

hyperparameters = {
    # File paths
    'path': 'C:/Users/ideapad 5 15ITL05/Desktop/TFM/Predicciones_y_comparacion_ecmwf/Datasets/',
    'path_x': 'HadISST1_sst_1870-2019.nc',
    'path_y': 'slp_ERA5_1940-2023.nc',

    # Time limits
    'time_lims': [1940, 2020],
    'years': np.arange(1940, 2020+1, 1),

    # Output limits
    'lat_lims_y': [70, 20],
    'lon_lims_y': [-100, 40],

    # Input limits
    'lat_lims_x': [55, -20],
    'lon_lims_x': [120, 280],

    #Variable names
    'name_x': 'sst',
    'name_y': 'msl',

    # Months and months to skip
    'months_x': [11, 12],
    'months_skip_x': ['None'],
    'months_y': [1, 2],
    'months_skip_y': ['1940-01', '1940-02'],

    # Regrid degrees
    'regrid_degree_x': 2, #if you want to do a regrid of the data, if not, just put 1
    'regrid_degree_y': 2, #if you want to do a regrid of the data, if not, just put 1

    # Data scales
    'scale_x': 1, #if you want to do a scale of the data, if not, just put 1
    'scale_y': 100, #if you want to do a scale of the data, if not, just put 1

    # Overlapping
    'overlapping_x': False, #this is necessary if there is data for 0 and 360
    'overlapping_y': False, #this is necessary if there is data for 0 and 360

    # Latitude regrid
    'lat_regrid_x': False, #this is necessary if the latitudes are in different points in X and Y, say 1 and 0.5
    'lat_regrid_y': False, #this is necessary if the latitudes are in different points in X and Y, say 1 and 0.5

    # Detrending
    'detrend_x': True, #if you want to do a detrending
    'detrend_y': True,  #if you want to do a detrending

    # Renaming
    'rename_x': False, #if there is a mismatch between X and Y names of latitude-longitude and lat-lon
    'rename_y': False, #if there is a mismatch between X and Y names of latitude-longitude and lat-lon

    # 1 Output
    '1output': False, #if there is only 1 output_point

    # Nans policy
    'replace_nans_with_0_predictor': False, #define how to deal with nans in X
    'replace_nans_with_0_predictant': False, #define how to deal with nans in Y

    # Years for output
    'years_finally_x': np.arange(1940, 2019+1, 1),
    'jump_year': 1, #this is necesary when the predictor and predictand has different years for each sample, if no set to 0
    'years_finally_y': np.arange(1941, 2020+1, 1),

    # Reference period
    'reference_period': [1950, 2000], #period for computing the mean and std

    # Train, validation, and testing years
    'train_years': [1940, 1989],
    'validation_years': [1990, 1999],
    'testing_years': [2000, 2019],

    # Neural network hyperparameters
    'layer_sizes': [1024, 256, 64],
    'activations': [tf.keras.activations.elu, tf.keras.activations.elu, tf.keras.activations.elu],
    'dropout_rates': [0.0],
    'kernel_regularizer': 'l1_l2',
    'learning_rate': 0.0001,
    'epochs': 2500,
    'num_conv_layers':0,
    'use_batch_norm':True, 
    'use_initializer':True, 
    'use_dropout':True, 
    'use_initial_skip_connections':False, 
    'use_intermediate_skip_connections':False,

    # Plotting parameters
    'mapbar': 'bwr',
    'titulo1': 'Comparison of JF anomalies SLP',
    'titulo2': 'SST anomaly',
    'subtitulo1': 'Predictions',
    'periodo': 'ND',
    'subtitulo2': 'Observations',
    'unidades': 'hPa',
    'titulo_corr': 'Pacific',
    'p_value': 0.1,
    # Outputs path 
    'outputs_path': "C:/Users/ideapad 5 15ITL05/Desktop/Doctorado/Libreria/Outputs_prueba/"}

Dictionary_saver(hyperparameters) #this is to save the dictionary, it will ask to overwrite if there is another with the same name in the directory

# Access the informative variables
print('***Informative variables***')
print(f'Predictor region: {titulo_corr}')
print(f'Predictor months: {months_x} ; Predictant months: {months_y}')
print(f'Predictor lat_lims: {lon_lims_x} ; lon_lims: {lat_lims_x} || Predictant lat_lims: {lat_lims_y} ; lon_lims: {lon_lims_y}')
print(f'Periods for: training= {train_years} ; validation= {validation_years}; testing= {testing_years}')
print(f'Layers sizes: {layer_sizes} ; activations: {activations} ; dropout_rates: {dropout_rates} ; kernel_regularizer: {kernel_regularizer}')
```

### II. Preprocessing, Training & Testing the model
```python
dictionary_preprocess= Preprocess(dictionary_hyperparams= hyperparameters)
predicted_value,observed_value= Model_build_and_test(dictionary_hyperparams= hyperparameters, dictionary_preprocess= dictionary_preprocess, cross_validation=False, n_cv_folds=0)
predicted_global,observed_global= Model_build_and_test(dictionary_hyperparams= hyperparameters, dictionary_preprocess= dictionary_preprocess, cross_validation=True, n_cv_folds=8)
```

### III. Hyperparameter tunning and testing again
```python
params_selection = {
    'pos_number_layers': 5,  # set the maximum value of fully connected layers
    'pos_layer_sizes': [16, 64, 256],  # set the possible layer sizes
    'pos_activations': ["elu", "linear"],  # set the possible activation functions (possibilities are all the ones availabe in tensorflow: tf.keras.layers.activations())
    'pos_dropout': [0.0, 0.01],  # set the possible dropout rates
    'pos_kernel_regularizer': ["l1_l2"],  # set the possible kernel regularizer (possibilities are: l1_l2, l1, l2, None)
    'search_skip_connections': False,  # set if searching for skip connections (either intermediate or end_to_end)
    'pos_conv_layers': 0,  # set the maximum number of convolutional layers, the entry data must be 2D
    'pos_learning_rate':  [1e-4,1e-3]} # set the possible learning rates

fig4,fig5,fig6, predicted_global_bm, observed_global_bm= Model_searcher(dictionary_hyperparams= hyperparameters, dictionary_preprocess=dictionary_preprocess, dictionary_possibilities= params_selection, max_trials=2, n_cv_folds=8)
```
