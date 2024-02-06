# NN4CAST
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
    (<your-env-name>) $ conda install pydot graphviz
```

## Example usage: predict Atlantic SLP anomalies from JF with Pacific SST ND anomalies
### I. Import the functions and define the hyperparameters as a dictionary
```python
from nn4cast.predefined_classes import Dictionary_saver,Preprocess,Model_build_and_test,Model_searcher
import numpy as np
import tensorflow as tf

hyperparameters = {
    # File paths
    'path': '/home/victor/Desktop/prueba_nn4cast/Datasets/',
    'path_x': 'HadISST1_sst_1870-2019.nc',
    'path_y': 'aemet_pcp_monthly_5km_1951to2021_regrid005.nc',

    # Time limits
    'time_lims': [1951, 2019],

    # Years for output: using the policy of the predictor years
    'years_finally': np.arange(1951, 2019+1, 1), 
    'jump_year': 0, #this is necesary when the predictor and predictand has different years for each sample (default=0)

    # Reference period: period for computing the mean and standard deviation
    'reference_period': [1951, 2000], 

    # Train, validation, and testing years: if dealing with X and Y of different years for each sample,
    # the years policy is from the input (X)
    'train_years': [1951, 1989],
    'validation_years': [1990, 1999],
    'testing_years': [2000, 2019],

    # Input and utput limits: for latitude first the northernmost, for longitude either -
    # -180-(+180) or 0-360, putting first the smaller number
    'lat_lims_x': [55, -20],
    'lon_lims_x': [0, 360],
    'lat_lims_y': [44, 36], 
    'lon_lims_y': [-10, 5], 

    #Variable names, as defined in the .nc datasets
    'name_x': 'sst',
    'name_y': 'precipitation',

    # Months and months to skip
    'months_x': [9, 10],
    'months_skip_x': ['None'],
    'months_y': [9, 10],
    'months_skip_y': ['None'],

    # Seasonal method: select if computing seasonal means of aggregrates (True if means)
    'mean_seasonal_method_x': True,
    'mean_seasonal_method_y': False,
    
    # Regrid degrees: if you want to do a regrid of the data, if not, just put 0
    'regrid_degree_x': 2, 
    'regrid_degree_y': 0, 

    # Data scales: if you want to do a scale of the data, if not, just put 1
    'scale_x': 1, 
    'scale_y': 1, 

    # Overlapping: this is necessary if there is data for 0 and 360
    'overlapping_x': False, 
    'overlapping_y': False, 

    # Detrending
    'detrend_x': True, 
    'detrend_y': True,  

    # Renaming: if there is a mismatch between X and Y names of latitude-longitude and lat-lon
    'rename_x': False, 
    'rename_y': True, 

    # 1 Output: if there is only 1 output_point
    '1output': False, 

    # Nans policy: define how to deal with nans, either delete them or substitute to 0,
    # when deleting, it flattens the array
    'replace_nans_with_0_predictor': False, 
    'replace_nans_with_0_predictant': False, 

    # Neural network hyperparameters (default parameters)
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
    'units_y': '$kg/m^2$',
    'region_predictor': 'Pacific+Indian+Atlantic',
    'p_value': 0.1,

    # Outputs path: define where to save all the plots and datasets
    'outputs_path': '/home/victor/Desktop/prueba_nn4cast/Prueba_precip/Outputs_sst_all/'}

Dictionary_saver(hyperparameters) #this is to save the dictionary, it will ask to overwrite if there is another with the same name in the directory

# Access the informative variables
print('****Informative variables****')
print(f"Predictor region: {hyperparameters['titulo_corr']}")
print(f"Predictor months: {hyperparameters['months_x']} ; Predictant months: {hyperparameters['months_y']}")
print(f"Predictor lat_lims: {hyperparameters['lon_lims_x']} ; lon_lims: {hyperparameters['lat_lims_x']} || Predictant lat_lims: {hyperparameters['lat_lims_y']} ; lon_lims: {hyperparameters['lon_lims_y']}")
print(f"Periods for: training= {hyperparameters['train_years']} ; validation= {hyperparameters['validation_years']}; testing= {hyperparameters['testing_years']}")
print(f"Layers sizes: {hyperparameters['layer_sizes']} ; activations: {hyperparameters['activations']} ; dropout_rates: {hyperparameters['dropout_rates']} ; kernel_regularizer: {hyperparameters['kernel_regularizer']}")
```

### II. Preprocessing, Training & Testing the model
```python
dictionary_preprocess= Preprocess(dictionary_hyperparams= hyperparameters)
predicted_value,observed_value= Model_build_and_test(dictionary_hyperparams= hyperparameters, dictionary_preprocess= dictionary_preprocess, cross_validation=False, n_cv_folds=0)
predicted_global,observed_global= Model_build_and_test(dictionary_hyperparams= hyperparameters, dictionary_preprocess= dictionary_preprocess, cross_validation=True, n_cv_folds=8)
```

### III. Hyperparameter tunning and testing again (optional)
```python
params_selection = {
    'pos_number_layers': 5,  # set the maximum value of fully connected layers (int)
    'pos_layer_sizes': [16, 64, 256],  # set the possible layer sizes (list)
    'pos_activations': ["elu", "linear"],  # set the possible activation functions (possibilities are all the ones availabe in tensorflow: tf.keras.layers.activations()) (list)
    'pos_dropout': [0.0, 0.01],  # set the possible dropout rates (list)
    'pos_kernel_regularizer': ["l1_l2"],  # set the possible kernel regularizer (possibilities are: l1_l2, l1, l2, None) (list)
    'search_skip_connections': False,  # set if searching for skip connections (either intermediate or end_to_end) (bool)
    'pos_conv_layers': 0,  # set the maximum number of convolutional layers, the entry data must be 2D (int)
    'pos_learning_rate':  [1e-4,1e-3]} # set the possible learning rates (list)

#In the following line, you can modify the maximum number of trials ("max_trials") that the searcher does to look for the more optimum hyperparameters
fig4,fig5,fig6, predicted_global_bm, observed_global_bm= Model_searcher(dictionary_hyperparams= hyperparameters, dictionary_preprocess=dictionary_preprocess, dictionary_possibilities= params_selection, max_trials=10, n_cv_folds=8)
```

## Citation
If you use this software in your work, please cite our paper: 
```markdown
### PONER CITA
```

## Contribution
This project welcomes contributions and suggestions.
