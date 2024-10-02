# NN4CAST
 Repository containing the nn4cast library and some necessary tools to work with it. It is designed to work with .nc data with coordinates: time, latitude and longitude

## Installation
**WARNING**: The environment must be compatible with all the dependencies, and it has been tested with python 3.9.15, future versions could be compatible with newer versions of python and other packages 

**NOTE**: Cartopy has to be installed with conda because pip version does not work

To get the latest version:
```console
    $ conda create -n <your-env-name>
    $ conda activate <your-env-name>
    (<your-env-name>) $ conda install cartopy python=3.9.15
    (<your-env-name>) $ pip install git+https://github.com/Victorgf00/nn4cast
```

To update to the latest version:
```console
    (<your-env-name>) $ pip install --upgrade git+https://github.com/Victorgf00/nn4cast
```
## Example usage: predict Atlantic SLP anomalies from ND with Pacific SST SO anomalies
### I. Import the functions and define the hyperparameters as a dictionary
```python
from nn4cast.predefined_classes import Dictionary_saver,Preprocess,Model_build_and_test,Model_searcher,Results_plotter,PC_analysis
import numpy as np
from tensorflow.keras import activations

hyperparameters = {
    # File paths, replace this with the actual paths to the files on your system.
    'path_x': '/path/to/your/data/HadISST1_sst_1870-2019.nc',
    'path_y': '/path/to/your/data/slp_ERA5_1940-2023.nc',

    # Time limits
    'time_lims': [1940, 2019],
    'jump_year': 0, #this is necesary when the predictor and predictand has different years for each sample (default=0)

    # Train, validation, and testing years: if dealing with X and Y of different years for each sample,
    # the years policy is from the input (X)
    'train_years': [1940, 1989],
    'validation_years': [1990,1999],
    'testing_years': [2000, 2019],

    # Input and output limits: for latitude first the northernmost, for longitude either -
    # -180-(+180) or 0-360, putting first the smaller number
    'lat_lims_x': [+75, -20],
    'lon_lims_x': [+120, +280],
    'lat_lims_y': [+75, -20], 
    'lon_lims_y': [-180, +180], 

    #Variable names, as defined in the .nc datasets
    'name_x': 'sst',
    'name_y': 'msl',

    # Months and months to skip
    'months_x': [9, 10],
    'months_skip_x': ['None'],
    'months_y': [11, 12],
    'months_skip_y': ['None'],

    # Seasonal method: select if computing seasonal means of aggregrates (True if means)
    'mean_seasonal_method_x': True,
    'mean_seasonal_method_y': True,
    
    # Regrid degrees: if you want to do a regrid of the data, if not, just put 0
    'regrid_degree_x': 2, 
    'regrid_degree_y': 2, 

    # Data scales: if you want to do a scale of the data, if not, just put 1
    'scale_x': 1, 
    'scale_y': 100, 

    # Detrending:
    'detrend_x': True, 
    'detrend_x_window': 50,
    'detrend_y': True,
    'detrend_y_window': 50,

    # Neural network hyperparameters (default parameters)
    'layer_sizes': [1024, 256, 64, 256, 1024],
    'activations': [activations.elu, activations.elu, activations.elu, activations.elu, tf.keras.activations.elu],
    'dropout_rates': [0.1],
    'kernel_regularizer': 'l2',
    'learning_rate': 0.0001,
    'epochs': 2500,
    'num_conv_layers':0,
    'use_batch_norm':True, 
    'use_initializer':True, 
    'use_dropout':True, 
    'use_init_skip_connections':False, 
    'use_inter_skip_connections':False,

    # Plotting parameters
    'mapbar': 'bwr',
    'units_x': '[$^{\circ} C$]',
    'units_y': '[$hPa$]',
    'region_predictor': 'Pacific',
    'p_value': 0.1,

    # Outputs path: define where to save all the plots and datasets, replace this with the actual paths to the files on your system.
    'outputs_path': '/path/to/the/directory/Outputs_ND_sst_SO_Pac/'}

Dictionary_saver(hyperparameters) #this is to save the dictionary, it will ask to overwrite if there is another with the same name in the directory

# Access the informative variables
print('****Informative variables****')
print(f"Predictor region: {hyperparameters['region_predictor']}")
print(f"Predictor months: {hyperparameters['months_x']} ; Predictant months: {hyperparameters['months_y']}")
print(f"Predictor lat_lims: {hyperparameters['lon_lims_x']} ; lon_lims: {hyperparameters['lat_lims_x']} || Predictant lat_lims: {hyperparameters['lat_lims_y']} ; lon_lims: {hyperparameters['lon_lims_y']}")
print(f"Periods for: training= {hyperparameters['train_years']} ; validation= {hyperparameters['validation_years']}; testing= {hyperparameters['testing_years']}")
print(f"Layers sizes: {hyperparameters['layer_sizes']} ; activations: {hyperparameters['activations']} ; dropout_rates: {hyperparameters['dropout_rates']} ; kernel_regularizer: {hyperparameters['kernel_regularizer']}")
```

### II. Preprocessing, Training & Testing the model
```python
dictionary_preprocess= Preprocess(dictionary_hyperparams= hyperparameters)
outputs_hold_out = Model_build_and_test(dictionary_hyperparams= hyperparameters, dictionary_preprocess= dictionary_preprocess, cross_validation= False, n_cv_folds=0)
outputs_cross_validation= Model_build_and_test(dictionary_hyperparams= hyperparameters, dictionary_preprocess= dictionary_preprocess, cross_validation= True, n_cv_folds=4, plot_differences=False, importances=True, region_importances=[[50,65],[-25,-10]])
```

### III. Evaluation of some results
```python
Results_plotter(hyperparameters, dictionary_preprocess, rang_x=2.5, rang_y=10, predictions=outputs_cross_validation['predictions'], observations=outputs_cross_validation['observations'], years_to_plot=[1962,1963], plot_with_contours=False, importances=outputs_cross_validation['importances'], region_importances=outputs_cross_validation['region_attributed'])
eof_analysis = PC_analysis(hyperparameters, outputs_cross_validation['predictions'], outputs_cross_validation['observations'], n_modes=4, n_clusters=3, cmap='RdBu_r')
```

### IV. Hyperparameter tunning and testing again
```python
params_selection = {
    'pos_number_layers': 3,  # set the maximum value of fully connected layers (int)
    'pos_layer_sizes': [16, 64, 256],  # set the possible layer sizes (list)
    'pos_activations': ["elu", "linear"],  # set the possible activation functions (possibilities are all the ones availabe in tensorflow: tf.keras.layers.activations()) (list)
    'pos_dropout': [0.1],  # set the possible dropout rates (list)
    'pos_kernel_regularizer': ["l2","l1_l2"],  # set the possible kernel regularizer (possibilities are: l1_l2, l1, l2, None) (list)
    'search_skip_connections': False,  # set if searching for skip connections (either intermediate or end_to_end) (bool)
    'pos_conv_layers': 0,  # set the maximum number of convolutional layers, the entry data must be 2D (int)
    'pos_learning_rate':  [1e-4,1e-3]} # set the possible learning rates (list)

#In the following line, you can modify the maximum number of trials ("max_trials") that the searcher does to look for the more optimum hyperparameters
outputs_bm_cross_validation = Model_searcher(dictionary_hyperparams=hyperparameters, dictionary_preprocess=dictionary_preprocess, dictionary_possibilities=params_selection, max_trials=1, n_cv_folds=2)

```

## Citation
If you use this software in your work, please cite our paper: 
```markdown
@article{galvan2024nn4cast,
  title={NN4CAST: An end-to-end deep learning application for seasonal climate forecasts},
  author={Galván Fraile, Víctor and Rodríguez-Fonseca, Belén and Polo, Irene and Martín-Rey, Marta and Moreno-García, María N},
  journal={EGUsphere},
  volume={2024},
  pages={1--24},
  year={2024},
  publisher={Copernicus Publications G{\"o}ttingen, Germany}
}
```

## Contribution
This project welcomes contributions and suggestions.
