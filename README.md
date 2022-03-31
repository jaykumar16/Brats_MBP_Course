Unet and Unet ++

Jaykumar Patel and Samal Munidasa

This code is tested with the ```python3.8``` and ```matlab2018b```.

Require Packages:
```
pip3 install -r requirements.txt
```

The preprocess to create the training, validation and test data,
``` 
preprocessing.ipynb
```

To run the optuna hyperparameter tuning for Unet++ and Unet is,

```
Unet++HyperOptimization.ipynb & UnetHyperOptimization.ipynb 
```
The result of the optimization is saved in studyUnet.pkl and studyUnetPlusPlusNew.pkl. 


To run training:

```
./train.py 
```

The parameters can be changed inside the train.py file. To change the number of layers, number of features of the layers and optimizer, please change inside ```./models/__init__.py``` before running training and test.


To run testing:

```
./test.py
``` 
To calculate the HD and DICE of the test dataset use matlab, ``` Unet_UnetP_Analysis.m ```
To visualize the contours of the brain, ``` VisualizeSegmenrationContour.m ```

Jupyter Notebook: ```test_ROC_Presicion.ipynb```

To visualize the result and plot ROC and precision curves.
