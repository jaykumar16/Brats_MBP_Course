Unet and Unet ++

Jaykumar Patel and Samal Munidasa

Require Packages:
```
pip3 install git+https://github.com/lucasb-eyer/pydensecrf.git
pip3 install logging 
pip3 install nibabel 
pip3 install h5py 
pip3 install pathlib 
pip3 install helpers 
pip3 install pdbpp 
pip3 install Pillow
pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip3 install opencv-python
```

To run training:

```
./train.py 
```
The parameters can be changed inside the train.py file. To change the number of layers, number of features of the layers and optimizer, please change inside ```./models/__init__.py```.


To run testing:

```./test.py``` 

The parameters can be changed inside the test.py file. To change the number of layers,
 number of features of the layers and optimizer, please change inside ```./models/__init__.py```.


To calculate the HD and DICE of the test dataset use matlab, ```Unet_UnetP_Analysis.m```
To visualize the contours of the brain, ```VisualizeSegmenrationContour.m```

Jupyter Notebook: ```test_ROC_Presicion.ipynb```

To visualize the result and plot ROC and precision curves.
