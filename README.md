# README

## Configuration to install and execute the **extractor-tool** app. 
From this Jupyter Lab environment, click on the "New" button, open a "Terminal" and then execute all the following shell commands:

* create a conda environment with all the necessary Python packages:
```
conda env create -f ./environment.yml
```

* activate the environment:
```
conda activate env_intaros_app
```

* to build and install the project locally:
```
python setup.py install
```

* test the command line command:
```
extractor-tool --help
```
If the app is installed successfully, the extractor-tool help will display.

## Configuration to execute the Jupyter Notebook (for development and debugging)
To and activate a kernel in Jupyter Notebook with the **env_intaros_app** environment just created, execute the following command in: 
```
python -m ipykernel install --user --name env_intaros_app --display-name 'env_intaros_app'
```
Then refresh the window and you should be able to select the **env_intaros_app** kernel in your Jupyter Notebook. 