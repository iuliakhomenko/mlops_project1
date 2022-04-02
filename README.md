# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
The purpose of this project is to identify credit card customers that are most likely to churn using machine learning

This project is implemented as Python script so could be ran from the command line.
The project contains the following files:
1) churn_library.py - main functions covering the key stages of the project (loading the datam, EDA, training and testing models)
All plots and reports produced while running the script are saved in ```images``` folder.
All the trained models used in project are saved in ```models``` folder.

2) churn_script_logging_and_tests.py - unit tests civering main functions of churn_library.py
3) constants.py - the main constant variables used in the project
4) helpers.py - small helper functions used in churn_library.py


## Running Files
How do you run your files? What should happen when you run your files?
To run churn_library.py, run the command below in the command line:

```python churn_library.py```

The command above will start the main script with reading in the data, cleaning and feature engineering, model training 
and generating reports with models results. 

To run the tests script run the following command:

```pytest -o log_file='./logs/churn_library.log'  --log-file-level=INFO churn_script_logging_and_tests.py```

The command above will run unit tests using Python package ```pytest``` and saving all the logs to ```./logs/churn_library.log``` 


