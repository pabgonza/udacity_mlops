# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity


### Table of Contents

1. [Project Description](#description)
2. [Installation](#installation)
3. [Running Files](#running_files)
3. [Contributing](#contributing)
3. [License](#license)


## Project Description <a name="description"></a>
Library with helper functions to train models for customer churn prediction. Includes unit testing. 

## Installation <a name="installation"></a>

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements:

    pip install -r requirements.txt 

## Running Files <a name="running_files"></a>

### Train models to predict customer churn

To train implemented models with the default provided data in **./data/bank_data.csv** you can run the churn library from the root directory:

    python churn_library.py

The script performs EDA, feature engineering, model training and evaluation. The results are stored in the following folders and files:

```bash
├── models
│   ├── logistic_model.pkl  (Logistic Regression model)
│   └── rfc_model.pkl       (Random Forest Classifier model)
│
├── images
│   ├── eda
│   │   ├── Churn.png
│   │   ├── Correlation_matrix.png
│   │   ├── Customer_Age.png
│   │   ├── Marital_Status.png
│   │   └── Total_Trans_Ct.png
│   └── results
│       ├── logistic_report.png
│       ├── logistic_ROC.png
│       ├── logistic_vs_rfc_ROC.png
│       ├── rfc_report.png
│       ├── rfc_ROC.png
│       ├── rfc_tree_explainer.png
│       └── rfc_feature_importance.png
│
└── logs
    └── churn_library.log
```

You can override bank_data.csv to train on your own data as long as you keep the same data structure.

### Testing

You can launch the test suite from the root directory (you will need to have pytest installed):

    pytest churn_script_logging_and_test.py

The tests results messages are stored in **./logs/churn_library.log**


## Contributing <a name="contributing"></a>
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License <a name="license"></a>
[MIT](https://choosealicense.com/licenses/mit/)