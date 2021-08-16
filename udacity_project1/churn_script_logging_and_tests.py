'''
Churn library testing functions

author: Pablo Gonzalez
date: July, 2021
'''
import os
import logging
import pandas as pd
import churn_library as cls

logging.basicConfig(filename='./logs/churn_library.log',
                    level=logging.INFO,
                    filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the
    other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err
    except pd.errors.EmptyDataError as err:
        logging.error("Testing import_eda: No columns to parse from file")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't have rows or columns"
        )
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    try:
        # df = cls.import_data("./data/bank_data_nan.csv")
        # df = cls.import_data("./data/bank_data_less_columns.csv")
        df = cls.import_data("./data/bank_data.csv")
        nan_vals_sums = df.isnull().sum()
        total_nan_vals = nan_vals_sums.sum()
        assert total_nan_vals == 0
    except AssertionError as err:
        for col_name in nan_vals_sums.index:
            if nan_vals_sums[col_name] > 0:
                logging.error(
                    "Testing perform_eda: The column %s contains NaN values",
                    col_name
                )
        raise err

    try:
        required_cols = ['Attrition_Flag', 'Customer_Age',
                         'Marital_Status', 'Total_Trans_Ct']
        for col_name in required_cols:
            assert col_name in df.columns
    except AssertionError as err:
        for col_name in required_cols:
            if col_name not in df.columns:
                logging.error(
                    "Testing perform_eda: Missing required column %s",
                    col_name
                )
        raise err

    try:
        perform_eda(df)
        expected_ouputs = [
            'images/eda/Churn.png',
            'images/eda/Correlation_Matrix.png',
            'images/eda/Customer_Age.png',
            'images/eda/Marital_Status.png',
            'images/eda/Total_trans_Ct.png'
        ]
        for expected_ouput in expected_ouputs:
            assert os.path.exists(expected_ouput)
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        for expected_ouput in expected_ouputs:
            if not os.path.exists(expected_ouput):
                logging.error(
                    "Testing perform_eda: Missing expected ouput %s",
                    expected_ouput
                )
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test test_encoder_helper
    '''
    try:
        # Load data and calculate Churn column
        df = cls.import_data("./data/bank_data.csv")
        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)

        cat_columns = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'
        ]
        for col_name in cat_columns:
            assert col_name in df.columns
    except AssertionError as err:
        for col_name in cat_columns:
            if col_name not in df.columns:
                logging.error(
                    "Testing encoder_helper: Missing required column %s",
                    col_name
                )
        raise err

    try:
        response_columns = [
            'Gender_Churn',
            'Education_Level_Churn',
            'Marital_Status_Churn',
            'Income_Category_Churn',
            'Card_Category_Churn'
        ]

        out_df = encoder_helper(df, cat_columns, response_columns)
        for col_name in response_columns:
            assert col_name in out_df.columns
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        for col_name in response_columns:
            if col_name not in df.columns:
                logging.error(
                    "Testing encoder_helper: Missing output column %s",
                    col_name
                )
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    try:
        # Load data and calculate Churn column
        df = cls.import_data("./data/bank_data.csv")
        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)

        X_train, X_test, y_train, y_test = perform_feature_engineering(df, None)
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: empty output set")
        raise err


def test_train_models(train_models):
    '''
    test train_models
    '''
    try:
        # Load data and calculate Churn column
        df = cls.import_data("./data/bank_data.csv")
        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)

        # Get train & test splits and train
        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(df, None)
        train_models(X_train, X_test, y_train, y_test)

        expected_ouputs = [
            'images/results/logistic_report.png',
            'images/results/logistic_ROC.png',
            'images/results/logistic_vs_rfc_ROC.png',
            'images/results/rfc_feature_importance.png',
            'images/results/rfc_report.png',
            'images/results/rfc_ROC.png',
            'images/results/rfc_tree_explainer.png',
            'models/logistic_model.pkl',
            'models/rfc_model.pkl'
        ]
        for expected_ouput in expected_ouputs:
            assert os.path.exists(expected_ouput)
        logging.info("Testing train_models: SUCCESS")
    except AssertionError as err:
        for expected_ouput in expected_ouputs:
            if not os.path.exists(expected_ouput):
                logging.error(
                    "Testing train_models: Missing expected ouput %s",
                    expected_ouput
                )
        raise err


if __name__ == "__main__":
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)
    logging.info("All tests passed!!")
