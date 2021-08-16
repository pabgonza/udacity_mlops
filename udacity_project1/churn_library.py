'''
Churn library project 1 solution

author: Pablo Gonzalez
date: July, 2021
'''
import os
import shap
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report
import matplotlib
import matplotlib.pyplot as plt

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sns.set()
matplotlib.use('Agg')


def save_current_fig(fig_title, out_pth):
    '''
    saves current matplotlib fig in out_pth

    input:
            fig_title (str): Figure title
            out_pth (str): ouput path with any matplotlib supported extensions
    output:
            None
    '''
    fig = plt.gcf()
    fig.set_size_inches(20, 10)
    fig.suptitle(fig_title, fontsize=20)
    plt.tight_layout()
    fig.savefig(out_pth)


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df_out = pd.read_csv(pth)
    return df_out


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    # Calculate Churn column
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # Save Churn histogram
    plt.figure()
    df['Churn'].hist()
    save_current_fig('Customer Churn histogram',
                     'images/eda/Churn.png')

    # Save Customer age histogram
    plt.figure()
    df['Customer_Age'].hist()
    save_current_fig('Customer age histogram',
                     'images/eda/Customer_Age.png')

    # Save Marital status counts plot
    plt.figure()
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    save_current_fig('Marital Status counts',
                     'images/eda/Marital_Status.png')

    # Save Total Trans Ct distribution plot
    plt.figure()
    sns.distplot(df['Total_Trans_Ct'])
    save_current_fig('Total Trans Ct distribution',
                     'images/eda/Total_Trans_Ct.png')

    # Save Correlation Matrix
    plt.figure()
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    save_current_fig('Correlation Matrix',
                     'images/eda/Correlation_Matrix.png')


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the
    notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could
                be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    # Copy input dataframe to avoid modifications
    out_df = df.copy()

    # Obtain new category names
    category_new_lst = response
    if category_new_lst is None:
        category_new_lst = [cat+'_Churn' for cat in category_lst]

    # Encode categories in category_lst
    for cat, cat_new in zip(category_lst, category_new_lst):
        cat_lst = []
        cat_groups = out_df.groupby(cat).mean()['Churn']

        for val in out_df[cat]:
            cat_lst.append(cat_groups.loc[val])
        out_df[cat_new] = cat_lst

    return out_df


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could
                be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    cat_columns = [
        'Gender', 'Education_Level', 'Marital_Status', 'Income_Category',
        'Card_Category'
    ]

    keep_cols = [
        'Customer_Age', 'Dependent_count', 'Months_on_book',
        'Total_Relationship_Count', 'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
        'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
        'Income_Category_Churn', 'Card_Category_Churn'
    ]

    # perform_eda()
    # Encode categorical columns
    df_encoded = encoder_helper(df, cat_columns, response)

    # Get input and output data
    X = df_encoded[keep_cols]
    y = df_encoded['Churn']

    # Get train and test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores
    report as image in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    plt.figure(figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.tight_layout()
    plt.gcf().savefig('./images/results/rfc_report.png')

    plt.figure(figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.tight_layout()
    plt.gcf().savefig('./images/results/logistic_report.png')


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances and sort indescending order
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create and save plot
    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(X_data.shape[1]), importances[indices])
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.tight_layout()
    plt.gcf().savefig(output_pth)


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # Train Random Forest and Logistic Regression models
    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc = LogisticRegression()
    lrc.fit(X_train, y_train)

    rfc_model = cv_rfc.best_estimator_
    lr_model = lrc

    # rfc_model = joblib.load('./models/rfc_model.pkl')
    # lr_model = joblib.load('./models/logistic_model.pkl')

    # Obtain model predictions
    y_train_preds_rf = rfc_model.predict(X_train)
    y_test_preds_rf = rfc_model.predict(X_test)

    y_train_preds_lr = lr_model.predict(X_train)
    y_test_preds_lr = lr_model.predict(X_test)

    # Save models results
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    joblib.dump(rfc_model, './models/rfc_model.pkl')
    joblib.dump(lr_model, './models/logistic_model.pkl')

    plot_roc_curve(rfc_model, X_test, y_test)
    save_current_fig('Random Forest Classifier ROC',
                     './images/results/rfc_ROC.png')

    plot_roc_curve(lr_model, X_test, y_test)
    save_current_fig('Logistic Regression ROC',
                     './images/results/logistic_ROC.png')

    plot_roc_curve(lr_model, X_test, y_test)
    axis = plt.gca()
    plot_roc_curve(rfc_model, X_test, y_test, ax=axis, alpha=0.8)
    save_current_fig('Logistic Regression V/S Random Forest ROC',
                     './images/results/logistic_vs_rfc_ROC.png')

    plt.figure()
    explainer = shap.TreeExplainer(rfc_model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    save_current_fig('Tree SHAP Explainer',
                     './images/results/rfc_tree_explainer.png')

    feature_importance_plot(rfc_model, X_test,
                            './images/results/rfc_feature_importance.png')


if __name__ == "__main__":
    CSV_PTH = r"./data/bank_data.csv"
    df = import_data(CSV_PTH)

    perform_eda(df)
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, None)
    train_models(X_train, X_test, y_train, y_test)
