import sys
import numpy as np
import pandas as pd
import seaborn as sb
# import matplotlib.pyplot as plt
# import sklearn

# from pandas import Series, DataFrame
from pylab import rcParams
from sklearn import preprocessing

#
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict

# from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score

#%matplotlib inline
rcParams['figure.figsize'] = 5, 4
sb.set_style('whitegrid')


DEBUGFLAG = False


''' load data from csv file '''
def load_data(address):
    input_data = pd.read_csv(address)

    if DEBUGFLAG:
        print('input data: ')
        print(input_data.info())
        print(input_data.head())

    return input_data


''' calculate missing value for a column '''
def temperature_approx(cols):
    # show average values of data group by base_column
    # Parch_groups = data.groupby(data[base_column])
    # Parch_groups.mean()
    Temp = cols[0]
    Parch = cols[1]
    if pd.isnull(Temp):
        if Parch == 'SK2':
            return 32  # average value of Model ID = SK2
        else:
            return 31  # average value of column
            # if data has other missing value cases, we need to calculate them here
    else:
        return Temp


''' input missing values '''
def input_missing_value(data, column, base_column):

    data[column] = data[[column, base_column]].apply(temperature_approx, axis=1)

    if DEBUGFLAG:
        data.isnull().sum()


''' return a binary value '''
def bin_usv(cols):
    usv_resitance = cols[0]
    min_resistance = cols[1]
    if usv_resitance < min_resistance:
        return 0
    else:
        return 1

''' convert values of a column into binary values'''
def bin_column(data, column, base_column):
    data[column] = data[[column, base_column]].apply(bin_usv, axis=1)


''' clean training data '''
def clean_data(data):
    # input missing value for column 'Glass temp. prior coating' based on 'Model ID'
    input_missing_value(data, 'Glass temp. prior coating', 'Model ID')

    # bin values of column 'UV test result (Resistance)' based on column 'Min Resistance'
    bin_column(data, 'UV test result (Resistance)', 'Min Resistance')

    # check variable correlation
    if DEBUGFLAG:
        sb.heatmap(data.corr())

    # TODO: check training data to determine which columns should be removed
    # remove unnecessary columns
    data.drop(['Model ID', 'Glass type', 'Coat date', 'Coating Line', 'Min Resistance', 'Max Resistance'],
                      axis=1, inplace=True)

    if DEBUGFLAG:
        data.head()

def build_model(training_data):
    # Deploying and evaluating the model
    X_train, x_test, y_train, y_test = train_test_split(training_data.drop(['UV test result (Resistance)'], axis=1),
                                                        training_data['UV test result (Resistance)'], test_size=0.2,
                                                        random_state=200)
    # check shape of train, test data
    if DEBUGFLAG:
        print(X_train.shape)
        print(y_train.shape)

    # fit data into model
    LogReg = LogisticRegression(solver='liblinear')
    LogReg.fit(X_train, y_train)

    y_pred = LogReg.predict(x_test)

    # Model Evaluation
    # Classification report without cross-validation
    if DEBUGFLAG:
        print('performance of the model')
        print(classification_report(y_test, y_pred))

    # K-fold cross-validation & confusion matrices
    y_train_pred = cross_val_predict(LogReg, X_train, y_train, cv=5)
    # print(confusion_matrix(y_train, y_train_pred))

    # output precision score of the model
    # precision_score(y_train, y_train_pred)
    return LogReg

def main():

    train_data_address = 'triazine_coating_measurements.csv'
    predict_data_address = 'triazine_predict_data.csv'

    # input training data and predict data from command line
    if len(sys.argv) == 2:
        train_data_address_new = sys.argv[1]
        predict_data_address_new = sys.argv[2]
        print(train_data_address_new)
        print(predict_data_address_new)

    # load training data
    triazine_training = load_data(train_data_address)
    triazine_training.columns = ['Model ID', 'Glass type', 'Coat date', 'Coating Line', 'Emulsion Thickness',
                                 'Triazine % in Paint 1', 'Triazine % in Paint 2', 'Paint 1 Cohesion',
                                 'Paint 2 Cohesion', 'Paints 1 & 2 Mix Ratio (%)', 'Glass temp. prior coating',
                                 'UV test result (Resistance)', 'Min Resistance', 'Max Resistance']

    # clean data
    clean_data(triazine_training)

    # training LogisticRegression model
    LogReg = build_model(triazine_training)

    # load predict data
    predict_data = load_data(predict_data_address)
    predict_data.columns = ['Emulsion Thickness', 'Triazine % in Paint 1', 'Triazine % in Paint 2', 'Paint 1 Cohesion',
                            'Paint 2 Cohesion', 'Paints 1 & 2 Mix Ratio (%)', 'Glass temp. prior coating']

    # test one sample
    # test_suv = np.array([13.0, 60.0, 85.0, 50.5, 34.0, 55.5, 33.0]).reshape(1, -1)

    # test all samples given in a csv file
    print('--------------------------------')
    for i in range(predict_data.shape[0]):
        test_suv = np.array(predict_data.iloc[i]).reshape(1, -1)
        print('input values of sample : ',  str(i))
        print(test_suv)
        print('predict label : ', LogReg.predict(test_suv))
        print('probability : ', LogReg.predict_proba(test_suv))
        print('--------------------------------')

if __name__ == "__main__":
    main()