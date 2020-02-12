import sys
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
# import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix

from sklearn import svm
from sklearn.linear_model import Perceptron

#from pylab import rcParams
#rcParams['figure.figsize'] = 5, 4
#sb.set_style('whitegrid')


DEBUG_FLAG = False
MODEL_STATISTIC = True


''' load data from csv file '''
def load_data(address):
    input_data = pd.read_csv(address)

    if DEBUG_FLAG:
        print('input data: ')
        print(input_data.info())
        print(input_data.head())

    return input_data


''' calculate missing value for a column '''
def temperature_approx(cols):
    # show average values of data group by base_column
    # Parch_groups = data.groupby(data[base_column])
    # Parch_groups.mean()
    temp = cols[0]
    model = cols[1]
    if pd.isnull(temp):
        if model == 'SK2':
            return 32  # average temperature of Model ID = SK2
        else:
            return 31  # average value of column
            # if data has other missing value cases, we need to calculate them here
    else:
        return temp


''' input missing values for a column '''
def input_missing_value(data, column, base_column):
    data[column] = data[[column, base_column]].apply(temperature_approx, axis=1)


''' return a binary value '''
def bin_usv(cols):
    uv_resistance = cols[0]
    min_resistance = cols[1]
    max_resistance = cols[2]
    if min_resistance <= uv_resistance <= max_resistance:
        return 1
    else:
        return 0


''' convert values of a column into binary values'''
def bin_column(data, column, base_column1, base_column2):
    data[column] = data[[column, base_column1, base_column2]].apply(bin_usv, axis=1)


''' Converting categorical variable to a dummy indicators'''
def recode_variable(data, column):
    label_encoder = LabelEncoder()
    model_id_cat = data[column]
    model_id_encoder = label_encoder.fit_transform(model_id_cat)

    # get the list of model id names
    column_categorical = data[column]
    column_categorical = list(dict.fromkeys(column_categorical))

    binary_encoder = OneHotEncoder(categories='auto')
    model_1hot = binary_encoder.fit_transform(model_id_encoder.reshape(-1, 1))
    model_1hot_mat = model_1hot.toarray()
    model_id_DF = pd.DataFrame(model_1hot_mat, columns=column_categorical)

    data.drop([column], axis=1, inplace=True)
    data_dmp = pd.concat([model_id_DF, data], axis=1, verify_integrity=True)

    return data_dmp


def visualize_training_data(data):
    # visualize the correlation between variables
    sb_fig = sb.heatmap(data.corr())
    fig = sb_fig.get_figure()
    fig.savefig('correlation.png')

    # visualize the number of passed and failed UV resistance test
    sb_fig = sb.countplot(x='UV test result (Resistance)', data=data, palette='hls')
    fig = sb_fig.get_figure()
    fig.savefig('countplot.png')


def visualize_data_shape(data):
    sb_fig = sb.distplot(data)
    sb_fig.set_title("Data distribution")
    fig = sb_fig.get_figure()
    fig.savefig('data_distribution.png')


''' reprocessing training data '''
def reprocessing_data(data):

    # drop duplicates
    data.drop_duplicates()

    # input missing value for column 'Glass temp. prior coating' based on 'Model ID'
    input_missing_value(data, 'Glass temp. prior coating', 'Model ID')

    # calculate value of column 'UV test result (Resistance)' based on columns 'Min Resistance' and 'Max Resistance'
    bin_column(data, 'UV test result (Resistance)', 'Min Resistance', 'Max Resistance')
    '''
        if UV test result (Resistance) lies between Min and Max Resistance, the test is considered 'passed'.
        Otherwise, the test is 'failed'            
    '''

    # remove unnecessary columns
    data.drop(['Model ID', 'Glass type', 'Coat date', 'Coating Line'], axis=1, inplace=True)
    ''' 
        Model ID and Glass type can be inferred by Min and Max Resistance
        Coat date and Coating Line are not necessary for computing result of the UV resistance    
    '''

    # save fig of variable correlation of the available columns
    if DEBUG_FLAG:
        visualize_training_data(data)


''' build a model based on given algorithm'''
def build_model(training_data, alg):
    # Deploying and evaluating the model
    X_train, X_test, y_train, y_test = train_test_split(training_data.drop(['UV test result (Resistance)'], axis=1),
                                                        training_data['UV test result (Resistance)'], test_size=0.2,
                                                        random_state=200)
    # standardize train and test data
    standardize = StandardScaler()
    standardize.fit(X_train)
    X_train_scaled = standardize.transform(X_train)
    X_test_scaled = standardize.fit_transform(X_test)

    # store means and var to standardize input prediction data
    means = standardize.mean_
    var = standardize.var_

    # fit model to data
    if alg == 'lr':
        # Logistic Regression
        # model = LogisticRegression(solver='liblinear', fit_intercept=True, intercept_scaling=4, C=0.5)
        model = LogisticRegression(solver='liblinear')
        model.fit(X_train_scaled, y_train)
    else:
        if alg == 'svm':
            # Support Vector Machine
            model = svm.SVC(kernel='linear', probability=True)
            model.fit(X_train_scaled, y_train)
        elif alg == 'nn':
                # Neural Networks with a perceptron
                model = Perceptron()
                model.fit(X_train_scaled, y_train)
        else:
            print('invalid algorithm name')
            exit(0)

    # make a prediction
    y_pred = model.predict(X_test_scaled)

    # Model Evaluation
    if MODEL_STATISTIC:
        print('==================================')
        print('        MODEL PERFORMANCE         ')
        print('==================================')
        print('Classification report without cross-validation')
        print(classification_report(y_test, y_pred))

        # K-fold cross-validation & confusion matrices
        print('--------------------------------')
        print('K-fold cross-validation')
        y_train_pred = cross_val_predict(model, X_train_scaled, y_train, cv=10)
        print('confusion matrices:')
        print(confusion_matrix(y_train, y_train_pred))
        print('precision: ', precision_score(y_train, y_train_pred))
        print('recall:    ', recall_score(y_train, y_train_pred))

    return model, means, var


def run_prediction(model, means, var, predict_data, algorithm):
    print('\n==================================')
    print('      PREDICTION RESULTS        ')
    print('==================================')
    for i in range(predict_data.shape[0]):
        test_uv = np.array(predict_data.iloc[i]).reshape(1, -1)
        print('sample :', str(i + 1))
        print('input values :', test_uv)
        # standardize input data
        test_uv_scaled = (test_uv - means) / var
        print('predict label :', model.predict(test_uv_scaled))
        if algorithm == 'nn':
            print('probability :', model.score(test_uv_scaled, model.predict(test_uv_scaled)))
        else:
            print('probability :', model.predict_proba(test_uv_scaled))
            print('--------------------------------')


def main():

    train_data_address = 'triazine_coating_measurements.csv'
    predict_data_address = 'triazine_predict_data.csv'
    algorithm = 'lr'

    # get input training data and predict data from command line
    if len(sys.argv) == 4:
        train_data_address = sys.argv[1]
        predict_data_address = sys.argv[2]
        algorithm = sys.argv[3]
    else:
        print('USAGE:')
        print('python3 UV_resistance_predictor.py training_data predicting_data algorithm')
        print('- training_data is a csv file that contains coating measurements')
        print('- predicting_data is a csv file that contains a set of coating mix need to be predicted')
        print('- algorithm is one of the follows:')
        print('  + lr  : Logistic Regression')
        print('  + svm : Support Vector Machine')
        print('  + nn  : Neural Networks with a Perceptron')
        exit(0)

    # load training data
    training_data = load_data(train_data_address)
    # training_data.columns = ['Model ID', 'Glass type', 'Coat date', 'Coating Line', 'Emulsion Thickness',
    #                             'Triazine % in Paint 1', 'Triazine % in Paint 2', 'Paint 1 Cohesion',
    #                             'Paint 2 Cohesion', 'Paints 1 & 2 Mix Ratio (%)', 'Glass temp. prior coating',
    #                             'UV test result (Resistance)', 'Min Resistance', 'Max Resistance']

    # reprocessing training data
    reprocessing_data(training_data)

    # training a model based on given algorithm
    model, means, var = build_model(training_data, algorithm)

    # load predict data
    predict_data = load_data(predict_data_address)
    # predict_data.columns = ['Emulsion Thickness', 'Triazine % in Paint 1', 'Triazine % in Paint 2',
    #                           'Paint 1 Cohesion', 'Paint 2 Cohesion', 'Paints 1 & 2 Mix Ratio (%)',
    #                           'Glass temp. prior coating', 'Min Resistance', 'Max Resistance']

    # predict input samples
    run_prediction(model, means, var, predict_data, algorithm)


if __name__ == "__main__":
    main()