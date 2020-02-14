# Deliverable: AGC case study

### 1.Introduction
This deliverable reports upon a software used to predict a UV resistance result of a coating mix. Before introducing the solution let us summary the problem as follows: we are given a coating measurements dataset of different glasses in the past. It includes some information such as percentage of triazines of two paints, cohension of two paints, the ratio of paint 1 and paint 2 after mix, resistance measured after coating, Min resistance, Max resistance. The problem is to predict whether or not a UV resistance value of a new coating mix will be within the target intervals. This prediction result is useful since it allows experts to decrease waiting and testing phase.
This is a binary classification problem since we want to predict the value of UV resistance satisfying the given threshold intervals or not. To tackle this problem, we propose to use a machine learning model to build a two-class classification. There is a large number of machine learning models for this problem such as Logistic Regression (LR), Support Vector Machine (SVM), Neural Network with a Perceptron (NNP). In this software, however, we propose to use Logistic Regression model to build a binary classification. This is a simple machine learning model used to predict the value of a numeric categorical variable based on its relationship with predictor variables. This is one of the most simple and commonly used machine learning algorithms for two-class classification. In addition, it is easy to implement and can be used as the baseline for any binary classification problem.
In particular, this delivarable reports how to reprocess training data; how to train model and evaluate its performance and how to use the software to predict a UV resistance test result of a new coating mix data.

### 2. Reprocessing data
#### Create binary variable
As mentioned above, LR model is used to build a binary classification. A first condition to build this model is that the predictant must be a binary variable. Since we want to use the `UV resistance` measured after coating as a predictant variable, we need to transform it from continuous numeric to binary. The result of a `UV resistance` depends on the target intervals. In particular, if it lies between `Min resistance` and `Max resistance` it is considered as a passed test. Otherwise, it is considered as a failed test. Based on this information, we can generate binary value for the variable `UV resistance` as follow: If `UV resistance` is lies between `Min resistance` and `Max resistance`, it is assigned 1. Otherwise, it is assigned 0.
#### Remove variables
In the measurement dataset, there are some variables that give equivalent information. In particular, 
`Model ID`, `Glass type` and a pair of `Min resistance` and `Max resistance` provide equivalent information. It means that given a pair of `Min resistance` and `Max resistance`, we can infer the `Model ID` and `Glass Type`. Therefore, we could remove the columns `Model ID` and `Glass type` from training data.
Additionally, columns `Coat data` and `Coating Line` don't have any relationship with UV resistance results. So we also need to remove them from the training data.
#### Clean data
To further clean data, we check missing values for all variables and replace them with appropriate values. In addition, duplicated instances in the training data are also removed.

After reprocessing step, the training data remains 10 variables (or 10 columns): `Emulsion Thickness`, `Triazine % in Paint 1`, `Triazine % in Paint 2`, `Paint 1 Cohesion`, `Paint 2 Cohesion`, `Paints 1 & 2 Mix Ratio (%)`, `Glass temp. prior coating`, `UV test result (Resistance)`, `Min Resistance`, `Max Resistance`.
In which, `UV test result (Resistance)` is the predictant variable and another variables are predictor variables.

### 3. Training Logistic Regression model
We applied common parameters to train the Logistic Regression model. The used parameters are given as follows:
- percentage of training data: 80%
- percentage of test data: 20%
- Algorithm to use in the optimization problem: liblinear

The performance of the model is reported below:

     Classification report without cross-validation               
				precision    recall  f1-score   support
           0       1.00      0.01      0.03        67
           1       0.69      1.00      0.82       148
    accuracy                           0.69       215
	macro avg      0.85      0.51      0.42       215
	weighted avg   0.79      0.69      0.57       215

    Report on K-fold cross-validation (K=10)
	- accuracy :  0.70
	- precision:  0.70
	- recall   :  0.99

*** Note that we have applied different sets of parameters to test the model. As a result, the performance of the model doesn't change significantly. In addition, we also ran other machine learning models such as SVM and NNP on this training data and compared their performances with LR model. As a result. The performance of LR is slightly better than the others ***


### 4. Usages
This software was implemented in Python. It required a running machine must have installed Python version 3.7 with necessary libraries such as numpy, pandas, matplotlib, seaborn, sklean.
To lanch the program, using the following command:

    python3 UV_resistance_predictor.py training_data predicting_data

**Parameters:**
- `training_data` is a csv file that contains coating measurements in the past.
- `predicting_data` is a csv file that contains a set of coating mix need to be predicted.

#### Input data
- tranining data: The format of training data is the same as the coating measurements that plant experts provided. However, we need to change it to format csv.
- predicting data is a csv file. The first line presents a list of column names. Each following line corresponds to a coating mix which need to be predicted. *** Note that, these csv files use `Tab` to seperate values ***

For example, predicting data given in `triazine-predict-data.csv` has following contents:

	Emulsion Thickness	Triazine % in Paint 1	Triazine % in Paint 2	Paint 1 Cohesion	Paint 2 Cohesion	Paints 1 & 2 Mix Ratio (%)	Glass temp. prior coating	Min Resistance	Max Resistance
	13	60	85	50.4	54.0	45.5	30	0.783	0.919
	14	60	85	57.2	54.8	76.0	31	0.62	0.751
	11	60	85	43.2	53.2	82.0	32	0.637	0.758
	10	60	85	58.8	52.0	58.0	32	0.862	1.002
	19	60	85	58.8	52.8	70.0	32	0.848	1.039

#### Output 
For each coating mix, the software produces the following information:
- sample : id
- input values: the input values of coating mix
- predict label: [1] or [0]. [1] means that the final UV resistance satisfies the given intervals. In contrast, 0 means that the final UV resistance doesn't satisfy the given intervals.
- probability: [a b]. a is a probabily to have prediction label 0; b is a probability to have prediction label 1

For example, below is prediction results of the software when executing the following command:

    python3 UV_resistance_predictor triazine-coating-measurements.csv triazine-predict-data.csv
	==================================
	      PREDICTION RESULTS        
	==================================
	sample : 1
	input values : [[13.    60.    85.    50.4   54.    45.5   30.     0.783  0.919]]
	predict label : [0]
	probability : [[0.64580596 0.35419404]]
	--------------------------------
	sample : 2
	input values : [[14.    60.    85.    57.2   54.8   76.    31.     0.62   0.751]]
	predict label : [1]
	probability : [[0.02225821 0.97774179]]
	--------------------------------
	sample : 3
	input values : [[11.    60.    85.    43.2   53.2   82.    32.     0.637  0.758]]
	predict label : [1]
	probability : [[0.06465484 0.93534516]]
	--------------------------------
	sample : 4
	input values : [[10.    60.    85.    58.8   52.    58.    32.     0.862  1.002]]
	predict label : [0]
	probability : [[0.92924405 0.07075595]]
	--------------------------------
	sample : 5
	input values : [[19.    60.    85.    58.8   52.8   70.    32.     0.848  1.039]]
	predict label : [1]
	probability : [[0.17034574 0.82965426]]
	--------------------------------
	

Based on the results, we can see that the final UV resistance of samples 2, 3 and 5 satisfy the intervals with the probabilities of 98%, 94% and 83%, respectively. Whereas, the the final UV resistance of samples 1 and 3 are predicted as failed.