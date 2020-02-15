# Deliverable: AGC case study

### 1. Introduction
This deliverable reports upon a software used to predict a UV resistance result of a coating mix. As a short summary the problem is given as follows: we are given a coating measurements dataset of different glasses in the past. It includes some information such as percentage of triazines of two paints, cohension of two paints, the ratio of paint 1 and paint 2 after mix, resistance measured after coating, Min resistance, Max resistance. The problem is to predict whether or not a UV resistance value of a new coating mix will be within the target intervals (Min resistance, Max resistance). This prediction result is useful since it allows experts to decrease waiting and testing phase.

This is a binary classification problem since we want to predict the value of UV resistance which satisfies the given target intervals or not. To tackle this problem, we propose to use a machine learning model to build a two-class classification. Then, we use this model to predict a result of a new coating mix. There is a large number of machine learning models for this problem such as Logistic Regression (LR), Support Vector Machine (SVM), Neural Network with a Perceptron (NNP). In this software, however, we propose to use Logistic Regression model to build a binary classification. This is a simple machine learning model used to predict the value of a numeric categorical variable based on its relationship with predictor variables. This is one of the most simple and commonly used machine learning algorithm for two-class classification. In addition, it is easy to implement and can be used as the baseline for many binary classification problems.

In particular, this delivarable reports how to reprocess the dataset of measurements to have a training data; how to train machine learning model, evaluate its performance and use the software to predict a UV resistance test result of a new coating mix.

### 2. Reprocessing data
#### Create binary variable
As mentioned above, LR model is used to build a binary classification. It requires that the predictant must be a binary variable. Since we want to use the UV resistance measured after coating (`UV resistance` in short) as a predictant variable, we need to transform it from continuous numeric to binary. The result of a `UV resistance` depends on the target intervals. In particular, if it lies between `Min resistance` and `Max resistance` it is considered as a `passed test`. Otherwise, it is considered as a `failed test`. Based on this information, we can generate binary value for the variable `UV resistance` as follow: If `UV resistance` is between `Min resistance` and `Max resistance`, it is assigned 1. Otherwise, it is assigned 0.
#### Remove variables
In the measurement dataset, there are some variables that give equivalent information. Particularly, 
`Model ID`, `Glass type` and a pair of `Min resistance` and `Max resistance` provide equivalent information. It means that given a pair of `Min resistance` and `Max resistance`, we can infer the `Model ID` and `Glass Type`. Therefore, we could remove the columns `Model ID` and `Glass type` from the training data.
Additionally, columns `Coat data` and `Coating Line` don't have any relationship with UV resistance results. Therefore, we also need to remove them from the training data.
#### Clean data
To clean data, we need to check missing values for all variables and replace them with appropriate values. In addition, duplicated instances (rows of data) in the training data are also removed.

After reprocessing step, the training data remains 10 variables (or 10 columns): `Emulsion Thickness`, `Triazine % in Paint 1`, `Triazine % in Paint 2`, `Paint 1 Cohesion`, `Paint 2 Cohesion`, `Paints 1 & 2 Mix Ratio (%)`, `Glass temp. prior coating`, `UV test result (Resistance)`, `Min Resistance`, `Max Resistance`.
In which, `UV test result (Resistance)` is the predictant variable and another variables are predictor variables.

### 3. Training Logistic Regression model
We applied common parameters to train the Logistic Regression model. There are some optimization algorithms built for Logistic Regression model in Python. However, in this implementation, we select `liblinear` since it is suitable for small data. 

To have a training dataset and test dataset, we split the measurement dataset into two subsets with amount of 70% and 30%, respectively.

After training, the model achieves 70% precision on both training and test datasets. Details of performance evaluation of the model are given as follows:


     Classification report without cross-validation               
				precision    recall  f1-score   support
           0       0.67      0.04      0.07        52
           1       0.74      0.99      0.85       144
    accuracy                           0.74       196
	macro avg      0.70      0.52      0.46       196
	weighted avg   0.72      0.74      0.64       196

    Report on K-fold cross-validation (K=10)
	- accuracy :  0.70
	- precision:  0.70
	- recall   :  0.98

*** Note that we have applied different sets of parameters to test the model. As a result, the performance of the model doesn't change significantly. In addition, we also ran other machine learning models such as SVM and NNP on this training data and compared their performances with LR model. As a result, their performances are approximate ***


### 4. Usages
This software was implemented in Python. It required a running machine must have installed Python version 3.7 with necessary libraries such as numpy, pandas, matplotlib, seaborn, sklean.

To launch the program, using the following command:

    python3 UV_resistance_predictor.py training_data predicting_data

**Parameters:**

- `training_data` is a xlsx file that contains coating measurements in the past.
- `predicting_data` is a csv file that contains a set of coating mix required to be predicted.

#### Input data
- tranining data: The format of training data is the same coating measurements that experts provided.
- predicting data is a csv file. The first line presents a list of column names. Each following line corresponds to a coating mix which is needed to be predicted. *** Note that, the csv file uses `Tab` to seperate values ***

For example, predicting data given in `triazine-predict-data.csv` contains following information:

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
- probability: [x y]. x is a probabily to have prediction label of 0; y is a probability to have prediction label of 1

For example, belows are results of the software when executing the following command:
`python3 UV_resistance_predictor triazine-coating-measurements.xlsx triazine-predict-data.csv`
    
	
	==================================
	      PREDICTION RESULTS        
	==================================
	sample        : 1
	input values  : [[13.    60.    85.    50.4   54.    45.5   30.     0.783  0.919]]
	predict label : [0]
	probability   : [[0.53454755 0.46545245]]
	--------------------------------
	sample        : 2
	input values  : [[14.    60.    85.    57.2   54.8   76.    31.     0.62   0.751]]
	predict label : [1]
	probability   : [[0.04501496 0.95498504]]
	--------------------------------
	sample        : 3
	input values  : [[11.    60.    85.    43.2   53.2   82.    32.     0.637  0.758]]
	predict label : [1]
	probability   : [[0.09615758 0.90384242]]
	--------------------------------
	sample        : 4
	input values  : [[10.    60.    85.    58.8   52.    58.    32.     0.862  1.002]]
	predict label : [0]
	probability   : [[0.8338067 0.1661933]]
	--------------------------------
	sample        : 5
	input values  : [[19.    60.    85.    58.8   52.8   70.    32.     0.848  1.039]]
	predict label : [1]
	probability   : [[0.23789435 0.76210565]]
	--------------------------------
	
	The best coating mix
	==================================
	sample       : 2
	input values : [[14.    60.    85.    57.2   54.8   76.    31.     0.62   0.751]]
	probability  : 0.9549850424353914
	

Based on the results, we can see that the UV resistance test results of samples 2, 3 and 5 satisfy the intervals with the probabilities of 95%, 90% and 76% respectively. Whereas, the UV resistance test results of samples 1 and 4 are predicted 'failed test'. The best coating mix is the sample 2 because it has the highest probability.