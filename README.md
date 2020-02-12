# UV Resistance predictor
This tool is used to predict the final UV resistance test result of coating mix.
In paritcular, given a dataset of measurements for different glasses in the past, this tool can be able to predict whether the UV resistance value of a new coating mix (or a set of coating mix) will be within the target intervals or not.

## Usage
    python3 UV_resistance_predictor.py training_data predicting_data algorithm
**Parameters:**
- `training_data` is a csv file that contains coating measurements in the past.
- `predicting_data` is a csv file that contains a set of coating mix need to be predicted.
- `algorithm` is an algorithm used to build prediction model. It is one of the follows: `lr`: Logistic Regression, `svm`: Support Vector Machine, `nn`: Neural Networks with a Perceptron.

## Input data
- tranining data: The format of training data is the same as the coating measurements that plant experts provided. However, we need to change it to csv format.
- predicting data: is a csv file. The first line presents a list of columns' names. Each following line corresponds to a coating mix which need to be predicted. 

For example, triazine_predict_data.csv has contents as follows:

	Emulsion Thickness,Triazine % in Paint 1 ,Triazine % in Paint 2,Paint 1 Cohesion,Paint 2 Cohesion,Paints 1 & 2 Mix Ratio (%),Glass temp. prior coating,Min Resistance,Max Resistance
	13,60,85,50.4,54,45.5,30,0.783,0.919
	14,60,85,57.2,54.8,76,31,0.62,0.751
	11,60,85,43.2,53.2,82,32,0.637,0.758
	10,60,85,58.8,52,58,32,0.862,1.002
	19,60,85,58.8,52.8,70,32,0.848,1.039

## Output 
For each coating mix, the UV_resistance_predictor produces the following information:
- sample : id
- input values: the input values of coating mix
- predict label: [1] or [0]. [1] means that the final UV resistance satisfies the given intervals. In contrast, 0 means that the final UV resistance doesn't satisfy the given intervals.
- probability: [a b]. a is the probabily to have prediction label 0; b is probability to have prediction label 1

For example, the output of UV_resistance_predictor on the data given in file triazine_predict_data.csv is shown below:

	PREDICTION RESULTS
	sample : 1
	input values : [[13.    60.    85.    50.4   54.    45.5   30.     0.783  0.919]]
	predict label :  [0]
	probability :  [[0.6175529 0.3824471]]
	--------------------------------
	sample : 2
	input values : [[14.    60.    85.    57.2   54.8   76.    31.     0.62   0.751]]
	predict label :  [1]
	probability :  [[0.2138875 0.7861125]]
	--------------------------------
	sample : 3
	input values : [[11.    60.    85.    43.2   53.2   82.    32.     0.637  0.758]]
	predict label :  [1]
	probability :  [[0.10731715 0.89268285]]
	--------------------------------
	sample : 4
	input values : [[10.    60.    85.    58.8   52.    58.    32.     0.862  1.002]]
	predict label :  [0]
	probability :  [[0.58659249 0.41340751]]
	--------------------------------
	sample : 5
	input values : [[19.    60.    85.    58.8   52.8   70.    32.     0.848  1.039]]
	predict label :  [1]
	probability :  [[0.25859728 0.74140272]]
	

Based on the results, we can see that the final UV resistance of samples 2, 3 and 5 satisfy the intervals with the probabilities of 78%, 89% and 74%, respectively. Whereas, the the final UV resistance of samples 1 and 3 are predicted as failed.