# UV Resistance predictor
This software is used to predict the final UV resistance result of a coating mix.
In paritcular, given a dataset of measurements for different glasses in the past, this tool can be able to predict whether a UV resistance value of a new coating mix will be within the target intervals or not.

## Usage
This software was implemented in Python. To use it, the machine must have installed Python version 3.7 with necessary libraries such as numpy, pandas, matplotlib, seaborn, sklean, etc. To lanch the program, using the following command:

    python3 UV_resistance_predictor.py training_data predicting_data
**Parameters:**
- `training_data` is a csv file that contains coating measurements in the past.
- `predicting_data` is a csv file that contains a set of coating mix need to be predicted.

## Input data
- tranining data: The format of training data is the same as the coating measurements that plant experts provided. However, we need to change it to format csv.
- predicting data is a csv file. The first line presents a list of column names. Each following line corresponds to a coating mix which need to be predicted. *** Note that, these csv files use `Tab` to seperate values ***

For example, predicting data given in `triazine-predict-data.csv` has following contents:

	Emulsion Thickness	Triazine % in Paint 1	Triazine % in Paint 2	Paint 1 Cohesion	Paint 2 Cohesion	Paints 1 & 2 Mix Ratio (%)	Glass temp. prior coating	Min Resistance	Max Resistance
	13	60	85	50.4	54.0	45.5	30	0.783	0.919
	14	60	85	57.2	54.8	76.0	31	0.62	0.751
	11	60	85	43.2	53.2	82.0	32	0.637	0.758
	10	60	85	58.8	52.0	58.0	32	0.862	1.002
	19	60	85	58.8	52.8	70.0	32	0.848	1.039

## Output 
For each coating mix, the software produces the following information:
- sample : id
- input values: the input values of coating mix
- predict label: [1] or [0]. [1] means that the final UV resistance satisfies the given intervals. In contrast, 0 means that the final UV resistance doesn't satisfy the given intervals.
- probability: [a b]. a is a probabily to have prediction label 0; b is a probability to have prediction label 1

For example, below is prediction results of the software when executing the following command:
python3 UV_resistance_predictor triazine-coating-measurements-combined.csv triazine-predict-data.csv:

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