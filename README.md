# UV Resistance predictor
This software is used to predict a UV resistance test result of a coating mix. In paritcular, given a dataset of measurements for different glasses in the past, this software can be able to predict whether or not a UV resistance value of a new coating mix will be within the target intervals.

#### Usage
This software was implemented in Python. To run it, a machine must have installed Python version 3.7 with necessary libraries such as numpy, pandas, matplotlib, seaborn, sklean.

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