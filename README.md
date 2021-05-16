# Deep-String-Matching-for-Duplicate-Detection

Deep string matching for record linkage that deals with typography errors, domain dependancy and toponyms.	

** The self-attention layer is currently not working correctly. This is being fixed.**

Article Link:

Python 3, Tensorflow 2.4.1 and Numpy 1.19.5 (note that using a higher version of Numpy is incompatible with Tensorflow)

The datasets used are:

	Historical places distances
	Entity Matching
	Quora Questions

	Amazon-Google (software)
	Amazon-Walmart (tech items)
	Restaurants
	Academic Citations

These can all be downloaded from the datasets folder.
The last 4 are taken from https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md#amazon-google and https://arxiv.org/pdf/2010.11075.pdf

The datasets were cleaned and manipulated to fit the model via the 3 python files in the folder titled Data Cleaning

The models are run using the 'main.py' file (where one can also change the settings and architecture of the model). Be sure to include the path to the correct dataset.
