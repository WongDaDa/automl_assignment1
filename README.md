# AutoML Assignment 1

In this assignment, you are expected to implement both
* average ranking method
* greedy defaults method

## Files
This repository consists of four files:
* assignment.py: This file contains the base classes neccessary to program the assignment
* test.py: This file contains public unit tests, to verify the correctness
* data_toy.csv: The implementations will be tested on this dataset
* data_svm.csv: Real world dataset

The data files will be loaded in as panda data frames, each column represents a dataset and each row represents a configuration. 
The index of a row will be the hyperparameter setting, and the value of each cell represents the performance of the configuration on the dataset. 

In the assignment file, two functions are left unimplemented, i.e., the fit method for both the Average Rank and Greedy Defaults class. 
You will need to fill these in. 
Both implementations should not require you more than 20 lines of code.
Note that the fit method obtains the meta data as a preprocessed pandas data frame. 
It requires you to return a list of the configurations (represented as tuple).

Before implementing anything have a good look at the Unit Tests (test.py).
These can help you on your way.
