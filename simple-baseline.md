First, we read in our csv file for the dataset. Then, we outputted a csv with the same length as the input csv. Our simple baseline consists of all of the predicted labels being the "same," which means that both of our texts are from the Charles Dickens corpus.

An example of how to run this from the command line:
Prior to running the line below, please make sure you have pandas installed.

--python3 simple-baseline.py -s [insert "train", "dev", or "test"]

Sample output:
same
same
same
....