The strong baseline employs a pipeline in which the data is passed through two GRU-cell RNNs (one direction) and the cosine similarity of the two outputs is used in a single feature logistic regression classifier.

The results are as follows:
Training Loss   Training Accuracy   Training F1-score   Dev. Loss   Dev. Accuracy   Dev. F1-score
-0.5011         0.5527	            0.6473	            -0.5011     0.5672	        0.6577

Before running the script, ensure the following libraries/packages are (conda/pip/etc.) installed in your python environment:
pytorch
numpy
pandas
pymagnitude
scikit-learn

Then excute the following line
python strong-baseline.py

No arguments are needed. *** Ensure that within the directory where the script is there is a folder data/ that contains the data sets and the glove embeddings .magnitude file. ***