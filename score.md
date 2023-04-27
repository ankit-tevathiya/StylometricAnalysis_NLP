We use accuracy and F1 score to evaluate our classifier.

Pass the path to the gold labels csv and then pass the path to the predicted labels csv.

Before running, ensure the numpy and csv libraries are installed in your environment.

python score.py --gold y_test.csv --pred y_pred_test.csv
>>> Accuracy: 0.5057
>>> F1 score: 0.6716