Project 1: Linear Feature Engineering
---

This repo consists of codebase for linear feature engineering, the first project for Quantitative Foundations. We use [Linear Regression](https://en.wikipedia.org/wiki/Linear_regression) as the learning algorithm. The training data in `traindata.txt` consists of 926 data points with 8 features and 1 output real value. And, the test data in `testinputs.txt` consists of 103 data points. Since, the number of training data points is extremely low, we rely on cross validation to avoid overfitting.

To run the codebase, follow the steps below.

1. Install the dependencies using the following command.

```bash
pip install -r requirements.txt
```


2. To generate the test predictions, using the obtained best model, run the following command. It will write a file named `prediction_results.csv` inside `reports/` directory. 

```bash
python main.py --mode test
```

3. Run the training pipeline using the following command. It will search the best feature expansion and writes a training visualization of validation MSE scores against different basis settings and the predictions on test set inside `results` directory.

```bash
python main.py --mode train
```

4. To visualize the analysis done on the data, run the following command.
```bash
python main.py --data-analysis
```
