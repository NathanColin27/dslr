# 42_dslr - Data Science Logistic Regression
Multinomial logistic regression from sratch using numpy, where the goal is to predict the house of Hogwarts students (gryffindor, slytherin, hufflepuff, ravenclaw), given their scored on various magic courses.

## Setup
just run `make` and activate the virtual env with the code below

## Activate Python3 virutal environment
`source .venv/bin/activate`. To finish virtual env, simpmly type `deactivate`

## Data Analysis
- `python3 describe.py` to get a descriptive table of our training set
- `python3 histogram.py` to get a histogram visualization of all numerical values
- `python3 scatter_plot.py` to get a scatter_plot. This one answers the question from the subject.pdf `which features are similar?`
- `python3 pair_plot.py` to get a pair-plot of all numerical values

## Logistic Regression
Train the model on the dataset.
`python3 logreg_train.py [-d dataset] [-a learning_rate] [-i iterations]`

## Prediction
`python3 logreg_predict.py [dataset]`