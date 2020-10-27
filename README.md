

# MLLib
Machine learning regression and classification model templates 
on multiple platforms including 
`PyTorch`, `scikit-learn`, and `Spark MLLib`. 
A variety of utility tools are also included in the package.
This repository was built to make your life easier when starting a new ML project.
The source code files came from various projects the developer had worked on,
which may not be directly applicable to your problem yet should be easily transferable.
If you have any questions or valuable suggestions,
you are very welcome to contact the developer at
```bash
raymondwang@u.northwestern.edu
```

## Dataset preparation
This repository contains some data acquisition and feature engineering utility tools. Please see details enumerated below.

[1] Toy dataset for both regression and classification tasks can be found in `UCI_repo.tar.gz`

[2] Template for scraping data from unstructured online resources (e.g. balance sheet)
is provided in `scrape_website.py`. 
Google Chrome drivers for both MacOS and Linux can be found
[here](https://chromedriver.chromium.org/downloads).

[3] An example of fetching materials data from the [Materials Project](https://materialsproject.org/)
using their API is given in `fetch_MPdata.py`.
Parallel post-processing of original data is included in the source code,
where the function to be parallelized could be easily modified for your own task.

[4] An example of obtaining economic data from [FRED](https://fred.stlouisfed.org/) is presented in 
`fred_VAR.py`. Vector autoregression is used to analyze and predict the economic trend.

[5] `high_frequency_trade.py` demonstrates some basic feature engineering skills applicable to 
[limit order books](https://www.tradientblog.com/2020/03/understanding-the-limit-order-book/)
for high frequency trading tasks. 
Simple linear regression is used here for fundamental feature analysis.

[6] `stocks.py` is a small program for stock beta prediction. It is just a toy model.


## Sklearn templates
This repository contains multiple machine learning regression and classification model templates 
using the scikit-learn package, including:
```bash
Simple linear regression
Linear SVR
AdaBoost regresion
Gradient boosting regression/classification
Kernel ridge regression
SGD regression
Lasso-Lars regression
Multilayer perceptron regression
XGBoost regression
KNN classification
Support vector machine classification
Gaussian processes classification
```
We also provide tools to plot correlation heatmap (`correlation_heatmap.py`) 
as well as learning curve (`learning_curve.py`).


## PyTorch templates
This repository contains the major components required for a typical deep learning project, 
i.e. 
```bash
main.py # user/developer interface which defines modle parameters and work flow
data.py # driver program for data loader
model.py # the deep neural network model is defined here
```
`predict.py` is not necessary but we include it here for user convenience.
It bypasses the model training process and 
directly loads the pre-trained network to make predictions.

## Spark templates


