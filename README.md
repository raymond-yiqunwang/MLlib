

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

## Dataset\_prep
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


## Feature engineering
    standard scaler
    one-hot encoding
    ordinal encoding

## Preprocessing
    train-test-val split

## Regression models
    linear regression

##Classification models
    decision tree

## Generative models
    GAN
    VAE

## Cross validation
    k-fold CV

## Error metrics
    MAE, MSE, RMSE
    F1, r2, AUC

## Visualization

