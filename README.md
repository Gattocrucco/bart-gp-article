# bart-gp-article

Code to reproduce the results in Petrillo (2024), "On the Gaussian process limit of Bayesian Additive Regression Trees".

## Files setup

* Copy/clone the files to your computer
* Set the working directory to `bart-gp-article/code`

It takes a while to clone the repository because it includes about 1 GB of data files.

## R setup

Install:

  * R 4.4.1 https://www.r-project.org
  
  * JDK 19.0.1 https://jdk.java.net (on macOS, put the directory into `/Library/Java/JavaVirtualMachines/`)

Then install the following R packages:

```R
library(remotes)
install_version('bartMachine', version='1.3.4.1')
install_version('BART', version='2.9.9')
install_version('dbarts', version='0.9-28')
install_version('BayesTree', version='0.3-1.5')
```

Everything probably works with newer versions, but I've listed the ones I used to run the code myself for reproducibility.

## Python setup

### Brief version

Install Python 3.12, make a virtual environment, then install from `requirements.txt`.

### Long version

If you don't have Python experience, here is a possible detailed path to install everything without messing up other Python installations. On Mac; I don't know about Windows or Linux.

* Install [homebrew](https://brew.sh)
* `$ brew install micromamba`
* `$ micromamba env create -f condaenv.yml`
* `$ micromamba activate bart-gp-article`
* `(bart-gp-article) $ pip install -r requirements.txt`

## How to run scripts

From the `code` directory, do

```sh
(bart-gp-article) $ python path/to/script.py
```

Or, if you prefer to use IPython:

```sh:
(bart-gp-article) $ pip install ipython
(bart-gp-article) $ ipython
In [1]: run path/to/script.py
```

The scripts save figures alongside the script file that produces them.

The scripts that take a long time to run (hours to days) can be hot-interrupted and restarted without losing progress.

## Scripts

* `acic2022/loop/driver.py`: runs the synthetic data test, takes about 1 day
* `acic2022/loop/analysis.py`: makes figures 5, 14, 15, using the output of `driver.py`
* `comp42/comp42.py`: run benchmark on real data, takes about 1 day
* `comp42/articleplot*.py`: scripts to make figures 4, 8, 9, 10, 11 with the output of `comp42.py`
* `comparison_r_packages/comparison.py`: makes figure 12, takes about 10 minutes
* `testnd2/testnd2.py`: computes the BART kernel at various accuracies, takes O(days)
* `testnd2/testnd2-plot2.py`: makes figure 6 with the output of `testnd2.py`
* `checkprior.py`: makes figure 7, takes about 1 hour
* `figcounts.py`: makes figure 2
* `plotcov.py`: makes figure 3

## Original data sources

* `datasets/nipsdata/`: https://sites.google.com/site/hughchipman/Home
* `acic2022/track2_20220404/`, `acic2022/results/`, `acic2022/practice_year_aux/`: https://acic2022.mathematica.org/data

## Troubleshooting

If there is a problem, [open a new issue on github](https://github.com/Gattocrucco/bart-gp-article/issues).
