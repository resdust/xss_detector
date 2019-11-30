# xss_detector
A real-time detector deployed in server end for xss. (It will be like that)

## frame
- *capture internet traffic packets*
- *process raw data*
- machine learning claciffier
  - *normal machine learning method using just evil url features*
    Scripts in code/Paper1.py
  - *pso-bp nueral network using url features & TCP flow features*
    Scripts in pso.py, but maybe not works.
  - deep nueral network using text classification features
  - semi-supervised classifer using evil url features
- paper

## TODO
All except the italics.

## Dataset
Datasets are in `data` folder.
### training set
- normal_data_162k_dl.csv
  Got from https://github.com/SparkSharly/DL_for_xss/tree/master/data and removed repeating datas.
- xss_data_28k_xssed.csv
  Got from http://xssed.com real efficient xss attacks, also removed repeating datas.
### test set
- normal_data_39k_bupt.csv
  Got from BUPT college network GET requests, removed repeatings.
- xss_data_3k_xsstrike.csv
  Got from https://github.com/s0md3v/XSStrike xss attack tool suit, colleted 3 thousands generated payloads.
  Xssed environment is built in linux server end, using https://github.com/aj00200/xssed project.
