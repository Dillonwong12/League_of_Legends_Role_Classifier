# League of Legends Role Classifier
Utilizes the K-NN and Decision Trees supervised machine learning algorithms to classify records into roles based on gameplay statistics with approximately 85% F1 score and 81% accuracy. 
- Data pre-processing and wrangling
- Feature selection based on normalised mutual information scores
- K-fold cross validation for hyperparameter tuning and model selection


## Usage
1. Retrieve datasets from https://www.kaggle.com/datasets/andrewasuter/lol-challenger-soloq-data-jan-krnaeuw
    - Andrew Suter, “LoL Challenger Soloq Data (Jan, Kr-Na-Euw).” Kaggle, 2022, doi: 10.34740/KAGGLE/DSV/3193532.

2. Run the following command in terminal: 
```bash
python Role_Classifier.py
```


## Structure Overview
```
League_of_Legends_Role_Classifier
│   README.md
│   Role_Classifier.py   # The main Python script
│
└───Outliers Not Removed  # Contains graphical output produced when the script is run without removing outliers
│
└───Outliers Removed  # Contains graphical output produced when the script is run after removing outliers
```


## Credits
Team members:
1. Angeline Cassie Ganily
2. Christhalia Sanjaya
3. Joshua Ch'ng Wei Han
