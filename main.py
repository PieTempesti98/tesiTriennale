from modelLaunch import ModelLaunch
from xgboost_tuning import XGBoostTuning

model = ModelLaunch('data/istanbulCalls.csv')

# test del modello Gradient Boosting
model.launch('Gradient Boosting')

# test del modello XGBoost
model.launch('XGBoost')

# istanziazione e lancio del tuning Optuna per XGBoost
tuner = XGBoostTuning(model.path)
tuner.launch_study()
