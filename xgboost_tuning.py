import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_slice
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from pandas import read_csv
import plotly


class XGBoostTuning:
    def __init__(self, path):
        self.data = read_csv(path)
        self.study = optuna.create_study(direction='maximize', sampler=TPESampler())

    def launch_study(self):
        x = self.data.iloc[:, 1:]
        y = self.data.iloc[:, 0]
        self.study.optimize(lambda trial: self.objective(trial, x, y), n_trials=500)

        print('Best trial: score {},\nparams {}'.format(self.study.best_trial.value, self.study.best_trial.params))

        plotly.io.show(plot_optimization_history(self.study))
        plotly.io.show(plot_slice(self.study))

    def objective(self, trial: Trial, x, y) -> float:

        train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=0)

        param = {
            'n_estimators': trial.suggest_int('n_estimators', 20, 1000),
            'max_depth': trial.suggest_int('max_depth', 2, 25),
            'reg_alpha': trial.suggest_int('reg_alpha', 0, 5),
            'reg_lambda': trial.suggest_int('reg_lambda', 0, 5),
            'min_child_weight': trial.suggest_int('min_child_weight', 0, 5),
            'gamma': trial.suggest_int('gamma', 0, 5),
            'eta': trial.suggest_loguniform('eta', 0.005, 0.5),
            'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.1, 1, 0.01),
            'eval_metric': 'mlogloss',
            'use_label_encoder': False
        }

        model = XGBClassifier(**param)
        model.fit(train_x, train_y)
        return cross_val_score(model, test_x, test_y).mean()
