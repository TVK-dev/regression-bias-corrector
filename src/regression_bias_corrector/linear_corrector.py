import numpy as np
from sklearn.linear_model import LinearRegression, TheilSenRegressor, RANSACRegressor
from ._version import __version_tuple__, __version__
import yaml


class LinearBiasCorrector:
    def __init__(self, regressor_type='linear', regressor_params=None):
        self.regressor_type = regressor_type            # 'linear', 'theilsen', 'ransac'
        self.regressor_params = regressor_params or {}  # dict with params for models
        self.rotation_lr = None
        self.intercept_lr = None

    def _get_coefficients(self, model):
        if self.regressor_type == 'linear':
            return model.coef_[0], model.intercept_
        elif self.regressor_type == 'theilsen':
            # TheilSenRegressor имеет coef_ и intercept_ напрямую
            return model.coef_[0], model.intercept_
        elif self.regressor_type == 'ransac':
            # RANSACRegressor — обёртка, коэффициенты в estimator_
            return model.estimator_.coef_[0], model.estimator_.intercept_
        else:
            raise ValueError(f"Unknown regressor_type: {self.regressor_type}")
            
    def fit(self, true_train, pred_train):
        true_train = np.array(true_train).reshape(-1, 1)
        pred_train = np.array(pred_train)

        if self.regressor_type == 'linear':
            ln = LinearRegression()
        elif self.regressor_type == 'theilsen':
            ln = TheilSenRegressor(**self.regressor_params)
        elif self.regressor_type == 'ransac':
            params = dict(**self.regressor_params)
            ln = RANSACRegressor(**params)
        else:
            raise ValueError(f"Unknown regressor_type: {self.regressor_type}")

        ln.fit(true_train, pred_train)
        #slope, intercept = ln.coef_[0], ln.intercept_
        slope, intercept = self._get_coefficients(ln)
        self.rotation_lr, self.intercept_lr = self.transform_line_linreg(slope, 1, intercept)

    def predict(self, pred_train):
        if self.rotation_lr is None or self.intercept_lr is None:
            raise ValueError("Model must be fitted before predicting.")
        return np.array(pred_train) * self.rotation_lr + self.intercept_lr

    def transform_line_linreg(self, slope1, slope2, intercept):
        rotation_factor = slope2 / slope1
        d = -intercept / slope1
        return rotation_factor, d

    @classmethod
    def load(cls, path_to_yml):
        with open(path_to_yml, 'r') as fin:
            model_dict = yaml.full_load(fin)
        corrector = LinearBiasCorrector(
            regressor_type=model_dict.get('regressor_type', 'linear'),
            regressor_params=model_dict.get('regressor_params', {})
        )
        corrector.rotation_lr = model_dict['rotation_lr']
        corrector.intercept_lr = model_dict['intercept_lr']
        return corrector

    def save(self, path_to_yml):
        model_dict = {
            'model_name': 'linear_bias_correction',
            'version': __version__,
            'version_tuple': __version_tuple__,
            'regressor_type': self.regressor_type,
            'regressor_params': self.regressor_params,  
            'rotation_lr': self.rotation_lr.item(),
            'intercept_lr': self.intercept_lr.item()
        }
        with open(path_to_yml, 'w') as fout:
            yaml.dump(model_dict, fout, default_flow_style=False)

    def __str__(self):
        return f'Linear bias corrector (type={self.regressor_type}): a={self.rotation_lr} b={self.intercept_lr}'
