import os
import dill
import torch
import numpy as np
import pandas as pd

from math import pi
from copy import deepcopy
from models import friedman_model, linear_model, overlap_model, \
    power_plant_model, concrete_model


def sample_parameters(problem, n_features, seed=None):
    params = {}

    rng = np.random.default_rng(seed=seed)

    if problem in ['friedman1', 'corr_friedman1']:
        params['p0'] = rng.standard_normal() + 10
        params['p1'] = rng.standard_normal() * 0.5 + np.pi
        params['p2'] = rng.standard_normal() + 20
        params['p3'] = rng.standard_normal() * 0.1 + 0.5
        params['p4'] = rng.standard_normal() + 10
        params['p5'] = rng.standard_normal() + 5
        params['noise_mean'] = 0
        params['noise_std'] = 1

    params['n_features'] = n_features
    return params

def make_friedman1_data(params, n_sim_ls, n_sim_test, seed=None, term=None,
    problem='friedman1', selected_pairs=None):

    rng = np.random.default_rng(seed=seed)

    # Generate input features according to input problem
    if problem == 'friedman1':
        X = rng.uniform(size=(n_sim_ls, params['n_features']))
        X_test = rng.uniform(size=(n_sim_test, params['n_features']))

    elif problem == 'corr_friedman1':

        # Mean and covariance
        mean = np.ones(params['n_features']) * 0.5
        cov = np.ones((params['n_features'], params['n_features'])) * 0.3
        for i in range(len(cov)):
            cov[i, i] = 0.75

        X = rng.multivariate_normal(mean=mean, cov=cov, size=n_sim_ls)
        X_test = rng.multivariate_normal(mean=mean, cov=cov, size=n_sim_test)

        # Randomly flip some variables
        to_flip = rng.uniform(size=params['n_features']) < 0.5
        X[:, to_flip] = 1 - X[:, to_flip]
        X_test[:, to_flip] = 1 - X_test[:, to_flip]

        # Rescale input features
        X /= 3
        X_test /= 3

    noise = params['noise_std'] * rng.standard_normal(size=(n_sim_ls)) + params['noise_mean']
    noise_test = params['noise_std'] * rng.standard_normal(size=(n_sim_test)) + params['noise_mean']

    if term == 'first':
        y = params['p0'] * np.sin(params['p1'] * X[:, 0] * X[:, 1]) + noise
        y_test = params['p0'] * np.sin(params['p1'] * X_test[:, 0] * X_test[:, 1]) + noise_test
    elif term == 'second':
        y = params['p2'] * (X[:, 2] - params['p3']) ** 2 + noise
        y_test = params['p2'] * (X_test[:, 2] - params['p3']) ** 2 + noise_test
    elif term == 'third':
        y = params['p4'] * X[:, 3] + params['p5'] * X[:, 4] + noise
        y_test = params['p4'] * X_test[:, 3] + params['p5'] * X_test[:, 4] + noise_test
    else:
        y = params['p0'] * np.sin(params['p1'] * X[:, 0] * X[:, 1]) + \
                params['p2'] * (X[:, 2] - params['p3']) ** 2 + \
                params['p4'] * X[:, 3]  + params['p5'] * X[:, 4] + \
                noise
        y_test = params['p0'] * np.sin(params['p1'] * X_test[:, 0] * X_test[:, 1]) + \
                params['p2'] * (X_test[:, 2] - params['p3']) ** 2 + \
                params['p4'] * X_test[:, 3]  + params['p5'] * X_test[:, 4] + \
                noise_test

    return ((X, y), (X_test, y_test), selected_pairs)

def make_linear_data(n_samples_ls, n_samples_test, seed=None, no_covariance=False, noise=True):

    rng = np.random.default_rng(seed=seed)

    mu = np.array([0, 0])
    if no_covariance:
        sigma = np.array([[2, 0], [0, 3]])
    else:
        sigma = np.array([[2, 2.25], [2.25, 3]])

    beta, gamma = -0.5, 1.

    X = rng.multivariate_normal(mu, sigma, size=n_samples_ls)
    X_test = rng.multivariate_normal(mu, sigma, size=n_samples_test)

    if noise:
        noise = 0.5 * rng.standard_normal(size=(n_samples_ls))
        noise_test = 0.5 * rng.standard_normal(size=(n_samples_test))
    else:
        noise = 0.0 * rng.standard_normal(size=(n_samples_ls))
        noise_test = 0.0 * rng.standard_normal(size=(n_samples_test))

    y = beta * X[:, 0] + gamma * X[:, 1] + noise
    y_test = beta * X_test[:, 0] + gamma * X_test[:, 1] + noise_test

    return ((X, y), (X_test, y_test))

def make_overlap_data(n_samples_ls, n_samples_test, seed=None, no_covariance=False, noise=True):

    rng = np.random.default_rng(seed=seed)

    mu = np.array([0, 0])
    if no_covariance:
        sigma = np.array([[2, 0], [0, 3]])
    else:
        sigma = np.array([[2, 2.25], [2.25, 3]])

    beta, gamma, delta = 0.2, 1.5, 1

    X = rng.multivariate_normal(mu, sigma, size=n_samples_ls)
    X_test = rng.multivariate_normal(mu, sigma, size=n_samples_test)

    if noise:
        noise = 0.5 * rng.standard_normal(size=(n_samples_ls))
        noise_test = 0.5 * rng.standard_normal(size=(n_samples_test))
    else:
        noise = 0.0 * rng.standard_normal(size=(n_samples_ls))
        noise_test = 0.0 * rng.standard_normal(size=(n_samples_test))

    y = beta * X[:, 0] ** 2 + np.sin(gamma * X[:, 0]) + delta * X[:, 1] + noise
    y_test = beta * X_test[:, 0] ** 2 + np.sin(gamma * X_test[:, 0]) + delta * X_test[:, 1] + noise_test

    return ((X, y), (X_test, y_test))

def make_power_plant_data(n_samples_ls=None, n_samples_test=None, y_low=None, y_high=None, seed=None):

    rng = np.random.default_rng(seed=seed)
    file = 'data/power_plant.xlsx'

    data = pd.read_excel(file).to_numpy()
    rng.shuffle(data)

    # Extract input and target
    X, y = data[:, :-1], data[:, -1]

    # Extract learning and test sets
    if y_low is None and y_high is None:
        X_LS, y_LS = X[:n_samples_ls], y[:n_samples_ls]
        X_test, y_test = X[-n_samples_test:], y[-n_samples_test:]
    else:
        if y_low == y_high:
            LS_mask = y >= y_low
            TS_mask = y < y_low
        else:
            LS_mask = (y <= y_low) | (y >= y_high)
            TS_mask = (y > y_low) & (y < y_high)
        X_LS, y_LS = X[LS_mask], y[LS_mask]
        X_test, y_test = X[TS_mask], y[TS_mask]

    return((X_LS, y_LS), (X_test, y_test))

def make_concrete_data(n_samples_ls=None, n_samples_test=None, y_low=None, y_high=None, seed=None):

    rng = np.random.default_rng(seed=seed)
    file = 'data/Concrete_Data.xls'

    data = pd.read_excel(file)
    data = data.dropna()
    data = data.sample(frac=1, random_state=rng)
    data = data.apply(pd.to_numeric)
    data = data.to_numpy()

    X, y = data[:, :-1], data[:, -1]
    cement_to_water = np.expand_dims(X[:, 0] / X[:, 3], axis=1)
    X = np.concatenate([cement_to_water, X], axis=1)

    # Extract learning and test sets
    if y_low is None and y_high is None:
        X_LS, y_LS = X[:n_samples_ls], y[:n_samples_ls]
        X_test, y_test = X[n_samples_ls:], y[n_samples_ls:]
    else:
        if y_low == y_high:
            LS_mask = y >= y_low
            TS_mask = y < y_low
        else:
            LS_mask = (y <= y_low) | (y >= y_high)
            TS_mask = (y > y_low) & (y < y_high)
        X_LS, y_LS = X[LS_mask], y[LS_mask]
        X_test, y_test = X[TS_mask], y[TS_mask]

    return((X_LS, y_LS), (X_test, y_test))

def save_model(model, output, name):
    path = os.path.join(output, name + '.pkl')
    with open(path, 'wb') as f:
        dill.dump(model, f)

def load_model(path):
    with open(path, 'rb') as f:
        model = dill.load(f)
    return model

class mse_objective(object):

    def __init__(self, phy_term, lambda_j, lambda_loss):
        self.phy_term = phy_term
        self.lambda_j = lambda_j
        self.lambda_loss = lambda_loss

    def __call__(self, y_true, y_pred):
        pred_next_state = self.phy_term + y_pred

        if self.lambda_loss:
            grad = - 2 * self.lambda_j * (y_true - pred_next_state) + 2 * y_pred
            hess = []
            for i in range(len(y_pred)):
                hess.append(2 * self.lambda_j + 2)
        else:
            grad = - 2 * (y_true - pred_next_state)
            hess = []
            for i in range(len(y_pred)):
                hess.append(2)

        return grad, hess

def make_known_model(problem, term, inc, constant=False, real_params=None,
    seed=None):
    if problem in ['friedman1', 'corr_friedman1']:
        model = friedman_model(term, inc, params=real_params, constant=constant, seed=seed)
    elif problem == 'linear_data':
        model = linear_model(constant=constant, params=real_params, seed=seed)
    elif problem == 'overlap_data':
        model = overlap_model(constant=constant, params=real_params, seed=seed)
    elif problem == 'power_plant':
        model = power_plant_model(constant=constant, params=real_params, seed=seed)
    elif problem == 'concrete':
        model = concrete_model(constant=constant, params=real_params, seed=seed)
    return model

def get_batch(x, y, indices, device='cpu'):
    return x[indices, ...].to(device), y[indices, ...].to(device)

def compute_pdp(model, grid_resolution, lower_X, upper_X, X, features, true_total_fn, linspace=False):

    if linspace:

        # Generate all evaluation points for the input features
        linspaces = torch.zeros(len(features), grid_resolution)
        for i, bounds in enumerate(zip(lower_X, upper_X)):
            _linspace = np.linspace(bounds[0], bounds[1], grid_resolution, endpoint=True)
            linspaces[i] = torch.from_numpy(_linspace)

        # true_total = torch.zeros(*[grid_resolution for _ in range(len(features))])
        y = torch.zeros(*[grid_resolution for _ in range(len(features))])

        if len(features) == 1:
            for i in range(grid_resolution):
                _x = deepcopy(X).float()
                _x[:, features] = linspaces[0, i]

                with torch.no_grad():
                    pred = model.predict(_x)

                y[i] = pred.mean()
                # true_total[i] = true_total_fn(_x).mean()

        elif len(features) == 2:
            for i in range(grid_resolution):
                for j in range(grid_resolution):
                    _x = deepcopy(X).float()
                    _x[:, features] = torch.cat([linspaces[0, j].unsqueeze(0), linspaces[1, i].unsqueeze(0)])

                    with torch.no_grad():
                        pred = model.predict(_x)

                    y[i, j] = pred.mean()
                    # true_total[i, j] = true_total_fn(_x).mean()

        else:
            sys.exit('The number of relevant features should be <= 2.')

        return y

    else:
        true_total = torch.zeros(len(X))
        y = torch.zeros(len(X)).double()

        for i in range(len(X)):
            _x = deepcopy(X)
            _x[:, features] = X[i, features]

            with torch.no_grad():
                pred = model.predict(_x)

            y[i] = pred.mean()
            # true_total[i] = true_total_fn(_x).mean()

        return y

def load_dataset(data_path, problem, extrapolation=False, standardize=False, train_size=0.5):
    train_X_path = os.path.join(data_path, 'learning_set_X.pt')
    train_y_path = os.path.join(data_path, 'learning_set_y.pt')

    if extrapolation:
        learning_set_X = torch.load(train_X_path)[:200]
        learning_set_y = torch.load(train_y_path)[:200]
    else:
        learning_set_X = torch.load(train_X_path)[:200]
        learning_set_y = torch.load(train_y_path)[:200]
    val_set_end = int(train_size * len(learning_set_X))

    train_X = learning_set_X[val_set_end:]
    train_y = learning_set_y[val_set_end:]
    valid_X = learning_set_X[:val_set_end]
    valid_y = learning_set_y[:val_set_end]

    test_X = torch.load(os.path.join(data_path, 'test_set_X.pt'))
    test_y = torch.load(os.path.join(data_path, 'test_set_y.pt'))

    if standardize:
        train_X_mean = torch.mean(train_X, dim=0)
        train_X_std = torch.std(train_X, dim=0)
        train_y_mean = torch.mean(train_y, dim=0)
        train_y_std = torch.std(train_y, dim=0)

        if problem in ['boston', 'concrete', 'power_plant']:
            train_X = (train_X - train_X_mean) / train_X_std
            train_y = (train_y - train_y_mean) / train_y_std
            valid_X = (valid_X - train_X_mean) / train_X_std
            valid_y = (valid_y - train_y_mean) / train_y_std
            test_X = (test_X - train_X_mean) / train_X_std
            test_y = (test_y - train_y_mean) / train_y_std
            learning_set_X = (learning_set_X - train_X_mean) / train_X_std

            train_X_mean = None
            train_X_std = None
            train_y_mean = None
            train_y_std = None
    else:
        train_X_mean = None
        train_X_std = None
        train_y_mean = None
        train_y_std = None

    if problem in ['friedman1', 'corr_friedman1', 'linear_data', 'overlap_data']:
        with open(os.path.join(data_path, 'params.pkl'), 'rb') as f:
            real_params = dill.load(f)

        for param in real_params:
            real_params[param] = torch.tensor([real_params[param]])
    else:
        real_params = None

    return (train_X, train_y, valid_X, valid_y, test_X, test_y), \
           (train_X_mean, train_X_std, train_y_mean, train_y_std), \
           real_params, learning_set_X
