import os
import torch
import wandb
import argparse
import settings
import numpy as np
import torch.nn as nn

from pdp_model import Fp_Model, Fa_Model
from utils import load_dataset, compute_pdp


settings.init()

fp_inputs = {'friedman1': {'first': [0, 1], 'second': [2], 'third': [3, 4], 'None': []},
             'corr_friedman1': {'first': [0, 1], 'second': [2], 'third': [3, 4], 'None': []},
             'linear_data': {'first': [0], 'None': []},
             'overlap_data': {'first': [0], 'None': []},
             'power_plant': {'first': [0], 'None': []},
             'concrete': {'first': [0], 'None': []}}
fa_inputs = {'friedman1': {'first': [2, 3, 4, 5, 6, 7, 8, 9],
                           'second': [0, 1, 3, 4, 5, 6, 7, 8, 9],
                           'third': [0, 1, 2, 5, 6, 7, 8, 9],
                           'None': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]},
             'corr_friedman1': {'first': [2, 3, 4, 5, 6, 7, 8, 9],
                                'second': [0, 1, 3, 4, 5, 6, 7, 8, 9],
                                'third': [0, 1, 2, 5, 6, 7, 8, 9],
                                'None': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]},
             'linear_data': {'first': [1], 'None': [0, 1]},
             'overlap_data': {'first': [1], 'None': [0, 1]},
             'power_plant': {'first': [1, 2, 3], 'None': range(4)},
             'concrete': {'first': [1, 2, 3, 4, 5, 6, 7, 8], 'None': range(9)}}

parser = argparse.ArgumentParser()
parser.add_argument('--problem', type=str, default='friedman1')
parser.add_argument('--problem_setting', type=str, default='entire')
parser.add_argument('--fp_term', type=str, default='third')
parser.add_argument('--pdp_method', type=str, default='mlp')
parser.add_argument('--optim_loss', type=str, default='no_loss')
parser.add_argument('--b_size', type=int, default=25)
parser.add_argument('--fp_lr', type=int, default=0.005)
parser.add_argument('--dataset_id', type=int, default=0)
parser.add_argument('--extrapolation', type=str, default='no')
parser.add_argument('--n_repeats', type=int, default=40)
args = parser.parse_args()

args.grid_resolution = 300 if args.fp_term == 'second' else 50
extrapolation = args.extrapolation == 'yes'
fp_model = None

if args.problem in ['friedman1', 'corr_friedman1']:
    best_config = {'mlp': {'fp_learning_rate': 0.005, 'mlp_learning_rate': 0.005, 'num_layers': 2, 'hidden_size': 15, 'standardize': True, 'out_size': 1},
                   'rf': {'fp_learning_rate': 0.005, 'N': 500, 'max_d': None, 'max_features': None, 'min_samples_split': 5, 'learning_rate_trees': None},
                   'boosting': {'fp_learning_rate': 0.005, 'N': 700, 'max_d': 2, 'learning_rate_trees': 0.3, 'max_features': None, 'min_samples_split': None}}
elif args.problem in ['linear_data', 'overlap_data']:
    best_config = {'mlp': {'fp_learning_rate': 0.005, 'mlp_learning_rate': 0.005, 'num_layers': 2, 'hidden_size': 10, 'standardize': True, 'out_size': 1},
                   'rf': {'fp_learning_rate': 0.005, 'N': 500, 'max_d': None, 'max_features': None, 'min_samples_split': 5, 'learning_rate_trees': None},
                   'boosting': {'fp_learning_rate': 0.005, 'N': 400, 'max_d': 2, 'learning_rate_trees': 0.3, 'max_features': None, 'min_samples_split': None}}
elif args.problem in ['power_plant', 'concrete']:
    best_config = {'mlp': {'fp_learning_rate': 0.005, 'mlp_learning_rate': 0.005, 'num_layers': 2, 'hidden_size': 30, 'standardize': True, 'out_size': 1},
                   'rf': {'fp_learning_rate': 0.005, 'N': 200, 'max_d': None, 'max_features': None, 'min_samples_split': 5, 'learning_rate_trees': None},
                   'boosting': {'fp_learning_rate': 0.005, 'N': 300, 'max_d': 5, 'learning_rate_trees': 0.3, 'max_features': None, 'min_samples_split': None}}

# STEP 0: load dataset
if extrapolation:
    if args.problem in ['power_plant', 'concrete']:
        data_path = f'data_extrapolation/{args.problem}/{args.dataset_id}'
else:
    if args.problem in ['friedman1', 'corr_friedman1']:
        data_path = f'data/{args.problem}/{args.problem_setting}/{args.dataset_id}'
    elif args.problem in ['linear_data', 'overlap_data', 'power_plant', 'concrete']:
        data_path = f'data/{args.problem}/{args.dataset_id}'

data, mean_std, params, learning_set_X = load_dataset(data_path,
                                                      extrapolation=extrapolation,
                                                      problem=args.problem,
                                                      standardize=True)
train_X, train_y, valid_X, valid_y, test_X, test_y = data
train_mean_X, train_std_X, train_mean_y, train_std_y = mean_std
param_tol = 5e-3
params_prev = {'p0': np.inf, 'p1': np.inf, 'p2': np.inf, 'p3': np.inf, 'p4': np.inf,
               'p5': np.inf, 'beta': np.inf}
seed = 30
best_global_loss, best_pdp_loss, prev_dist = np.inf, np.inf, np.inf

wandb.init(project='pdp', entity='yannclaes', config=args)
wandb.define_metric("fp/epoch")
wandb.define_metric("fp/*", step_metric="fp/epoch")
wandb.define_metric("mlp/epoch")
wandb.define_metric("mlp/*", step_metric="mlp/epoch")

# Create plot directory
path = f'{args.problem}/{args.fp_term}/plots/{args.pdp_method}'
os.makedirs(path, exist_ok=True)

config = best_config[args.pdp_method]
config['fa_inputs'] = fa_inputs[args.problem]['None']

for repeat in range(args.n_repeats):

    # STEP 1: Fit data-driven model on current residuals
    if fp_model is not None:
        with torch.no_grad():
            _train_y = train_y - fp_model.predict(train_X)
            _valid_y = valid_y - fp_model.predict(valid_X)

        train_mean_y = torch.mean(_train_y)
        train_std_y = torch.std(_train_y)
    else:
        _train_y = train_y
        _valid_y = valid_y


    if args.pdp_method == 'mlp':
        pdp_model = Fa_Model(problem=args.problem,
                             term=args.fp_term,
                             N=None,
                             d=None,
                             boosting=False,
                             learning_rate_trees=None,
                             max_features=None,
                             min_samples_split=None,
                             fa_inputs=config['fa_inputs'],
                             out_size=config['out_size'],
                             learning_rate_mlp=config['mlp_learning_rate'],
                             num_layers=config['num_layers'],
                             hidden_size=config['hidden_size'],
                             b_size=args.b_size,
                             saving=True,
                             fa_mlp=True,
                             pdp_method=args.pdp_method,
                             train_mean_X=train_mean_X,
                             train_std_X=train_std_X,
                             train_mean_y=train_mean_y,
                             train_std_y=train_std_y,
                             dataset_id=args.dataset_id,
                             extrapolation=extrapolation,
                             seed=seed)
    else:
        pdp_model = Fa_Model(problem=args.problem,
                             term=args.fp_term,
                             N=config['N'],
                             d=config['max_d'],
                             boosting=args.pdp_method == 'boosting',
                             learning_rate_trees=config['learning_rate_trees'],
                             max_features=config['max_features'],
                             min_samples_split=config['min_samples_split'],
                             fa_inputs=config['fa_inputs'],
                             out_size=None,
                             learning_rate_mlp=None,
                             num_layers=None,
                             hidden_size=None,
                             b_size=None,
                             saving=True,
                             fa_mlp=False,
                             pdp_method=args.pdp_method,
                             dataset_id=args.dataset_id,
                             extrapolation=extrapolation,
                             seed=seed)

    torch.manual_seed(seed)

    pdp_model.train(train_X, _train_y, valid_X, _valid_y, wandb_logging=True)
    pdp_model.load()

    # Validation phase
    if fp_model is not None:

        fp_pred = fp_model.predict(train_X)
        fa_pred = pdp_model.predict(train_X)
        train_mse = nn.MSELoss()(fp_pred + fa_pred, train_y)

    # STEP 2: Compute partial dependence on the current Fa
    relevant_features = fp_inputs[args.problem][args.fp_term]
    pdp_train_y = compute_pdp(pdp_model,
                              args.grid_resolution,
                              lower_X=None,
                              upper_X=None,
                              X=learning_set_X,
                              features=relevant_features,
                              true_total_fn=None,
                              linspace=False)

    # STEP 3: Compute current (Fp, C) pdp residuals
    if repeat != 0:
        with torch.no_grad():
            fp_train_y = pdp_train_y + fp_model.predict(learning_set_X)

    fp_model = Fp_Model(problem=args.problem,
                        term=args.fp_term,
                        dataset_id=args.dataset_id,
                        fp_inputs=relevant_features,
                        learning_rate=config['fp_learning_rate'],
                        b_size=args.b_size,
                        saving=True,
                        fp_with_constant=True,
                        pdp_method=args.pdp_method,
                        pdp_config=config,
                        extrapolation=extrapolation,
                        loss=args.optim_loss,
                        seed=seed)
    if repeat > 0:
        fp_model.model.load_state_dict(best_weights)

    # STEP 4: Train (Fp, C) on pdp
    if repeat == 0:
        _, best_weights, distance = fp_model.train(learning_set_X, pdp_train_y, learning_set_X, pdp_train_y, wandb_logging=True)
    else:
        _, best_weights, distance = fp_model.train(learning_set_X, fp_train_y, learning_set_X, fp_train_y, wandb_logging=True)
    fp_model.model.load_state_dict(best_weights)

    if fp_model is not None:

        fp_pred = fp_model.predict(learning_set_X)
        if repeat == 0:
            train_mse = nn.MSELoss()(fp_pred, pdp_train_y)
        else:
            train_mse = nn.MSELoss()(fp_pred, fp_train_y)


    wandb.run.summary['current repeat (done)'] = repeat+1

    # Parameter convergence criterion
    if args.problem in ['friedman1', 'corr_friedman1']:
        if args.fp_term == 'first':
            p0_dist = abs(params_prev['p0'] - fp_model.model.p0.item()) / params_prev['p0']
            p1_dist = abs(params_prev['p1'] - fp_model.model.p1.item()) / params_prev['p1']
            if p0_dist < param_tol and p1_dist < param_tol:
                print(f'Reached convergence at repetition number {repeat}')
                break
            else:
                params_prev['p0'] = fp_model.model.p0.item()
                params_prev['p1'] = fp_model.model.p1.item()

        elif args.fp_term == 'second':
            p2_dist = abs(params_prev['p2'] - fp_model.model.p2.item()) / params_prev['p2']
            p3_dist = abs(params_prev['p3'] - fp_model.model.p3.item()) / params_prev['p3']
            if p2_dist < param_tol and p3_dist < param_tol:
                print(f'Reached convergence at repetition number {repeat}')
                break
            else:
                params_prev['p2'] = fp_model.model.p2.item()
                params_prev['p3'] = fp_model.model.p3.item()

        elif args.fp_term == 'third':
            p4_dist = abs(params_prev['p4'] - fp_model.model.p4.item()) / params_prev['p4']
            p5_dist = abs(params_prev['p5'] - fp_model.model.p5.item()) / params_prev['p5']
            if p4_dist < param_tol and p5_dist < param_tol:
                print(f'Reached convergence at repetition number {repeat}')
                break
            else:
                params_prev['p4'] = fp_model.model.p4.item()
                params_prev['p5'] = fp_model.model.p5.item()

    elif args.problem == 'linear_data':
        beta_dist = abs(params_prev['beta'] - fp_model.model.beta.item()) / params_prev['beta']
        if beta_dist < param_tol:
            print(f'Reached convergence at repetition number {repeat}')
            break
        else:
            params_prev['beta'] = fp_model.model.beta.item()

    elif args.problem == 'overlap_data':
        beta_dist = abs(params_prev['beta'] - fp_model.model.beta.item()) / params_prev['beta']
        if beta_dist < param_tol:
            print(f'Reached convergence at repetition number {repeat}')
            break
        else:
            params_prev['beta'] = fp_model.model.beta.item()

    elif args.problem in ['power_plant', 'concrete']:
        p0_dist = abs(params_prev['p0'] - fp_model.model.p0.item()) / params_prev['p0']
        if p0_dist < param_tol:
            print(f'Reached convergence at repetition number {repeat}')
            break
        else:
            params_prev['p0'] = fp_model.model.p0.item()

# Last Fa for final repeat step
if fp_model is not None:
    with torch.no_grad():
        _train_y = train_y - fp_model.predict(train_X)
        _valid_y = valid_y - fp_model.predict(valid_X)

    train_mean_y = torch.mean(_train_y)
    train_std_y = torch.std(_train_y)
else:
    _train_y = train_y
    _valid_y = valid_y


if args.pdp_method == 'mlp':
    pdp_model = Fa_Model(problem=args.problem,
                         term=args.fp_term,
                         N=None,
                         d=None,
                         boosting=False,
                         learning_rate_trees=None,
                         max_features=None,
                         min_samples_split=None,
                         fa_inputs=config['fa_inputs'],
                         out_size=config['out_size'],
                         learning_rate_mlp=config['mlp_learning_rate'],
                         num_layers=config['num_layers'],
                         hidden_size=config['hidden_size'],
                         b_size=args.b_size,
                         saving=True,
                         fa_mlp=True,
                         pdp_method=args.pdp_method,
                         train_mean_X=train_mean_X,
                         train_std_X=train_std_X,
                         train_mean_y=train_mean_y,
                         train_std_y=train_std_y,
                         dataset_id=args.dataset_id,
                         extrapolation=extrapolation,
                         seed=seed)
else:
    pdp_model = Fa_Model(problem=args.problem,
                         term=args.fp_term,
                         N=config['N'],
                         d=config['max_d'],
                         boosting=args.pdp_method == 'boosting',
                         learning_rate_trees=config['learning_rate_trees'],
                         max_features=config['max_features'],
                         min_samples_split=config['min_samples_split'],
                         fa_inputs=config['fa_inputs'],
                         out_size=None,
                         learning_rate_mlp=None,
                         num_layers=None,
                         hidden_size=None,
                         b_size=None,
                         saving=True,
                         fa_mlp=False,
                         pdp_method=args.pdp_method,
                         dataset_id=args.dataset_id,
                         extrapolation=extrapolation,
                         seed=seed)

torch.manual_seed(seed)

pdp_model.train(train_X, _train_y, valid_X, _valid_y, wandb_logging=True)
pdp_model.load()

# Validation phase
if args.optim_loss == 'no_loss':
    fp_model.model.load_state_dict(best_weights)
    torch.save(fp_model.model.state_dict(), fp_model.save_path)
