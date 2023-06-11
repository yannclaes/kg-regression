import torch
import wandb
import argparse
import settings
import torch.nn as nn

from pdp_model import Fp_Model, Fa_Model
from utils import load_dataset

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
parser.add_argument('--fa_model', type=str, default='mlp')
parser.add_argument('--pdp_method', type=str, default='mlp')
parser.add_argument('--tree_model', type=str, default='rf')
parser.add_argument('--optim_loss', type=str, default='no_loss')
parser.add_argument('--b_size', type=int, default=25)
parser.add_argument('--fp_lr', type=int, default=0.05)
parser.add_argument('--dataset_id', type=int, default=0)
parser.add_argument('--extrapolation', type=str, default='no')
parser.add_argument('--filter_fa', type=str, default='yes')
args = parser.parse_args()

args.grid_resolution = 300 if args.fp_term == 'second' else 50
extrapolation = args.extrapolation == 'yes'
boosting = args.tree_model == 'boosting'

if args.fa_model == 'mlp':
    wandb.init(project='pdp', entity='yannclaes', config=args)
    wandb.define_metric("mlp/epoch")
    wandb.define_metric("mlp/*", step_metric="mlp/epoch")

if args.problem in ['friedman1', 'corr_friedman1']:
    best_config = {'mlp': {'fp_learning_rate': 0.005, 'mlp_learning_rate': 0.005, 'num_layers': 2, 'hidden_size': 15, 'standardize': True, 'out_size': 1},
                   'rf': {'fp_learning_rate': 0.005, 'N': 500, 'max_d': None, 'max_features': None, 'min_samples_split': 5, 'learning_rate_trees': None},
                   'boosting': {'fp_learning_rate': 0.005, 'N': 700, 'max_d': 2, 'learning_rate_trees': 0.3, 'max_features': None, 'min_samples_split': None}}
elif args.problem in ['linear_data', 'linear_data_no_covariance', 'overlap_data', 'overlap_data_no_covariance']:
    best_config = {'mlp': {'fp_learning_rate': 0.005, 'mlp_learning_rate': 0.005, 'num_layers': 2, 'hidden_size': 10, 'standardize': True, 'out_size': 1},
                   'rf': {'fp_learning_rate': 0.005, 'N': 500, 'max_d': None, 'max_features': None, 'min_samples_split': 5, 'learning_rate_trees': None},
                   'boosting': {'fp_learning_rate': 0.005, 'N': 400, 'max_d': 2, 'learning_rate_trees': 0.3, 'max_features': None, 'min_samples_split': None}}
elif args.problem in ['house16H', 'california', 'income', 'fish', 'boston', 'power_plant', 'protein', 'auto', 'concrete']:
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

data, mean_std, params, _ = load_dataset(data_path,
                                         extrapolation=extrapolation,
                                         problem=args.problem,
                                         standardize=True)
train_X, train_y, valid_X, valid_y, test_X, test_y = data
train_mean_X, train_std_X, train_mean_y, train_std_y = mean_std
seed = 30

config = best_config[args.pdp_method]
config['fp_inputs'] = fp_inputs[args.problem][args.fp_term]
config['fa_inputs'] = fa_inputs[args.problem]['None']

# Load best Fp model
fp_model = Fp_Model(problem=args.problem,
                    term=args.fp_term,
                    dataset_id=args.dataset_id,
                    fp_inputs=config['fp_inputs'],
                    learning_rate=config['fp_learning_rate'],
                    b_size=args.b_size,
                    saving=True,
                    fp_with_constant=True,
                    pdp_method=args.pdp_method,
                    pdp_config=config,
                    extrapolation=extrapolation,
                    loss=args.optim_loss,
                    seed=seed)
fp_model.load()

# Prepare final residual dataset
with torch.no_grad():
    _train_y = train_y - fp_model.predict(train_X)
    _valid_y = valid_y - fp_model.predict(valid_X)

train_mean_y = torch.mean(_train_y)
train_std_y = torch.std(_train_y)

if args.fa_model == 'trees':
    config = best_config[args.tree_model]

    if args.filter_fa == 'yes':
        config['fa_inputs'] = fa_inputs[args.problem][args.fp_term]
    else:
        config['fa_inputs'] = fa_inputs[args.problem]['None']

    fa_model = Fa_Model(problem=args.problem,
                        term=args.fp_term,
                        N=config['N'],
                        d=config['max_d'],
                        boosting=boosting,
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
                        filter_fa=args.filter_fa == 'yes',
                        extrapolation=extrapolation,
                        seed=seed)
else:
    config = best_config['mlp']

    if args.filter_fa == 'yes':
        config['fa_inputs'] = fa_inputs[args.problem][args.fp_term]
    else:
        config['fa_inputs'] = fa_inputs[args.problem]['None']

    fa_model = Fa_Model(problem=args.problem,
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
                        filter_fa=args.filter_fa == 'yes',
                        extrapolation=extrapolation,
                        verbose=True,
                        seed=seed)

torch.manual_seed(seed)
fa_model.train(train_X, _train_y, valid_X, _valid_y, wandb_logging=args.fa_model == 'mlp')
