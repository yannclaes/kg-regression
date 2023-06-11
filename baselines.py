import wandb
import argparse

from dawgz import schedule
from trainer import Trainer


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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='trees_only')
    parser.add_argument('--problem', type=str, default='friedman1')
    parser.add_argument('--problem_setting', type=str, default='entire')
    parser.add_argument('--dataset_id', type=int, default=0)
    parser.add_argument('--filter_fa', type=str, default='yes')
    parser.add_argument('--tree_model', type=str, default='rf')
    parser.add_argument('--extrapolation', type=str, default='no')
    args = parser.parse_args()

    filter_fa = args.filter_fa == 'yes'
    extrapolation = args.extrapolation == 'yes'
    boosting = args.tree_model == 'boosting'
    tree_model = args.tree_model
    incomplete = True
    problem = args.problem
    problem_setting = args.problem_setting
    dataset_id = args.dataset_id
    experiment = args.model_name
    seed = dataset_id

    if problem in ['friedman1', 'corr_friedman1']:
        FP_TERMS_EXP = {'fp_known_then_mlp': ['first', 'second', 'third'],
                        'fp_known_then_trees': ['first', 'second', 'third'],
                        'fp_known_only': ['first', 'second', 'third'],
                        'mlp_only': ['None'],
                        'trees_only': ['None'],
                        'fp_only_without_constant': ['first', 'second', 'third'],
                        'fp_only_with_constant': ['first', 'second', 'third'],
                        'fp_with_constant_then_mlp': ['first', 'second', 'third'],
                        'fp_with_constant_then_trees': ['first', 'second', 'third']}

    elif problem in ['linear_data', 'overlap_data', 'power_plant', 'concrete']:
        FP_TERMS_EXP = {'fp_known_then_mlp': ['first'],
                        'fp_known_then_trees': ['first'],
                        'fp_known_only': ['first'],
                        'mlp_only': ['None'],
                        'trees_only': ['None'],
                        'fp_only_without_constant': ['first'],
                        'fp_only_with_constant': ['first'],
                        'fp_with_constant_then_mlp': ['first'],
                        'fp_with_constant_then_trees': ['first']}

    if problem in ['friedman1', 'corr_friedman1']:
        best_config = {'mlp': {'fp_learning_rate': 0.005, 'mlp_learning_rate': 0.005, 'num_layers': 2, 'hidden_size': 15, 'standardize': True, 'out_size': 1},
                       'rf': {'fp_learning_rate': 0.005, 'N': 500, 'max_d': None, 'max_features': None, 'min_samples_split': 5, 'learning_rate_trees': None},
                       'boosting': {'fp_learning_rate': 0.005, 'N': 700, 'max_d': 2, 'learning_rate_trees': 0.3, 'max_features': None, 'min_samples_split': None}}
    elif problem in ['linear_data', 'overlap_data']:
        best_config = {'mlp': {'fp_learning_rate': 0.005, 'mlp_learning_rate': 0.005, 'num_layers': 2, 'hidden_size': 10, 'standardize': True, 'out_size': 1},
                       'rf': {'fp_learning_rate': 0.005, 'N': 500, 'max_d': None, 'max_features': None, 'min_samples_split': 5, 'learning_rate_trees': None},
                       'boosting': {'fp_learning_rate': 0.005, 'N': 400, 'max_d': 2, 'learning_rate_trees': 0.3, 'max_features': None, 'min_samples_split': None}}
    elif problem in ['power_plant', 'concrete']:
        best_config = {'mlp': {'fp_learning_rate': 0.005, 'mlp_learning_rate': 0.005, 'num_layers': 2, 'hidden_size': 30, 'standardize': True, 'out_size': 1},
                       'rf': {'fp_learning_rate': 0.005, 'N': 200, 'max_d': None, 'max_features': None, 'min_samples_split': 5, 'learning_rate_trees': None},
                       'boosting': {'fp_learning_rate': 0.005, 'N': 300, 'max_d': 5, 'learning_rate_trees': 0.3, 'max_features': None, 'min_samples_split': None}}


    jobs = []

    # Initialize wandb API
    api = wandb.Api()
    entity = 'yannclaes'

    FP_TERMS = FP_TERMS_EXP[experiment]
    for fp_term in FP_TERMS:

        if experiment == 'fp_known_only':

            trainer = Trainer(n_estimators=None,
                              depth=None,
                              fp_learning_rate=None,
                              mlp_learning_rate=None,
                              learning_rate_trees=None,
                              max_features=None,
                              min_samples_split=None,
                              num_layers=None,
                              hidden_size=None,
                              fp_inputs=args.fp_inputs,
                              fa_inputs=None,
                              out_size=None,
                              standardize=False,
                              incomplete=incomplete,
                              problem=problem,
                              problem_setting=problem_setting,
                              fp_term=fp_term,
                              boosting=False,
                              fp_warmup=False,
                              fp_control=False,
                              jointly=False,
                              model_name=experiment,
                              lambda_loss=False,
                              seed=None,
                              filter_fa=filter_fa,
                              extrapolation=extrapolation,
                              dataset_id=dataset_id)

            jobs.extend(trainer.train_model(wandb_logging=False))

        elif experiment == 'fp_known_then_trees':

            config = best_config[tree_model]
            config['fp_inputs'] = fp_inputs[problem][fp_term]

            if filter_fa:
                config['fa_inputs'] = fa_inputs[problem][fp_term]
            else:
                config['fa_inputs'] = fa_inputs[problem]['None']

            trainer = Trainer(n_estimators=config['N'],
                              depth=config['max_d'],
                              fp_learning_rate=None,
                              mlp_learning_rate=None,
                              learning_rate_trees=config['learning_rate_trees'],
                              max_features=config['max_features'],
                              min_samples_split=config['min_samples_split'],
                              num_layers=None,
                              hidden_size=None,
                              fp_inputs=config['fp_inputs'],
                              fa_inputs=config['fa_inputs'],
                              out_size=None,
                              standardize=True,
                              incomplete=incomplete,
                              problem=problem,
                              problem_setting=problem_setting,
                              fp_term=fp_term,
                              boosting=boosting,
                              fp_warmup=False,
                              fp_control=False,
                              jointly=False,
                              model_name=experiment,
                              lambda_loss=False,
                              seed=seed,
                              filter_fa=filter_fa,
                              extrapolation=extrapolation,
                              dataset_id=dataset_id)

            jobs.extend(trainer.train_model(wandb_logging=False))

        elif experiment == 'fp_known_then_mlp':

            config = best_config['mlp']
            config['fp_inputs'] = fp_inputs[problem][fp_term]

            if filter_fa:
                config['fa_inputs'] = fa_inputs[problem][fp_term]
            else:
                config['fa_inputs'] = fa_inputs[problem]['None']

            trainer = Trainer(n_estimators=None,
                              depth=None,
                              fp_learning_rate=None,
                              mlp_learning_rate=config['mlp_learning_rate'],
                              learning_rate_trees=None,
                              max_features=None,
                              min_samples_split=None,
                              num_layers=config['num_layers'],
                              hidden_size=config['hidden_size'],
                              fp_inputs=config['fp_inputs'],
                              fa_inputs=config['fa_inputs'],
                              out_size=config['out_size'],
                              standardize=True,
                              incomplete=incomplete,
                              problem=problem,
                              problem_setting=problem_setting,
                              fp_term=fp_term,
                              boosting=False,
                              fp_warmup=False,
                              fp_control=False,
                              jointly=False,
                              model_name=experiment,
                              lambda_loss=False,
                              seed=seed,
                              filter_fa=filter_fa,
                              extrapolation=extrapolation,
                              dataset_id=dataset_id)

            jobs.extend(trainer.train_model(wandb_logging=False))

        elif experiment == 'mlp_only':

            config = best_config['mlp']
            config['fp_inputs'] = fp_inputs[problem][fp_term]

            if filter_fa:
                config['fa_inputs'] = fa_inputs[problem][fp_term]
            else:
                config['fa_inputs'] = fa_inputs[problem]['None']

            trainer = Trainer(n_estimators=None,
                              depth=None,
                              fp_learning_rate=None,
                              mlp_learning_rate=config['mlp_learning_rate'],
                              learning_rate_trees=None,
                              max_features=None,
                              min_samples_split=None,
                              num_layers=config['num_layers'],
                              hidden_size=config['hidden_size'],
                              fp_inputs=None,
                              fa_inputs=config['fa_inputs'],
                              out_size=config['out_size'],
                              standardize=True,
                              incomplete=incomplete,
                              problem=problem,
                              problem_setting=problem_setting,
                              fp_term=fp_term,
                              boosting=False,
                              fp_warmup=False,
                              fp_control=False,
                              jointly=False,
                              model_name=experiment,
                              lambda_loss=False,
                              seed=seed,
                              filter_fa=filter_fa,
                              extrapolation=extrapolation,
                              dataset_id=dataset_id)

            jobs.extend(trainer.train_model(wandb_logging=True))

        elif experiment == 'trees_only':

            config = best_config[tree_model]
            config['fp_inputs'] = fp_inputs[problem][fp_term]

            if filter_fa:
                config['fa_inputs'] = fa_inputs[problem][fp_term]
            else:
                config['fa_inputs'] = fa_inputs[problem]['None']

            trainer = Trainer(n_estimators=config['N'],
                              depth=config['max_d'],
                              fp_learning_rate=None,
                              mlp_learning_rate=None,
                              learning_rate_trees=config['learning_rate_trees'],
                              max_features=config['max_features'],
                              min_samples_split=config['min_samples_split'],
                              num_layers=None,
                              hidden_size=None,
                              fp_inputs=None,
                              fa_inputs=config['fa_inputs'],
                              out_size=None,
                              standardize=True,
                              incomplete=incomplete,
                              problem=problem,
                              problem_setting=problem_setting,
                              fp_term=fp_term,
                              boosting=boosting,
                              fp_warmup=False,
                              fp_control=False,
                              jointly=False,
                              model_name=experiment,
                              lambda_loss=False,
                              seed=seed,
                              filter_fa=filter_fa,
                              extrapolation=extrapolation,
                              dataset_id=dataset_id)

            jobs.extend(trainer.train_model(wandb_logging=False))

        elif experiment == 'fp_only_without_constant':

            config = best_config['mlp']
            config['fp_inputs'] = fp_inputs[problem][fp_term]

            if filter_fa:
                config['fa_inputs'] = fa_inputs[problem][fp_term]
            else:
                config['fa_inputs'] = fa_inputs[problem]['None']

            trainer = Trainer(n_estimators=None,
                              depth=None,
                              fp_learning_rate=config['fp_learning_rate'],
                              mlp_learning_rate=None,
                              learning_rate_trees=None,
                              max_features=None,
                              min_samples_split=None,
                              num_layers=None,
                              hidden_size=None,
                              fp_inputs=config['fp_inputs'],
                              fa_inputs=None,
                              out_size=None,
                              standardize=True,
                              incomplete=incomplete,
                              problem=problem,
                              problem_setting=problem_setting,
                              fp_term=fp_term,
                              boosting=False,
                              fp_warmup=False,
                              fp_control=False,
                              jointly=False,
                              model_name=experiment,
                              lambda_loss=False,
                              seed=seed,
                              filter_fa=filter_fa,
                              extrapolation=extrapolation,
                              dataset_id=dataset_id)

            jobs.extend(trainer.train_model(wandb_logging=True))

        elif experiment == 'fp_only_with_constant':

            config = best_config['mlp']
            config['fp_inputs'] = fp_inputs[problem][fp_term]

            if filter_fa:
                config['fa_inputs'] = fa_inputs[problem][fp_term]
            else:
                config['fa_inputs'] = fa_inputs[problem]['None']

            trainer = Trainer(n_estimators=None,
                              depth=None,
                              fp_learning_rate=config['fp_learning_rate'],
                              mlp_learning_rate=None,
                              learning_rate_trees=None,
                              max_features=None,
                              min_samples_split=None,
                              num_layers=None,
                              hidden_size=None,
                              fp_inputs=config['fp_inputs'],
                              fa_inputs=None,
                              out_size=None,
                              standardize=True,
                              incomplete=incomplete,
                              problem=problem,
                              problem_setting=problem_setting,
                              fp_term=fp_term,
                              boosting=False,
                              fp_warmup=False,
                              fp_control=False,
                              jointly=False,
                              model_name=experiment,
                              lambda_loss=False,
                              seed=seed,
                              filter_fa=filter_fa,
                              extrapolation=extrapolation,
                              dataset_id=dataset_id)

            jobs.extend(trainer.train_model(wandb_logging=True))

        elif experiment == 'fp_with_constant_then_trees':

            config = best_config[tree_model]
            config['fp_inputs'] = fp_inputs[problem][fp_term]

            if filter_fa:
                config['fa_inputs'] = fa_inputs[problem][fp_term]
            else:
                config['fa_inputs'] = fa_inputs[problem]['None']

            trainer = Trainer(n_estimators=config['N'],
                              depth=config['max_d'],
                              fp_learning_rate=config['fp_learning_rate'],
                              mlp_learning_rate=None,
                              learning_rate_trees=config['learning_rate_trees'],
                              max_features=config['max_features'],
                              min_samples_split=config['min_samples_split'],
                              num_layers=None,
                              hidden_size=None,
                              fp_inputs=config['fp_inputs'],
                              fa_inputs=config['fa_inputs'],
                              out_size=None,
                              standardize=True,
                              incomplete=incomplete,
                              problem=problem,
                              problem_setting=problem_setting,
                              fp_term=fp_term,
                              boosting=boosting,
                              fp_warmup=False,
                              fp_control=False,
                              jointly=False,
                              model_name=experiment,
                              lambda_loss=False,
                              seed=seed,
                              filter_fa=filter_fa,
                              extrapolation=extrapolation,
                              dataset_id=dataset_id)

            jobs.extend(trainer.train_model(wandb_logging=True))

        elif experiment == 'fp_with_constant_then_mlp':

            config = best_config['mlp']
            config['fp_inputs'] = fp_inputs[problem][fp_term]

            if filter_fa:
                config['fa_inputs'] = fa_inputs[problem][fp_term]
            else:
                config['fa_inputs'] = fa_inputs[problem]['None']

            trainer = Trainer(n_estimators=None,
                              depth=None,
                              fp_learning_rate=config['fp_learning_rate'],
                              mlp_learning_rate=config['mlp_learning_rate'],
                              learning_rate_trees=None,
                              max_features=None,
                              min_samples_split=None,
                              num_layers=config['num_layers'],
                              hidden_size=config['hidden_size'],
                              fp_inputs=config['fp_inputs'],
                              fa_inputs=config['fa_inputs'],
                              out_size=config['out_size'],
                              standardize=True,
                              incomplete=incomplete,
                              problem=problem,
                              problem_setting=problem_setting,
                              fp_term=fp_term,
                              boosting=False,
                              fp_warmup=False,
                              fp_control=False,
                              jointly=False,
                              model_name=experiment,
                              lambda_loss=False,
                              seed=seed,
                              filter_fa=filter_fa,
                              extrapolation=extrapolation,
                              dataset_id=dataset_id)

            jobs.extend(trainer.train_model(wandb_logging=True))

    schedule(*jobs, backend='async')
    # schedule(*jobs,
    #          backend='slurm',
    #          env=['#SBATCH --export=ALL',
    #               '#SBATCH --partition=all',
    #               'eval "$(conda shell.bash hook)"',
    #               'conda activate phyML'])

