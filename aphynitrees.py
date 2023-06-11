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

    # Define default arguments to match wandb sweep configs
    parser = argparse.ArgumentParser()
    parser.add_argument('--tree_model', type=str, default='boosting')
    parser.add_argument('--problem', type=str, default='friedman1')
    parser.add_argument('--problem_setting', type=str, default='entire')
    parser.add_argument('--dataset_id', type=int, default=0)
    parser.add_argument('--fp_warmup', type=str, default='yes')
    parser.add_argument('--fp_control', type=str, default='no')
    parser.add_argument('--filter_fa', type=str, default='no')
    parser.add_argument('--extrapolation', type=str, default='no')
    args = parser.parse_args()

    fp_control = args.fp_control == 'yes'
    filter_fa = args.filter_fa == 'yes'
    fp_warmup = args.fp_warmup == 'yes'
    extrapolation = args.extrapolation == 'yes'
    boosting = args.tree_model == 'boosting'
    tree_model = args.tree_model
    incomplete = True
    l_loss = False
    problem = args.problem
    problem_setting = args.problem_setting
    dataset_id = args.dataset_id
    seed = dataset_id

    if problem in ['friedman1', 'corr_friedman1']:
        FP_TERMS = ['first', 'second', 'third']
    elif problem in ['linear_data', 'overlap_data', 'power_plant', 'concrete']:
        FP_TERMS = ['first']

    if problem in ['friedman1', 'corr_friedman1']:
        best_config = {'rf': {'fp_learning_rate': 0.005, 'N': 500, 'max_d': None, 'max_features': None, 'min_samples_split': 5, 'learning_rate_trees': None, 'fp_warmup': fp_warmup, 'fp_control': fp_control},
                       'boosting': {'fp_learning_rate': 0.005, 'N': 700, 'max_d': 2, 'learning_rate_trees': 0.3, 'max_features': None, 'min_samples_split': None, 'fp_warmup': fp_warmup, 'fp_control': fp_control}}
    elif problem in ['linear_data', 'overlap_data']:
        best_config = {'rf': {'fp_learning_rate': 0.005, 'N': 500, 'max_d': None, 'max_features': None, 'min_samples_split': 5, 'learning_rate_trees': None, 'fp_warmup': fp_warmup, 'fp_control': fp_control},
                       'boosting': {'fp_learning_rate': 0.005, 'N': 400, 'max_d': 2, 'learning_rate_trees': 0.3, 'max_features': None, 'min_samples_split': None, 'fp_warmup': fp_warmup, 'fp_control': fp_control}}
    elif problem in ['power_plant', 'concrete']:
        best_config = {'rf': {'fp_learning_rate': 0.005, 'N': 200, 'max_d': None, 'max_features': None, 'min_samples_split': 5, 'learning_rate_trees': None, 'fp_warmup': fp_warmup, 'fp_control': fp_control},
                       'boosting': {'fp_learning_rate': 0.005, 'N': 300, 'max_d': 5, 'learning_rate_trees': 0.3, 'max_features': None, 'min_samples_split': None, 'fp_warmup': fp_warmup, 'fp_control': fp_control}}

    jobs = []

    # Initialize wandb API
    api = wandb.Api()
    entity = 'yannclaes'
    project = 'aphynitrees'

    for fp_term in FP_TERMS:

        config = best_config[tree_model]
        config['fp_inputs'] = fp_inputs[problem][fp_term]

        if filter_fa:
            config['fa_inputs'] = fa_inputs[problem][fp_term]
        else:
            config['fa_inputs'] = fa_inputs[problem]['None']

        trainer = Trainer(n_estimators=config['N'],
                          depth=config['max_d'],
                          learning_rate_trees=config['learning_rate_trees'],
                          max_features=config['max_features'],
                          min_samples_split=config['min_samples_split'],
                          fp_learning_rate=config['fp_learning_rate'],
                          mlp_learning_rate=None,
                          lambda_loss=l_loss,
                          num_layers=None,
                          hidden_size=None,
                          incomplete=incomplete,
                          problem=problem,
                          problem_setting=problem_setting,
                          fp_term=fp_term,
                          boosting=boosting,
                          fp_inputs=config['fp_inputs'],
                          fa_inputs=config['fa_inputs'],
                          out_size=None,
                          standardize=True,
                          fp_warmup=config['fp_warmup'],
                          fp_control=config['fp_control'],
                          jointly=False,
                          model_name='aphynitrees',
                          seed=seed,
                          filter_fa=filter_fa,
                          extrapolation=extrapolation,
                          dataset_id=dataset_id)

        jobs.extend(trainer.train_model(wandb_logging=True))

    schedule(*jobs, backend='async')
    # schedule(*jobs,
    #          backend='slurm',
    #          env=['#SBATCH --export=ALL',
                  # '#SBATCH --partition=all',
    #               'eval "$(conda shell.bash hook)"',
    #               'conda activate phyML'])
