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
    parser.add_argument('--problem', type=str, default='friedman1')
    parser.add_argument('--problem_setting', type=str, default='entire')
    parser.add_argument('--dataset_id', type=int, default=0)
    parser.add_argument('--fp_warmup', type=str, default='yes')
    parser.add_argument('--fp_control', type=str, default='no')
    parser.add_argument('--jointly', type=str, default='no')
    parser.add_argument('--filter_fa', type=str, default='no')
    parser.add_argument('--extrapolation', type=str, default='no')
    args = parser.parse_args()

    fp_control = args.fp_control == 'yes'
    filter_fa = args.filter_fa == 'yes'
    extrapolation = args.extrapolation == 'yes'
    fp_warmup = args.fp_warmup == 'yes'
    jointly = args.jointly == 'yes'
    l_loss = False
    incomplete = True
    problem = args.problem
    problem_setting = args.problem_setting
    dataset_id = args.dataset_id
    seed = dataset_id

    if problem in ['friedman1', 'corr_friedman1']:
        FP_TERMS = ['first', 'second', 'third']
    elif problem in ['linear_data', 'overlap_data', 'power_plant', 'concrete']:
        FP_TERMS = ['first']

    if problem in ['friedman1', 'corr_friedman1']:
        best_config = {'mlp': {'fp_learning_rate': 0.005, 'mlp_learning_rate': 0.005, 'num_layers': 2, 'hidden_size': 15, 'standardize': True, 'out_size': 1, 'fp_warmup': fp_warmup, 'fp_control': fp_control, 'jointly': jointly}}
    elif problem in ['linear_data', 'overlap_data']:
        best_config = {'mlp': {'fp_learning_rate': 0.005, 'mlp_learning_rate': 0.005, 'num_layers': 2, 'hidden_size': 10, 'standardize': True, 'out_size': 1, 'fp_warmup': fp_warmup, 'fp_control': fp_control, 'jointly': jointly}}
    elif problem in ['power_plant', 'concrete']:
        best_config = {'mlp': {'fp_learning_rate': 0.005, 'mlp_learning_rate': 0.005, 'num_layers': 2, 'hidden_size': 30, 'standardize': True, 'out_size': 1, 'fp_warmup': fp_warmup, 'fp_control': fp_control, 'jointly': jointly}}

    jobs = []

    # Initialize wandb API
    api = wandb.Api()
    entity = 'yannclaes'
    project = 'aphynity'

    for fp_term in FP_TERMS:

        config = best_config['mlp']
        config['fp_inputs'] = fp_inputs[problem][fp_term]

        if filter_fa:
            config['fa_inputs'] = fa_inputs[problem][fp_term]
        else:
            config['fa_inputs'] = fa_inputs[problem]['None']

        trainer = Trainer(n_estimators=None,
                          depth=None,
                          learning_rate_trees=None,
                          fp_learning_rate=config['fp_learning_rate'],
                          mlp_learning_rate=config['mlp_learning_rate'],
                          max_features=None,
                          min_samples_split=None,
                          lambda_loss=l_loss,
                          num_layers=config['num_layers'],
                          hidden_size=config['hidden_size'],
                          incomplete=incomplete,
                          problem=problem,
                          problem_setting=problem_setting,
                          fp_term=fp_term,
                          boosting=False,
                          fp_warmup=config['fp_warmup'],
                          fp_control=config['fp_control'],
                          jointly=config['jointly'],
                          fp_inputs=config['fp_inputs'],
                          fa_inputs=config['fa_inputs'],
                          out_size=config['out_size'],
                          standardize=True,
                          model_name='aphynity',
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

