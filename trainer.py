import os
import dill
import torch
import numpy as np

from dawgz import job, context
from aphynity_model import APHYNITY
from tree_models import APHYNITrees
from baseline_models import Baseline, make_settings, make_hyperparameter


class Trainer():

    def __init__(self, n_estimators, depth, learning_rate_trees, fp_learning_rate,
        mlp_learning_rate, max_features, min_samples_split, lambda_loss, num_layers, hidden_size,
        incomplete, problem, problem_setting, fp_term, boosting, standardize,
        fp_warmup, fp_control, jointly, model_name, fp_inputs, fa_inputs, out_size, seed,
        filter_fa, extrapolation, dataset_id):

        # Tree models hyperparameters
        self.d = depth
        self.N = n_estimators
        self.learning_rate_trees = learning_rate_trees
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.train_aug = 2
        self.boosting = boosting

        # Parametric model hyperparameters
        self.fp_learning_rate = fp_learning_rate
        self.mlp_learning_rate = mlp_learning_rate
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.b_size = 25
        self.fp_inputs = fp_inputs
        self.fa_inputs = fa_inputs
        self.out_size = out_size

        self.lambda_loss = lambda_loss
        self.lambda_rate = 1.
        self.incomplete = incomplete
        self.fp_warmup = fp_warmup
        self.fp_control = fp_control
        self.jointly = jointly

        self.problem = problem
        self.problem_setting = problem_setting
        self.fp_term = fp_term
        self.dataset_id = dataset_id
        self.train_size = 0.5
        self.filter_fa = filter_fa
        self.extrapolation = extrapolation
        self.standardize = standardize

        self.seed = seed
        self.model_name = model_name
        self.jobs = []

        if self.extrapolation:
            self.data_path = f'data_extrapolation/{self.problem}'
        else:
            if self.problem in ['friedman1', 'corr_friedman1']:
                self.data_path = f'data/{self.problem}/{self.problem_setting}'
            else:
                self.data_path = f'data/{self.problem}'

        if self.model_name == 'aphynity':
            self.hyperparameter = (self.incomplete,
                                   self.fp_learning_rate,
                                   self.lambda_rate,
                                   self.num_layers,
                                   self.hidden_size,
                                   self.b_size,
                                   self.mlp_learning_rate)
        elif self.model_name == 'aphynitrees':
            self.hyperparameter = (self.N,
                                   self.d,
                                   self.incomplete,
                                   self.train_aug,
                                   self.fp_learning_rate,
                                   self.learning_rate_trees,
                                   self.max_features,
                                   self.min_samples_split,
                                   self.lambda_rate)
        else:
            self.hyperparameter = make_hyperparameter(self.model_name,
                                                      self.incomplete,
                                                      self.N,
                                                      self.d,
                                                      self.boosting,
                                                      self.fp_learning_rate,
                                                      self.learning_rate_trees,
                                                      self.max_features,
                                                      self.min_samples_split,
                                                      self.num_layers,
                                                      self.hidden_size,
                                                      self.b_size,
                                                      self.mlp_learning_rate)

    def train_model(self, wandb_logging=False):

        # Dataset path
        train_path = os.path.join(self.data_path, str(self.dataset_id))
        train_X_path = os.path.join(train_path, 'learning_set_X.pt')
        train_y_path = os.path.join(train_path, 'learning_set_y.pt')

        # Train hyperparameter combination
        job_name = f'{self.dataset_id}_TR_{self.model_name}_{self.problem}'

        @context(**locals())
        @job(name=job_name, cpus=1, memory='8GB', timelimit='1-00:00:00')
        def train():

            if self.extrapolation:
                learning_set_X = torch.load(train_X_path)[:200]
                learning_set_y = torch.load(train_y_path)[:200]
            else:
                learning_set_X = torch.load(train_X_path)[:200]
                learning_set_y = torch.load(train_y_path)[:200]
            val_set_end = int(self.train_size * len(learning_set_X))

            train_X = learning_set_X[val_set_end:]
            train_y = learning_set_y[val_set_end:]
            valid_X = learning_set_X[:val_set_end]
            valid_y = learning_set_y[:val_set_end]

            if self.standardize:
                train_X_mean = torch.mean(train_X, dim=0)
                train_X_std = torch.std(train_X, dim=0)
                train_y_mean = torch.mean(train_y, dim=0)
                train_y_std = torch.std(train_y, dim=0)

                if self.problem in ['concrete', 'power_plant']:
                    train_X = (train_X - train_X_mean) / train_X_std
                    train_y = (train_y - train_y_mean) / train_y_std
                    valid_X = (valid_X - train_X_mean) / train_X_std
                    valid_y = (valid_y - train_y_mean) / train_y_std

                    train_X_mean = None
                    train_X_std = None
                    train_y_mean = None
                    train_y_std = None
            else:
                train_X_mean = None
                train_X_std = None
                train_y_mean = None
                train_y_std = None

            # Initialize and train the corresponding model
            if self.model_name == 'aphynity':
                model = APHYNITY(self.hyperparameter,
                                 self.lambda_loss,
                                 self.problem,
                                 self.problem_setting,
                                 self.fp_term,
                                 self.dataset_id,
                                 fp_inputs=self.fp_inputs,
                                 fa_inputs=self.fa_inputs,
                                 out_size=self.out_size,
                                 saving=True,
                                 training=True,
                                 fp_warmup=self.fp_warmup,
                                 fp_control=self.fp_control,
                                 jointly=self.jointly,
                                 seed=self.seed,
                                 train_mean_X=train_X_mean,
                                 train_std_X=train_X_std,
                                 train_mean_y=train_y_mean,
                                 train_std_y=train_y_std,
                                 extrapolation=self.extrapolation,
                                 filter_fa=self.filter_fa)
                logging = wandb_logging
            elif self.model_name == 'aphynitrees':
                model = APHYNITrees(self.hyperparameter,
                                    self.lambda_loss,
                                    self.problem,
                                    self.problem_setting,
                                    self.fp_term,
                                    self.boosting,
                                    self.dataset_id,
                                    saving=True,
                                    training=True,
                                    fp_warmup=self.fp_warmup,
                                    fp_control=self.fp_control,
                                    fp_inputs=self.fp_inputs,
                                    fa_inputs=self.fa_inputs,
                                    seed=self.seed,
                                    extrapolation=self.extrapolation,
                                    filter_fa=self.filter_fa)
                logging = wandb_logging
            else:
                if self.problem in ['friedman1', 'corr_friedman1', 'linear_data', 'overlap_data']:
                    with open(os.path.join(train_path, 'params.pkl'), 'rb') as f:
                        real_params = dill.load(f)

                    for param in real_params:
                        real_params[param] = torch.tensor([real_params[param]])
                else:
                    real_params = None

                settings, logging = make_settings(self.problem,
                                                  self.problem_setting,
                                                  self.fp_term,
                                                  self.dataset_id,
                                                  self.model_name,
                                                  self.hyperparameter,
                                                  real_params,
                                                  wandb_logging,
                                                  fp_inputs=self.fp_inputs,
                                                  fa_inputs=self.fa_inputs,
                                                  out_size=self.out_size,
                                                  saving=True,
                                                  train_mean_X=None,
                                                  train_std_X=None,
                                                  train_mean_y=None,
                                                  train_std_y=None,
                                                  seed=self.seed,
                                                  extrapolation=self.extrapolation,
                                                  filter_fa=self.filter_fa)
                model = Baseline(**settings[self.model_name])
                logging = wandb_logging

            best_loss = model.train(train_X, train_y, valid_X, valid_y, logging)

        self.jobs.append(train)
        return self.jobs

    def test_model(self, rel=True):

        # Dataset path
        train_path = os.path.join(self.data_path, str(self.dataset_id))
        train_X_path = os.path.join(train_path, 'learning_set_X.pt')
        train_y_path = os.path.join(train_path, 'learning_set_y.pt')

        if self.extrapolation:
            learning_set_X = torch.load(train_X_path)[:200]
            learning_set_y = torch.load(train_y_path)[:200]
        else:
            learning_set_X = torch.load(train_X_path)[:200]
            learning_set_y = torch.load(train_y_path)[:200]
        val_set_end = int(self.train_size * len(learning_set_X))

        train_X = learning_set_X[val_set_end:]
        train_y = learning_set_y[val_set_end:]
        valid_X = learning_set_X[:val_set_end]
        valid_y = learning_set_y[:val_set_end]

        test_set_X = torch.load(os.path.join(train_path, 'test_set_X.pt'))
        test_set_y = torch.load(os.path.join(train_path, 'test_set_y.pt'))

        if self.standardize:
            train_X_mean = torch.mean(train_X, dim=0)
            train_X_std = torch.std(train_X, dim=0)
            train_y_mean = torch.mean(train_y, dim=0)
            train_y_std = torch.std(train_y, dim=0)

            if self.problem in ['concrete', 'power_plant']:
                train_X = (train_X - train_X_mean) / train_X_std
                train_y = (train_y - train_y_mean) / train_y_std
                valid_X = (valid_X - train_X_mean) / train_X_std
                valid_y = (valid_y - train_y_mean) / train_y_std
                test_set_X = (test_set_X - train_X_mean) / train_X_std
                test_set_y = (test_set_y - train_y_mean) / train_y_std

                train_X_mean = None
                train_X_std = None
                train_y_mean = None
                train_y_std = None
        else:
            train_X_mean = None
            train_X_std = None
            train_y_mean = None
            train_y_std = None

        # Load simulation params
        if self.problem in ['friedman1', 'corr_friedman1', 'linear_data', 'overlap_data']:
            with open(os.path.join(self.data_path, str(self.dataset_id), 'params.pkl'), 'rb') as f:
                params = dill.load(f)

            for param in params:
                params[param] = torch.tensor([params[param]])
        else:
            params = None

        # Initialize the corresponding model
        if self.model_name == 'aphynity':
            model = APHYNITY(self.hyperparameter,
                             self.lambda_loss,
                             self.problem,
                             self.problem_setting,
                             self.fp_term,
                             self.dataset_id,
                             fp_inputs=self.fp_inputs,
                             fa_inputs=self.fa_inputs,
                             out_size=self.out_size,
                             saving=True,
                             training=True,
                             fp_warmup=self.fp_warmup,
                             fp_control=self.fp_control,
                             jointly=self.jointly,
                             seed=self.seed,
                             train_mean_X=train_X_mean,
                             train_std_X=train_X_std,
                             train_mean_y=train_y_mean,
                             train_std_y=train_y_std,
                             extrapolation=self.extrapolation,
                             filter_fa=self.filter_fa)
            model.load(model.model_path)

        elif self.model_name == 'aphynitrees':
            model = APHYNITrees(self.hyperparameter,
                                self.lambda_loss,
                                self.problem,
                                self.problem_setting,
                                self.fp_term,
                                self.boosting,
                                self.dataset_id,
                                fp_inputs=self.fp_inputs,
                                fa_inputs=self.fa_inputs,
                                saving=True,
                                training=True,
                                fp_warmup=self.fp_warmup,
                                fp_control=self.fp_control,
                                seed=self.seed,
                                extrapolation=self.extrapolation,
                                filter_fa=self.filter_fa)
            model.load(model.model_phy_path, model.model_aug_path)

        else:
            settings, logging = make_settings(self.problem,
                                              self.problem_setting,
                                              self.fp_term,
                                              self.dataset_id,
                                              self.model_name,
                                              self.hyperparameter,
                                              params,
                                              False,
                                              fp_inputs=self.fp_inputs,
                                              fa_inputs=self.fa_inputs,
                                              out_size=self.out_size,
                                              saving=True,
                                              train_mean_X=train_X_mean,
                                              train_std_X=train_X_std,
                                              train_mean_y=train_y_mean,
                                              train_std_y=train_y_std,
                                              seed=self.seed,
                                              extrapolation=self.extrapolation,
                                              filter_fa=self.filter_fa)
            model = Baseline(**settings[self.model_name])
            model.load(model.model_path)

        test_mse = model.validate(test_set_X, test_set_y)

        # Compute error on parameters
        params_errors = model.compute_param_error(params, rel=rel)

        test_log_mse = np.log10(test_mse)
        return test_mse, test_log_mse, params_errors
