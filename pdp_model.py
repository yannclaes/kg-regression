import os
import copy
import wandb
import settings
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from models import MLP
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from utils import make_known_model, save_model, load_model, get_batch


class Fp_Model():

    def __init__(self, problem, term, dataset_id, fp_inputs, learning_rate,
        b_size, saving, fp_with_constant=False, pdp_method='mlp', pdp_config=None,
        loss='global_loss', n_iters=3000, extrapolation=False, seed=None):

        self.problem = problem
        self.term = term
        self.incomplete = True
        self.dataset_id = dataset_id
        self.fp_inputs = fp_inputs
        self.learning_rate = learning_rate
        self.b_size = b_size
        self.criterion = nn.MSELoss()
        self.fp_with_constant = fp_with_constant
        self.saving = saving
        self.pdp_method = pdp_method
        self.pdp_config = pdp_config
        self.loss = loss
        self.n_iters = n_iters
        self.extrapolation = extrapolation
        self.seed = seed

        self.model = make_known_model(self.problem,
                                      self.term,
                                      self.incomplete,
                                      constant=self.fp_with_constant,
                                      real_params=None,
                                      seed=self.seed)

        # Initialize model and results paths
        MOD_OUTPUT_PREFIX = self.problem + '/' + self.term + '/models'
        RES_OUTPUT_PREFIX = self.problem + '/' + self.term + '/results'

        if self.pdp_method == 'mlp':
            self.model_path = os.path.join(MOD_OUTPUT_PREFIX,
                                           'fp_with_constant' if self.fp_with_constant else 'fp_without_constant',
                                           'extrapolation' if self.extrapolation else 'interpolation',
                                           self.loss,
                                           str(self.learning_rate),
                                           self.pdp_method,
                                           str(self.pdp_config['num_layers']),
                                           str(self.pdp_config['hidden_size']),
                                           str(self.pdp_config['mlp_learning_rate']),
                                           'exp_' + str(self.dataset_id))
            self.res_path = os.path.join(RES_OUTPUT_PREFIX,
                                         'fp_with_constant' if self.fp_with_constant else 'fp_without_constant',
                                         'extrapolation' if self.extrapolation else 'interpolation',
                                         self.loss,
                                         str(self.learning_rate),
                                         self.pdp_method,
                                         str(self.pdp_config['num_layers']),
                                         str(self.pdp_config['hidden_size']),
                                         str(self.pdp_config['mlp_learning_rate']),
                                         'exp_' + str(self.dataset_id))

        elif self.pdp_method == 'rf':
            self.model_path = os.path.join(MOD_OUTPUT_PREFIX,
                                           'fp_with_constant' if self.fp_with_constant else 'fp_without_constant',
                                           'extrapolation' if self.extrapolation else 'interpolation',
                                           self.loss,
                                           str(self.learning_rate),
                                           self.pdp_method,
                                           str(self.pdp_config['N']),
                                           str(self.pdp_config['max_features']),
                                           str(self.pdp_config['min_samples_split']),
                                           'exp_' + str(self.dataset_id))
            self.res_path = os.path.join(RES_OUTPUT_PREFIX,
                                         'fp_with_constant' if self.fp_with_constant else 'fp_without_constant',
                                         'extrapolation' if self.extrapolation else 'interpolation',
                                         self.loss,
                                         str(self.learning_rate),
                                         self.pdp_method,
                                         str(self.pdp_config['N']),
                                         str(self.pdp_config['max_features']),
                                         str(self.pdp_config['min_samples_split']),
                                         'exp_' + str(self.dataset_id))

        elif self.pdp_method == 'boosting':
            self.model_path = os.path.join(MOD_OUTPUT_PREFIX,
                                           'fp_with_constant' if self.fp_with_constant else 'fp_without_constant',
                                           'extrapolation' if self.extrapolation else 'interpolation',
                                           self.loss,
                                           str(self.learning_rate),
                                           self.pdp_method,
                                           str(self.pdp_config['N']),
                                           str(self.pdp_config['max_d']),
                                           str(self.pdp_config['learning_rate_trees']),
                                           'exp_' + str(self.dataset_id))
            self.res_path = os.path.join(RES_OUTPUT_PREFIX,
                                         'fp_with_constant' if self.fp_with_constant else 'fp_without_constant',
                                         'extrapolation' if self.extrapolation else 'interpolation',
                                         self.loss,
                                         str(self.learning_rate),
                                         self.pdp_method,
                                         str(self.pdp_config['N']),
                                         str(self.pdp_config['max_d']),
                                         str(self.pdp_config['learning_rate_trees']),
                                         'exp_' + str(self.dataset_id))

        if self.saving:
            os.makedirs(self.model_path, exist_ok=True)
            os.makedirs(self.res_path, exist_ok=True)
            self.save_path = os.path.join(self.model_path, 'known.pt')

        self.optim = optim.Adam(self.model.parameters(),
                                lr=self.learning_rate)
        self.scheduler = CosineAnnealingLR(self.optim, T_max=self.n_iters)

    def compute_param_error(self, params, rel=True):
        errors = []
        if self.problem in ['friedman1', 'corr_friedman1']:
            if self.incomplete:
                if self.term == 'first':
                    if rel:
                        errors.append((abs(params['p0'] - self.model.p0.item()) / params['p0']).item())
                        errors.append((abs(params['p1'] - self.model.p1.item()) / params['p1']).item())
                    else:
                        errors.append(((params['p0'] - self.model.p0.item()) ** 2).item())
                        errors.append(((params['p1'] - self.model.p1.item()) ** 2).item())
                elif self.term == 'second':
                    if rel:
                        errors.append((abs(params['p2'] - self.model.p2.item()) / params['p2']).item())
                        errors.append((abs(params['p3'] - self.model.p3.item()) / params['p3']).item())
                    else:
                        errors.append(((params['p2'] - self.model.p2.item()) ** 2).item())
                        errors.append(((params['p3'] - self.model.p3.item()) ** 2).item())
                elif self.term == 'third':
                    if rel:
                        errors.append((abs(params['p4'] - self.model.p4.item()) / params['p4']).item())
                        errors.append((abs(params['p5'] - self.model.p5.item()) / params['p5']).item())
                    else:
                        errors.append(((params['p4'] - self.model.p4.item()) ** 2).item())
                        errors.append(((params['p5'] - self.model.p5.item()) ** 2).item())
            else:
                if rel:
                    errors.append((abs(params['p0'] - self.model.p0.item()) / params['p0']).item())
                    errors.append((abs(params['p1'] - self.model.p1.item()) / params['p1']).item())
                    errors.append((abs(params['p2'] - self.model.p2.item()) / params['p2']).item())
                    errors.append((abs(params['p3'] - self.model.p3.item()) / params['p3']).item())
                    errors.append((abs(params['p4'] - self.model.p4.item()) / params['p4']).item())
                    errors.append((abs(params['p5'] - self.model.p5.item()) / params['p5']).item())
                else:
                    errors.append(((params['p0'] - self.model.p0.item()) ** 2).item())
                    errors.append(((params['p1'] - self.model.p1.item()) ** 2).item())
                    errors.append(((params['p2'] - self.model.p2.item()) ** 2).item())
                    errors.append(((params['p3'] - self.model.p3.item()) ** 2).item())
                    errors.append(((params['p4'] - self.model.p4.item()) ** 2).item())
                    errors.append(((params['p5'] - self.model.p5.item()) ** 2).item())

        elif self.problem in ['linear_data', 'overlap_data']:
            if rel:
                errors.append((abs(params['beta'] - self.model.beta.item()) / abs(params['beta'])).item())
            else:
                errors.append(((params['beta'] - self.model.beta.item()) ** 2).item())

        return errors

    def load(self):
        self.model.load_state_dict(torch.load(self.save_path))

    def compute_fp_distance(self, train_x, train_y):
        with torch.no_grad():
            fp_pred = self.model(train_x).squeeze()

        distance = nn.MSELoss()(fp_pred, train_y).item()
        return distance

    def forward_epoch(self, X, y, train=False):

        n_batches = int(np.ceil(len(X) / self.b_size))
        permutation = torch.randperm(len(X))
        total_loss = 0.

        for index in range(0, len(X), self.b_size):

            batch_x, batch_y = get_batch(X,
                                         y,
                                         permutation[index:index+self.b_size])

            pred_x = self.model(batch_x)

            loss = self.criterion(batch_y, pred_x)
            if train:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                if self.problem in ['friedman1', 'corr_friedman1']:
                    w = self.model.p3.data
                    w = w.clamp(0, 1)
                    self.model.p3.data = w

            total_loss += loss.item() / n_batches
        return total_loss

    def train(self, train_x, train_y, valid_x, valid_y, wandb_logging):

        best_loss = np.inf

        for n in range(self.n_iters):

            # Training phase
            tr_loss = self.forward_epoch(train_x,
                                         train_y,
                                         train=True)

            self.scheduler.step()

            with torch.no_grad():

                # Loss of trained model on whole training set
                tr_loss = self.forward_epoch(train_x, train_y)

                # Validation phase
                va_loss = self.forward_epoch(valid_x, valid_y)

                # Compute distance(fp, f)
                distance = self.compute_fp_distance(train_x, train_y)

                # Save model
                if va_loss < best_loss:
                    if self.saving:
                        best_weights = copy.deepcopy(self.model.state_dict())
                    best_loss = va_loss

                    if wandb_logging:
                        wandb.run.summary['best pdp loss (curr. repeat)'] = va_loss
                        wandb.run.summary['best pdp epoch (curr. repeat)'] = n+1

                if wandb_logging:
                    log_dict = {f"fp/train_loss": tr_loss,
                                f"fp/valid_loss": va_loss,
                                f"fp/learning rate": self.optim.param_groups[0]['lr'] if self.optim is not None else None,
                                f"fp/d(fp, f)": distance,
                                f"fp/epoch": settings.fp_epoch+1}

                    if self.problem in ['friedman1', 'corr_friedman1', 'linear_data', 'overlap_data']:
                        params, params_name = [], []
                        if self.problem in ['friedman1', 'corr_friedman1']:
                            if self.incomplete:
                                if self.term == 'first':
                                    params_name = [f'fp/p0', f'fp/p1', f'fp/C']
                                    params.append(self.model.p0.item())
                                    params.append(self.model.p1.item())
                                    params.append(self.model.c.item())
                                elif self.term == 'second':
                                    params_name = [f'fp/p2', f'fp/p3', f'fp/C']
                                    params.append(self.model.p2.item())
                                    params.append(self.model.p3.item())
                                    params.append(self.model.c.item())
                                elif self.term == 'third':
                                    params_name = [f'fp/p4', f'fp/p5', f'fp/C']
                                    params.append(self.model.p4.item())
                                    params.append(self.model.p5.item())
                                    params.append(self.model.c.item())
                            else:
                                params_name = [f'fp/p0', f'fp/p1', f'fp/p2', f'fp/p3', f'fp/p4', f'fp/p5', f'fp/C']
                                params.append(self.model.p0.item())
                                params.append(self.model.p1.item())
                                params.append(self.model.p2.item())
                                params.append(self.model.p3.item())
                                params.append(self.model.p4.item())
                                params.append(self.model.p5.item())
                                params.append(self.model.c.item())

                        elif self.problem in ['linear_data', 'overlap_data']:
                            params.append(self.model.beta.item())
                            params_name = [f'fp/beta']

                        param_dict = {}
                        for param, name in zip(params, params_name):
                            param_dict[name] = param
                        log_dict.update(param_dict)

                    wandb.log(log_dict)

            settings.fp_epoch += 1
            print('Finished epoch {:04d}/{:04d} | Train loss : {:.6f} | Validation loss: {:.6f}'.format(n+1, self.n_iters, tr_loss, va_loss))

        return best_loss, best_weights, distance

    def validate(self, valid_x, valid_y):
        va_loss = self.forward_epoch(valid_x, valid_y)
        return va_loss

    def predict(self, test_x):
        with torch.no_grad():
            pred_x = self.model(test_x)
        return pred_x

class Fa_Model():

    def __init__(self, problem, term, N, d, boosting, learning_rate_trees,
        max_features, min_samples_split, fa_inputs, out_size, learning_rate_mlp,
        num_layers, hidden_size, b_size, saving, fa_mlp, pdp_method, dataset_id,
        train_mean_X=None, train_std_X=None, train_mean_y=None, train_std_y=None,
        seed=None, filter_fa=False, extrapolation=False, n_iters=3000, verbose=False):

        self.fa_mlp = fa_mlp
        self.problem = problem
        self.term = term
        self.fa_inputs = fa_inputs
        self.criterion = nn.MSELoss()
        self.saving = saving
        self.seed = seed
        self.dataset_id = dataset_id
        self.n_iters = n_iters
        self.filter_fa = filter_fa
        self.extrapolation = extrapolation
        self.pdp_method = pdp_method
        self.verbose = verbose

        # Tree hyperparameters
        self.N = N
        self.d = d
        self.boosting = boosting
        self.learning_rate_trees = learning_rate_trees
        self.max_features = max_features
        self.min_samples_split = min_samples_split

        # MLP hyperparameters
        self.learning_rate_mlp = learning_rate_mlp
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.b_size = b_size
        self.out_size = out_size

        self.train_mean_X = train_mean_X
        self.train_std_X = train_std_X
        self.train_mean_y = train_mean_y
        self.train_std_y = train_std_y

        # Initialize model and results paths
        MOD_OUTPUT_PREFIX = self.problem + '/' + self.term + '/models'
        RES_OUTPUT_PREFIX = self.problem + '/' + self.term + '/results'

        if self.fa_mlp:
            self.model_path = os.path.join(MOD_OUTPUT_PREFIX,
                                           self.pdp_method,
                                           'mlp',
                                           'filter' if self.filter_fa else 'full',
                                           'extrapolation' if self.extrapolation else 'interpolation',
                                           str(self.learning_rate_mlp),
                                           str(self.num_layers),
                                           str(self.hidden_size),
                                           'exp_' + str(self.dataset_id))
            self.res_path = os.path.join(RES_OUTPUT_PREFIX,
                                         self.pdp_method,
                                         'mlp',
                                         'filter' if self.filter_fa else 'full',
                                         'extrapolation' if self.extrapolation else 'interpolation',
                                         str(self.learning_rate_mlp),
                                         str(self.num_layers),
                                         str(self.hidden_size),
                                         'exp_' + str(self.dataset_id))
        else:
            if self.boosting:
                self.model_path = os.path.join(MOD_OUTPUT_PREFIX,
                                               self.pdp_method,
                                               'boosting',
                                               'filter' if self.filter_fa else 'full',
                                               'extrapolation' if self.extrapolation else 'interpolation',
                                               str(self.N),
                                               str(self.d),
                                               str(self.learning_rate_trees),
                                               'exp_' + str(self.dataset_id))
                self.res_path = os.path.join(RES_OUTPUT_PREFIX,
                                             self.pdp_method,
                                             'boosting',
                                             'filter' if self.filter_fa else 'full',
                                             'extrapolation' if self.extrapolation else 'interpolation',
                                             str(self.N),
                                             str(self.d),
                                             str(self.learning_rate_trees),
                                             'exp_' + str(self.dataset_id))
            else:
                self.model_path = os.path.join(MOD_OUTPUT_PREFIX,
                                               self.pdp_method,
                                               'rf',
                                               'filter' if self.filter_fa else 'full',
                                               'extrapolation' if self.extrapolation else 'interpolation',
                                               str(self.N),
                                               str(self.max_features),
                                               str(self.min_samples_split),
                                               'exp_' + str(self.dataset_id))
                self.res_path = os.path.join(RES_OUTPUT_PREFIX,
                                             self.pdp_method,
                                             'rf',
                                             'filter' if self.filter_fa else 'full',
                                             'extrapolation' if self.extrapolation else 'interpolation',
                                             str(self.N),
                                             str(self.max_features),
                                             str(self.min_samples_split),
                                             'exp_' + str(self.dataset_id))


        if self.saving:
            os.makedirs(self.model_path, exist_ok=True)
            os.makedirs(self.res_path, exist_ok=True)

        # Initialize models
        if self.fa_mlp:
            torch.manual_seed(self.seed)

            self.model = MLP(in_size=len(self.fa_inputs),
                             num_layers=self.num_layers,
                             hidden_size=self.hidden_size,
                             out_size=self.out_size,
                             train_mean_X=self.train_mean_X[self.fa_inputs] if self.train_mean_X is not None else None,
                             train_std_X=self.train_std_X[self.fa_inputs] if self.train_std_X is not None else None,
                             train_mean_y=self.train_mean_y,
                             train_std_y=self.train_std_y)

            self.optim = optim.Adam(self.model.parameters(),
                                    lr=self.learning_rate_mlp)
            self.scheduler = CosineAnnealingLR(self.optim, T_max=self.n_iters)

        else:
            if self.boosting:
                self.model = XGBRegressor(n_estimators=self.N,
                                          max_depth=self.d,
                                          learning_rate=self.learning_rate_trees,
                                          n_jobs=-1,
                                          random_state=self.seed)
            else:
                self.model = RandomForestRegressor(n_estimators=self.N,
                                                   max_features=self.max_features,
                                                   min_samples_split=self.min_samples_split,
                                                   random_state=self.seed,
                                                   n_jobs=-1)

    def load(self):
        if self.fa_mlp:
            self.model.load_state_dict(torch.load(os.path.join(self.model_path, 'augmented.pt')))
        else:
            self.model = load_model(os.path.join(self.model_path, 'augmented.pkl'))

    def forward_epoch(self, X, y, train=False):

        n_batches = int(np.ceil(len(X) / self.b_size))
        permutation = torch.randperm(len(X))
        total_loss = 0.
        for index in range(0, len(X), self.b_size):

            batch_x, batch_y = get_batch(X,
                                         y,
                                         permutation[index:index+self.b_size])

            pred_x = self.model(batch_x[:, self.fa_inputs])

            loss = self.criterion(batch_y, pred_x)
            if train:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            total_loss += loss.item() / n_batches
        return total_loss

    def train_mlp(self, train_x, train_y, valid_x, valid_y, wandb_logging):
        best_loss = np.inf

        for n in range(self.n_iters):

            # Training phase
            tr_loss = self.forward_epoch(train_x, train_y, train=True)
            self.scheduler.step()

            with torch.no_grad():

                # Loss of trained model on whole training set
                tr_loss = self.forward_epoch(train_x, train_y)

                # Validation phase
                va_loss = self.forward_epoch(valid_x, valid_y)

                # Save model
                if va_loss < best_loss:
                    if self.saving:
                        torch.save(self.model.state_dict(), os.path.join(self.model_path, 'augmented.pt'))
                    best_loss = va_loss

                    if wandb_logging:
                        wandb.run.summary['mlp best loss'] = va_loss
                        wandb.run.summary['mlp best epoch'] = n+1

                if wandb_logging:
                    log_dict = {f"mlp/train_loss_mlp": tr_loss,
                                f"mlp/valid_loss_mlp": va_loss,
                                f"mlp/learning rate_mlp": self.optim.param_groups[0]['lr'] if self.optim is not None else None,
                                f"mlp/epoch": settings.mlp_epoch+1}

                    wandb.log(log_dict)

            settings.mlp_epoch += 1
            if self.verbose:
                print('Finished epoch {:04d}/{:04d} | Train loss : {:.6f} | Validation loss: {:.6f}'.format(n+1, self.n_iters, tr_loss, va_loss))

        return best_loss

    def train(self, train_x, train_y, valid_x, valid_y, wandb_logging):

        if self.fa_mlp:
            best_loss = self.train_mlp(train_x, train_y, valid_x, valid_y, wandb_logging)
        else:
            self.model.fit(train_x[:, self.fa_inputs], train_y)
            best_loss = self.validate(valid_x, valid_y)
            if self.saving:
                save_model(self.model, self.model_path, 'augmented')

        return best_loss

    def validate(self, valid_x, valid_y):
        if self.fa_mlp:
            with torch.no_grad():
                pred = self.model(valid_x[:, self.fa_inputs])
        else:
            pred = torch.from_numpy(self.model.predict(valid_x[:, self.fa_inputs]))

        val_mse = self.criterion(valid_y, pred)
        return val_mse

    def predict(self, test_x):
        if self.fa_mlp:
            with torch.no_grad():
                pred = self.model(test_x[:, self.fa_inputs])
        else:
            pred = torch.from_numpy(self.model.predict(test_x[:, self.fa_inputs]))
        return pred
