import os
import wandb
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from xgboost import XGBRegressor
from models import Forecaster, MLP
from sklearn.ensemble import RandomForestRegressor
from utils import make_known_model, mse_objective, save_model, load_model, \
    get_batch


def make_settings(problem, problem_setting, fp_term, exp_id, model_name,
    hyperparameter, real_params, wandb_logging, fp_inputs, fa_inputs, out_size, saving,
    train_mean_X, train_std_X, train_mean_y, train_std_y, seed, extrapolation, filter_fa):
    settings = {
        'fp_known_then_mlp': {
            'fp_only': False, 'fa_only': False, 'fp_constant': False,
            'fp_known': True, 'fa_mlp': True, 'problem': problem,
            'fp_term': fp_term, 'problem_setting': problem_setting,
            'exp_id': exp_id, 'model_name': model_name,
            'real_params': real_params, 'hyperparameter': hyperparameter,
            'fp_inputs': fp_inputs, 'fa_inputs': fa_inputs, 'out_size': out_size,
            'saving': saving, 'train_mean_X': train_mean_X, 'train_std_X': train_std_X,
            'train_mean_y': train_mean_y, 'train_std_y': train_std_y, 'seed': seed,
            'filter_fa': filter_fa, 'extrapolation': extrapolation,
        },
        'fp_known_then_trees': {
            'fp_only': False, 'fa_only': False, 'fp_constant': False,
            'fp_known': True, 'fa_mlp': False, 'problem': problem,
            'fp_term': fp_term, 'problem_setting': problem_setting,
            'exp_id': exp_id, 'model_name': model_name,
            'real_params': real_params, 'hyperparameter': hyperparameter,
            'fp_inputs': fp_inputs, 'fa_inputs': fa_inputs, 'out_size': out_size,
            'saving': saving, 'train_mean_X': train_mean_X, 'train_std_X': train_std_X,
            'train_mean_y': train_mean_y, 'train_std_y': train_std_y, 'seed': seed,
            'filter_fa': filter_fa, 'extrapolation': extrapolation,
        },
        'fp_known_only': {
            'fp_only': True, 'fa_only': False, 'fp_constant': False,
            'fp_known': True, 'fa_mlp': False, 'problem': problem,
            'fp_term': fp_term, 'problem_setting': problem_setting,
            'exp_id': exp_id, 'model_name': model_name,
            'real_params': real_params, 'hyperparameter': hyperparameter,
            'fp_inputs': fp_inputs, 'fa_inputs': fa_inputs, 'out_size': out_size,
            'saving': saving, 'train_mean_X': train_mean_X, 'train_std_X': train_std_X,
            'train_mean_y': train_mean_y, 'train_std_y': train_std_y, 'seed': seed,
            'filter_fa': filter_fa, 'extrapolation': extrapolation,
        },
        'mlp_only': {
            'fp_only': False, 'fa_only': True, 'fp_constant': False,
            'fp_known': False, 'fa_mlp': True, 'problem': problem,
            'fp_term': fp_term, 'problem_setting': problem_setting,
            'exp_id': exp_id, 'model_name': model_name,
            'real_params': real_params, 'hyperparameter': hyperparameter,
            'fp_inputs': fp_inputs, 'fa_inputs': fa_inputs, 'out_size': out_size,
            'saving': saving, 'train_mean_X': train_mean_X, 'train_std_X': train_std_X,
            'train_mean_y': train_mean_y, 'train_std_y': train_std_y, 'seed': seed,
            'filter_fa': filter_fa, 'extrapolation': extrapolation,
        },
        'trees_only': {
            'fp_only': False, 'fa_only': True, 'fp_constant': False,
            'fp_known': False, 'fa_mlp': False, 'problem': problem,
            'fp_term': fp_term, 'problem_setting': problem_setting,
            'exp_id': exp_id, 'model_name': model_name,
            'real_params': real_params, 'hyperparameter': hyperparameter,
            'fp_inputs': fp_inputs, 'fa_inputs': fa_inputs, 'out_size': out_size,
            'saving': saving, 'train_mean_X': train_mean_X, 'train_std_X': train_std_X,
            'train_mean_y': train_mean_y, 'train_std_y': train_std_y, 'seed': seed,
            'filter_fa': filter_fa, 'extrapolation': extrapolation,
        },
        'fp_only_without_constant': {
            'fp_only': True, 'fa_only': False, 'fp_constant': False,
            'fp_known': False, 'fa_mlp': False, 'problem': problem,
            'fp_term': fp_term, 'problem_setting': problem_setting,
            'exp_id': exp_id, 'model_name': model_name,
            'real_params': real_params, 'hyperparameter': hyperparameter,
            'fp_inputs': fp_inputs, 'fa_inputs': fa_inputs, 'out_size': out_size,
            'saving': saving, 'train_mean_X': train_mean_X, 'train_std_X': train_std_X,
            'train_mean_y': train_mean_y, 'train_std_y': train_std_y, 'seed': seed,
            'filter_fa': filter_fa, 'extrapolation': extrapolation,
        },
        'fp_only_with_constant': {
            'fp_only': True, 'fa_only': False, 'fp_constant': True,
            'fp_known': False, 'fa_mlp': False, 'problem': problem,
            'fp_term': fp_term, 'problem_setting': problem_setting,
            'exp_id': exp_id, 'model_name': model_name,
            'real_params': real_params, 'hyperparameter': hyperparameter,
            'fp_inputs': fp_inputs, 'fa_inputs': fa_inputs, 'out_size': out_size,
            'saving': saving, 'train_mean_X': train_mean_X, 'train_std_X': train_std_X,
            'train_mean_y': train_mean_y, 'train_std_y': train_std_y, 'seed': seed,
            'filter_fa': filter_fa, 'extrapolation': extrapolation,
        },
        'fp_with_constant_then_mlp': {
            'fp_only': False, 'fa_only': False, 'fp_constant': True,
            'fp_known': False, 'fa_mlp': True, 'problem': problem,
            'fp_term': fp_term, 'problem_setting': problem_setting,
            'exp_id': exp_id, 'model_name': model_name,
            'real_params': real_params, 'hyperparameter': hyperparameter,
            'fp_inputs': fp_inputs, 'fa_inputs': fa_inputs, 'out_size': out_size,
            'saving': saving, 'train_mean_X': train_mean_X, 'train_std_X': train_std_X,
            'train_mean_y': train_mean_y, 'train_std_y': train_std_y, 'seed': seed,
            'filter_fa': filter_fa, 'extrapolation': extrapolation,
        },
        'fp_with_constant_then_trees': {
            'fp_only': False, 'fa_only': False, 'fp_constant': True,
            'fp_known': False, 'fa_mlp': False, 'problem': problem,
            'fp_term': fp_term, 'problem_setting': problem_setting,
            'exp_id': exp_id, 'model_name': model_name,
            'real_params': real_params, 'hyperparameter': hyperparameter,
            'fp_inputs': fp_inputs, 'fa_inputs': fa_inputs, 'out_size': out_size,
            'saving': saving, 'train_mean_X': train_mean_X, 'train_std_X': train_std_X,
            'train_mean_y': train_mean_y, 'train_std_y': train_std_y, 'seed': seed,
            'filter_fa': filter_fa, 'extrapolation': extrapolation,
        }
    }

    logging = {
        'fp_known_then_mlp': False, 'fp_known_then_trees': False,
        'fp_known_only': False, 'mlp_only': False, 'trees_only': False,
        'fp_only_without_constant': wandb_logging,
        'fp_only_with_constant': wandb_logging,
        'fp_with_constant_then_mlp': wandb_logging,
        'fp_with_constant_then_trees': wandb_logging
    }

    return settings, logging

def make_hyperparameter(model_name, incomplete, N, d, boosting, fp_learning_rate,
    learning_rate_trees, max_features, min_samples_split, num_layers, hidden_size,
    b_size, mlp_learning_rate):

    if model_name == 'fp_known_then_mlp':
        hyperparameter = (mlp_learning_rate,
                          num_layers,
                          hidden_size,
                          b_size,
                          incomplete)
    elif model_name == 'fp_known_then_trees':
        hyperparameter = (N,
                          d,
                          learning_rate_trees,
                          boosting,
                          max_features,
                          min_samples_split,
                          incomplete)
    elif model_name == 'fp_known_only':
        hyperparameter = (incomplete,)
    elif model_name == 'mlp_only':
        hyperparameter = (mlp_learning_rate,
                          num_layers,
                          hidden_size,
                          b_size,
                          incomplete)
    elif model_name == 'trees_only':
        hyperparameter = (N,
                          d,
                          learning_rate_trees,
                          boosting,
                          max_features,
                          min_samples_split,
                          incomplete)
    elif model_name == 'fp_only_without_constant':
        hyperparameter = (fp_learning_rate,
                          b_size,
                          incomplete)
    elif model_name == 'fp_only_with_constant':
        hyperparameter = (fp_learning_rate,
                          b_size,
                          incomplete)
    elif model_name == 'fp_with_constant_then_mlp':
        hyperparameter = (fp_learning_rate,
                          num_layers,
                          hidden_size,
                          b_size,
                          mlp_learning_rate,
                          incomplete)
    elif model_name == 'fp_with_constant_then_trees':
        hyperparameter = (fp_learning_rate,
                          b_size,
                          N,
                          d,
                          learning_rate_trees,
                          boosting,
                          max_features,
                          min_samples_split,
                          incomplete)
    return hyperparameter


class ParametricModel():

    def __init__(self, problem, term, incomplete, exp_id, fp_inputs, fa_inputs,
        out_size, learning_rate, num_layers, hidden_size, b_size, saving,
        criterion, save_path, augmented, device='cpu', fp_pretrained=None,
        fp_known=False, fp_with_constant=False, n_iters=3000, real_params=None,
        fp_eval=False, train_mean_X=None, train_std_X=None, train_mean_y=None,
        train_std_y=None, seed=None):

        self.problem = problem
        self.term = term
        self.incomplete = incomplete
        self.exp_id = exp_id
        self.fp_inputs = fp_inputs
        self.fa_inputs = fa_inputs
        self.out_size = out_size

        self.criterion = criterion
        self.augmented = augmented
        self.saving = saving
        self.device = device

        # If True, the parameters of F_p are assumed to be known
        self.fp_known = fp_known

        # If True, F_p will be fitted along with a constant
        self.fp_with_constant = fp_with_constant

        self.n_iters = n_iters
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.b_size = b_size
        self.learning_rate = learning_rate
        self.train_mean_X = train_mean_X
        self.train_std_X = train_std_X
        self.train_mean_y = train_mean_y
        self.train_std_y = train_std_y
        self.seed = seed

        # Relevant only for baselines dealing with an augmented component
        self.fp = fp_pretrained
        if self.fp is not None and fp_eval:
            self.fp.eval()

        # Initialize parametric model
        if self.augmented:
            self.model = MLP(in_size=len(self.fa_inputs),
                             num_layers=self.num_layers,
                             hidden_size=self.hidden_size,
                             out_size=self.out_size,
                             train_mean_X=None,
                             train_std_X=None,
                             train_mean_y=self.train_mean_y,
                             train_std_y=self.train_std_y).to(self.device)

            self.model = Forecaster(self.fp, self.model, is_augmented=True, jointly=True)
            self.save_path = os.path.join(save_path, 'augmented.pt')

            # Define optimizer
            self.optim = optim.Adam(self.model.parameters(),
                                    lr=self.learning_rate)
            self.scheduler = CosineAnnealingLR(self.optim, T_max=self.n_iters)

        else:
            self.model = make_known_model(self.problem,
                                          self.term,
                                          self.incomplete,
                                          constant=self.fp_with_constant,
                                          real_params=real_params,
                                          seed=self.seed).to(self.device)
            self.model = Forecaster(self.model, None, is_augmented=False, jointly=True)
            self.save_path = os.path.join(save_path, 'known.pt')

            # Define optimizer
            if not self.fp_known:
                self.optim = optim.Adam(self.model.parameters(),
                                        lr=self.learning_rate)
                self.scheduler = CosineAnnealingLR(self.optim, T_max=self.n_iters)
            else:
                self.optim = None
                self.scheduler = None

    def load_parametric(self, model_path, augmented):
        if augmented:
            self.model.model_aug.load_state_dict(torch.load(model_path))
        else:
            self.model.model_phy.load_state_dict(torch.load(model_path))

    def compute_fp_distance(self, train_x, train_y):
        with torch.no_grad():
            fp_pred = self.model.model_phy(train_x).squeeze()

        # This definition of the norm is only valid given that y has a single dimension
        distance = nn.MSELoss()(fp_pred, train_y).item()
        return distance

    def forward_epoch(self, X, y, train=False):

        n_batches = int(np.ceil(X.shape[0] / self.b_size))
        permutation = torch.randperm(X.shape[0])
        total_loss = 0.
        for index in range(0, X.shape[0], self.b_size):

            batch_x, batch_y = get_batch(X,
                                         y,
                                         permutation[index:index+self.b_size],
                                         self.device)

            pred_x = self.model(batch_x, batch_x[:, self.fa_inputs]).to(self.device)

            loss = self.criterion(batch_y, pred_x)
            if train:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                if self.problem in ['friedman1', 'corr_friedman1']:
                    if not self.augmented:
                        w = self.model.model_phy.p3.data
                        w = w.clamp(0, 1)
                        self.model.model_phy.p3.data = w

            total_loss += loss.item() / n_batches
        return total_loss

    def train(self, train_x, train_y, valid_x, valid_y, wandb_logging):

        # Training F_p but its parameters are already known
        if not self.augmented and self.fp_known:
            if self.saving:
                torch.save(self.model.model_phy.state_dict(), self.save_path)
            return 0.0

        else:

            best_loss = np.inf

            for n in range(self.n_iters):

                # Training phase
                tr_loss = self.forward_epoch(train_x,
                                             train_y,
                                             train=True)

                if self.scheduler is not None:
                    self.scheduler.step()

                with torch.no_grad():

                    # Loss of trained model on whole training set
                    tr_loss = self.forward_epoch(train_x, train_y)

                    # Validation phase
                    va_loss = self.forward_epoch(valid_x, valid_y)

                    if not self.augmented:
                        # Compute distance(fp, f)
                        distance = self.compute_fp_distance(train_x, train_y)

                    # Save model
                    if va_loss < best_loss:
                        if self.saving:
                            if self.augmented:
                                torch.save(self.model.model_aug.state_dict(), self.save_path)
                            else:
                                torch.save(self.model.model_phy.state_dict(), self.save_path)
                        best_loss = va_loss

                        if wandb_logging and not self.augmented:
                            wandb.run.summary['best loss'] = va_loss
                            wandb.run.summary['best epoch'] = n+1

                    if wandb_logging and not self.augmented:
                        log_dict = {f"train_loss": tr_loss,
                                    f"valid_loss": va_loss,
                                    f"learning rate": self.optim.param_groups[0]['lr'] if self.optim is not None else None,
                                    f"d(fp, f)": distance,
                                    f"epoch": n+1}

                        if self.problem in ['friedman1', 'corr_friedman1', 'linear_data', 'overlap_data']:
                            params = []
                            if self.problem in ['friedman1', 'corr_friedman1']:
                                if self.incomplete:
                                    if self.term == 'first':
                                        params_name = [f'p0', f'p1']
                                        params.append(self.model.model_phy.p0.item())
                                        params.append(self.model.model_phy.p1.item())
                                    elif self.term == 'second':
                                        params_name = [f'p2', f'p3']
                                        params.append(self.model.model_phy.p2.item())
                                        params.append(self.model.model_phy.p3.item())
                                    elif self.term == 'third':
                                        params_name = [f'p4', f'p5']
                                        params.append(self.model.model_phy.p4.item())
                                        params.append(self.model.model_phy.p5.item())
                                else:
                                    params_name = [f'p0', f'p1', f'p2', f'p3', f'p4', f'p5']
                                    params.append(self.model.model_phy.p0.item())
                                    params.append(self.model.model_phy.p1.item())
                                    params.append(self.model.model_phy.p2.item())
                                    params.append(self.model.model_phy.p3.item())
                                    params.append(self.model.model_phy.p4.item())
                                    params.append(self.model.model_phy.p5.item())

                            elif self.problem in ['linear_data', 'overlap_data']:
                                params.append(self.model.model_phy.beta.item())
                                params_name = [f'beta_{self.exp_id}']

                            param_dict = {}
                            for param, name in zip(params, params_name):
                                param_dict[name] = param
                            log_dict.update(param_dict)

                        wandb.log(log_dict)

                print('Finished epoch {:04d}/{:04d} | Train loss : {:.6f} | Validation loss: {:.6f}'.format(n+1, self.n_iters, tr_loss, va_loss))

            return best_loss

    def validate(self, valid_x, valid_y):

        # Validation phase
        with torch.no_grad():
            va_loss = self.forward_epoch(valid_x, valid_y)
        return va_loss

    def predict(self, test_x):

        with torch.no_grad():
            pred_x = self.model(test_x, test_x[:, self.fa_inputs]).to(self.device)
        return pred_x

class Tree_Model():

    def __init__(self, problem, term, N, d, boosting, learning_rate_trees,
        max_features, min_samples_split, saving, criterion, device, save_path,
        fa_inputs, seed, fp_pretrained=None):

        self.fa_inputs = fa_inputs
        self.problem = problem
        self.term = term

        self.N = N
        self.d = d
        self.learning_rate_trees = learning_rate_trees
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.boosting = boosting
        self.seed = seed

        self.criterion = criterion
        self.device = device
        self.saving = saving
        self.save_path = save_path
        self.fp = fp_pretrained

    def load_tree_model(self, model_path):
        self.model_aug = load_model(os.path.join(model_path))

    def train(self, train_x, train_y):

        if self.fp is not None:
            with torch.no_grad():
                phy_term = self.fp(train_x)
        else:
            phy_term = torch.zeros(train_x.shape[0])

        if self.boosting:
            self.model_aug = XGBRegressor(n_estimators=self.N,
                                          max_depth=self.d,
                                          learning_rate=self.learning_rate_trees,
                                          n_jobs=-1,
                                          random_state=self.seed,
                                          objective=mse_objective(phy_term.detach().cpu().numpy(),
                                                                  None,
                                                                  False))
            self.model_aug.fit(train_x[:, self.fa_inputs].cpu(), train_y.cpu())

        else:
            self.model_aug = RandomForestRegressor(n_estimators=self.N,
                                                   max_features=self.max_features,
                                                   min_samples_split=self.min_samples_split,
                                                   random_state=self.seed,
                                                   n_jobs=-1)
            self.model_aug.fit(train_x[:, self.fa_inputs], train_y - phy_term.detach())

        if self.saving:
            # Save augmented model
            save_model(self.model_aug, self.save_path, 'augmented')

    def validate(self, valid_x, valid_y):

        if self.fp is not None:
            with torch.no_grad():
                phy_term = self.fp(valid_x)
        else:
            phy_term = torch.zeros(valid_x.shape[0])

        pred_x = phy_term + torch.from_numpy(self.model_aug.predict(valid_x[:, self.fa_inputs]))

        va_loss = self.criterion(pred_x, valid_y)
        return va_loss

    def predict(self, test_x):

        if self.fp is not None:
            with torch.no_grad():
                phy_term = self.fp(test_x)
        else:
            phy_term = torch.zeros(test_x.shape[0])

        pred_x = phy_term + torch.from_numpy(self.model_aug.predict(test_x[:, self.fa_inputs]))
        return pred_x

class Baseline():

    def __init__(self, fp_only, fa_only, fp_constant, fp_known, fa_mlp,
        problem, problem_setting, fp_term, exp_id, saving, model_name,
        hyperparameter, real_params, fp_inputs, fa_inputs, out_size, train_mean_X=None,
        train_std_X=None, train_mean_y=None, train_std_y=None, seed=None,
        filter_fa=False, extrapolation=False, n_iters=3000):

        self.fp_only = fp_only
        self.fa_only = fa_only
        self.fp_constant = fp_constant
        self.fp_known = fp_known
        self.fa_mlp = fa_mlp
        self.problem = problem
        self.problem_setting = problem_setting
        self.fp_term = fp_term
        self.exp_id = exp_id
        self.saving = saving
        self.real_params = real_params
        self.fp_inputs = fp_inputs
        self.fa_inputs = fa_inputs
        self.out_size = out_size
        self.train_mean_X = train_mean_X
        self.train_std_X = train_std_X
        self.train_mean_y = train_mean_y
        self.train_std_y = train_std_y
        self.n_iters = n_iters
        self.seed = seed
        self.filter_fa = filter_fa
        self.extrapolation = extrapolation
        self.model_name = model_name
        self.incomplete = hyperparameter[-1]

        self.standardize = self.train_mean_X is not None

        # Initialize model and results paths
        MOD_OUTPUT_PREFIX = self.problem + '/' + self.problem_setting + '/' + self.fp_term + '/models/baselines'
        RES_OUTPUT_PREFIX = self.problem + '/' + self.problem_setting + '/' + self.fp_term + '/results/baselines'

        if self.model_name == 'fp_known_then_mlp':
            self.mlp_learning_rate = hyperparameter[0]
            self.num_layers = hyperparameter[1]
            self.hidden_size = hyperparameter[2]
            self.b_size = hyperparameter[3]
            self.model_path = os.path.join(MOD_OUTPUT_PREFIX,
                                           'fp_known_then_mlp',
                                           'incomplete' if self.incomplete else 'complete',
                                           'filter' if self.filter_fa else 'full',
                                           'extrapolation' if self.extrapolation else 'interpolation',
                                           str(self.mlp_learning_rate),
                                           str(self.num_layers),
                                           str(self.hidden_size),
                                           str(self.b_size),
                                           'exp_' + str(self.exp_id))
            self.res_path = os.path.join(RES_OUTPUT_PREFIX,
                                         'fp_known_then_mlp',
                                         'incomplete' if self.incomplete else 'complete',
                                         'filter' if self.filter_fa else 'full',
                                         'extrapolation' if self.extrapolation else 'interpolation',
                                         str(self.mlp_learning_rate),
                                         str(self.num_layers),
                                         str(self.hidden_size),
                                         str(self.b_size))

            self.known_model = ParametricModel(problem=self.problem,
                                               term=self.fp_term,
                                               incomplete=self.incomplete,
                                               exp_id=self.exp_id,
                                               fp_inputs=self.fp_inputs,
                                               fa_inputs=self.fa_inputs,
                                               out_size=None,
                                               learning_rate=None,
                                               num_layers=None,
                                               hidden_size=None,
                                               b_size=25,
                                               saving=self.saving,
                                               criterion=nn.MSELoss(),
                                               save_path=self.model_path,
                                               augmented=False,
                                               fp_known=self.fp_known,
                                               real_params=self.real_params,
                                               fp_eval=True,
                                               train_mean_X=None,
                                               train_std_X=None,
                                               train_mean_y=None,
                                               train_std_y=None,
                                               seed=self.seed)

            self.augmented_model = ParametricModel(problem=self.problem,
                                                   term=self.fp_term,
                                                   incomplete=self.incomplete,
                                                   exp_id=self.exp_id,
                                                   fp_inputs=self.fp_inputs,
                                                   fa_inputs=self.fa_inputs,
                                                   out_size=self.out_size,
                                                   learning_rate=self.mlp_learning_rate,
                                                   num_layers=self.num_layers,
                                                   hidden_size=self.hidden_size,
                                                   b_size=self.b_size,
                                                   saving=self.saving,
                                                   criterion=nn.MSELoss(),
                                                   save_path=self.model_path,
                                                   augmented=True,
                                                   fp_pretrained=self.known_model.model.model_phy,
                                                   fp_known=self.fp_known,
                                                   train_mean_X=self.train_mean_X,
                                                   train_std_X=self.train_std_X,
                                                   train_mean_y=self.train_mean_y,
                                                   train_std_y=self.train_std_y)

        elif self.model_name == 'fp_known_then_trees':
            self.N = hyperparameter[0]
            self.d = hyperparameter[1]
            self.learning_rate_trees = hyperparameter[2]
            self.boosting = hyperparameter[3]
            self.max_features = hyperparameter[4]
            self.min_samples_split = hyperparameter[5]

            if self.boosting:
                self.model_path = os.path.join(MOD_OUTPUT_PREFIX,
                                               'fp_known_then_trees',
                                               'incomplete' if self.incomplete else 'complete',
                                               'xgboost',
                                               'filter' if self.filter_fa else 'full',
                                               'extrapolation' if self.extrapolation else 'interpolation',
                                               str(self.N),
                                               str(self.d),
                                               str(self.learning_rate_trees),
                                               'exp_' + str(self.exp_id))
                self.res_path = os.path.join(RES_OUTPUT_PREFIX,
                                             'fp_known_then_trees',
                                             'incomplete' if self.incomplete else 'complete',
                                             'xgboost',
                                             'filter' if self.filter_fa else 'full',
                                             'extrapolation' if self.extrapolation else 'interpolation',
                                             str(self.N),
                                             str(self.d),
                                             str(self.learning_rate_trees))
            else:
                self.model_path = os.path.join(MOD_OUTPUT_PREFIX,
                                               'fp_known_then_trees',
                                               'incomplete' if self.incomplete else 'complete',
                                               'rf',
                                               'filter' if self.filter_fa else 'full',
                                               'extrapolation' if self.extrapolation else 'interpolation',
                                               str(self.N),
                                               str(self.max_features),
                                               str(self.min_samples_split),
                                               'exp_' + str(self.exp_id))
                self.res_path = os.path.join(RES_OUTPUT_PREFIX,
                                             'fp_known_then_trees',
                                             'incomplete' if self.incomplete else 'complete',
                                             'rf',
                                             'filter' if self.filter_fa else 'full',
                                             'extrapolation' if self.extrapolation else 'interpolation',
                                             str(self.N),
                                             str(self.max_features),
                                             str(self.min_samples_split))

            self.known_model = ParametricModel(problem=self.problem,
                                               term=self.fp_term,
                                               incomplete=self.incomplete,
                                               exp_id=self.exp_id,
                                               fp_inputs=self.fp_inputs,
                                               fa_inputs=self.fa_inputs,
                                               out_size=None,
                                               learning_rate=None,
                                               num_layers=None,
                                               hidden_size=None,
                                               b_size=25,
                                               saving=self.saving,
                                               criterion=nn.MSELoss(),
                                               save_path=self.model_path,
                                               augmented=False,
                                               fp_known=self.fp_known,
                                               real_params=self.real_params,
                                               fp_eval=True,
                                               train_mean_X=None,
                                               train_std_X=None,
                                               train_mean_y=None,
                                               train_std_y=None,
                                               seed=self.seed)

            self.augmented_model = Tree_Model(problem=self.problem,
                                              term=self.fp_term,
                                              N=self.N,
                                              d=self.d,
                                              boosting=self.boosting,
                                              learning_rate_trees=self.learning_rate_trees,
                                              max_features=self.max_features,
                                              min_samples_split=self.min_samples_split,
                                              saving=self.saving,
                                              criterion=nn.MSELoss(),
                                              device='cpu',
                                              save_path=self.model_path,
                                              fa_inputs=self.fa_inputs,
                                              seed=self.seed,
                                              fp_pretrained=self.known_model.model)

        elif self.model_name == 'fp_known_only':
            self.model_path = os.path.join(MOD_OUTPUT_PREFIX,
                                           'fp_known_only',
                                           'incomplete' if self.incomplete else 'complete',
                                           'exp_' + str(self.exp_id))
            self.res_path = os.path.join(RES_OUTPUT_PREFIX,
                                         'fp_known_only',
                                         'incomplete' if self.incomplete else 'complete')

            self.known_model = ParametricModel(problem=self.problem,
                                               term=self.fp_term,
                                               incomplete=self.incomplete,
                                               exp_id=self.exp_id,
                                               fp_inputs=self.fp_inputs,
                                               fa_inputs=self.fa_inputs,
                                               out_size=None,
                                               learning_rate=None,
                                               num_layers=None,
                                               hidden_size=None,
                                               b_size=25,
                                               saving=self.saving,
                                               criterion=nn.MSELoss(),
                                               save_path=self.model_path,
                                               augmented=False,
                                               fp_known=self.fp_known,
                                               real_params=self.real_params,
                                               fp_eval=True,
                                               train_mean_X=None,
                                               train_std_X=None,
                                               train_mean_y=None,
                                               train_std_y=None,
                                               seed=self.seed)

            self.augmented_model = None

        elif self.model_name == 'mlp_only':
            self.mlp_learning_rate = hyperparameter[0]
            self.num_layers = hyperparameter[1]
            self.hidden_size = hyperparameter[2]
            self.b_size = hyperparameter[3]
            self.model_path = os.path.join(MOD_OUTPUT_PREFIX,
                                           'mlp_only',
                                           'incomplete' if self.incomplete else 'complete',
                                           'filter' if self.filter_fa else 'full',
                                           'extrapolation' if self.extrapolation else 'interpolation',
                                           str(self.mlp_learning_rate),
                                           str(self.num_layers),
                                           str(self.hidden_size),
                                           str(self.b_size),
                                           'exp_' + str(self.exp_id))
            self.res_path = os.path.join(RES_OUTPUT_PREFIX,
                                         'mlp_only',
                                         'incomplete' if self.incomplete else 'complete',
                                         'filter' if self.filter_fa else 'full',
                                         'extrapolation' if self.extrapolation else 'interpolation',
                                         str(self.mlp_learning_rate),
                                         str(self.num_layers),
                                         str(self.hidden_size),
                                         str(self.b_size))

            self.known_model = None

            self.augmented_model = ParametricModel(problem=self.problem,
                                                   term=self.fp_term,
                                                   incomplete=self.incomplete,
                                                   exp_id=self.exp_id,
                                                   fp_inputs=self.fp_inputs,
                                                   fa_inputs=self.fa_inputs,
                                                   out_size=self.out_size,
                                                   learning_rate=self.mlp_learning_rate,
                                                   num_layers=self.num_layers,
                                                   hidden_size=self.hidden_size,
                                                   b_size=self.b_size,
                                                   saving=self.saving,
                                                   criterion=nn.MSELoss(),
                                                   save_path=self.model_path,
                                                   augmented=True,
                                                   fp_known=self.fp_known,
                                                   train_mean_X=self.train_mean_X,
                                                   train_std_X=self.train_std_X,
                                                   train_mean_y=self.train_mean_y,
                                                   train_std_y=self.train_std_y)

            self.wandb_config = {'mlp_learning_rate': self.mlp_learning_rate,
                                 'incomplete': self.incomplete,
                                 'problem': self.problem,
                                 'problem_setting': self.problem_setting,
                                 'fp_term': self.fp_term,
                                 'num_layers': self.num_layers,
                                 'hidden_size': self.hidden_size,
                                 'fa_inputs': self.fa_inputs,
                                 'b_size': self.b_size,
                                 'standardize': self.standardize,
                                 'exp_id': self.exp_id,
                                 'fp_only': self.fp_only,
                                 'fa_only': self.fa_only,
                                 'fa_mlp': self.fa_mlp,
                                 'fp_with_constant': self.fp_constant,
                                 'filter_fa': self.filter_fa,
                                 'epochs': self.n_iters}

        elif self.model_name == 'trees_only':
            self.N = hyperparameter[0]
            self.d = hyperparameter[1]
            self.learning_rate_trees = hyperparameter[2]
            self.boosting = hyperparameter[3]
            self.max_features = hyperparameter[4]
            self.min_samples_split = hyperparameter[5]

            if self.boosting:
                self.model_path = os.path.join(MOD_OUTPUT_PREFIX,
                                               'trees_only',
                                               'incomplete' if self.incomplete else 'complete',
                                               'xgboost',
                                               'filter' if self.filter_fa else 'full',
                                               'extrapolation' if self.extrapolation else 'interpolation',
                                               str(self.N),
                                               str(self.d),
                                               str(self.learning_rate_trees),
                                               'exp_' + str(self.exp_id))
                self.res_path = os.path.join(RES_OUTPUT_PREFIX,
                                             'trees_only',
                                             'incomplete' if self.incomplete else 'complete',
                                             'xgboost',
                                             'filter' if self.filter_fa else 'full',
                                             'extrapolation' if self.extrapolation else 'interpolation',
                                             str(self.N),
                                             str(self.d),
                                             str(self.learning_rate_trees))
            else:
                self.model_path = os.path.join(MOD_OUTPUT_PREFIX,
                                               'trees_only',
                                               'incomplete' if self.incomplete else 'complete',
                                               'rf',
                                               'filter' if self.filter_fa else 'full',
                                               'extrapolation' if self.extrapolation else 'interpolation',
                                               str(self.N),
                                               str(self.max_features),
                                               str(self.min_samples_split),
                                               'exp_' + str(self.exp_id))
                self.res_path = os.path.join(RES_OUTPUT_PREFIX,
                                             'trees_only',
                                             'incomplete' if self.incomplete else 'complete',
                                             'rf',
                                             'filter' if self.filter_fa else 'full',
                                             'extrapolation' if self.extrapolation else 'interpolation',
                                             str(self.N),
                                             str(self.max_features),
                                             str(self.min_samples_split))

            self.known_model = None

            self.augmented_model = Tree_Model(problem=self.problem,
                                              term=self.fp_term,
                                              N=self.N,
                                              d=self.d,
                                              learning_rate_trees=self.learning_rate_trees,
                                              max_features=self.max_features,
                                              min_samples_split=self.min_samples_split,
                                              boosting=self.boosting,
                                              criterion=nn.MSELoss(),
                                              saving=self.saving,
                                              device='cpu',
                                              save_path=self.model_path,
                                              fa_inputs=self.fa_inputs,
                                              seed=self.seed,
                                              fp_pretrained=None)

            self.wandb_config = {'incomplete': self.incomplete,
                                 'problem': self.problem,
                                 'problem_setting': self.problem_setting,
                                 'fp_term': self.fp_term,
                                 'learning_rate_trees': self.learning_rate_trees,
                                 'max_features': self.max_features,
                                 'min_samples_split': self.min_samples_split,
                                 'boosting': self.boosting,
                                 'fa_inputs': self.fa_inputs,
                                 'exp_id': self.exp_id,
                                 'fp_only': self.fp_only,
                                 'fa_only': self.fa_only,
                                 'fa_mlp': self.fa_mlp,
                                 'fp_with_constant': self.fp_constant,
                                 'N': self.N,
                                 'max_d': self.d,
                                 'filter_fa': self.filter_fa,
                                 'epochs': self.n_iters}

        elif self.model_name == 'fp_only_without_constant':
            self.fp_learning_rate = hyperparameter[0]
            self.b_size = hyperparameter[1]
            self.model_path = os.path.join(MOD_OUTPUT_PREFIX,
                                           'fp_only_without_constant',
                                           'incomplete' if self.incomplete else 'complete',
                                           'extrapolation' if self.extrapolation else 'interpolation',
                                           str(self.fp_learning_rate),
                                           str(self.b_size),
                                           'exp_' + str(self.exp_id))
            self.res_path = os.path.join(RES_OUTPUT_PREFIX,
                                         'fp_only_without_constant',
                                         'incomplete' if self.incomplete else 'complete',
                                         'extrapolation' if self.extrapolation else 'interpolation',
                                         str(self.fp_learning_rate),
                                         str(self.b_size))

            self.known_model = ParametricModel(problem=self.problem,
                                               term=self.fp_term,
                                               incomplete=self.incomplete,
                                               exp_id=self.exp_id,
                                               num_layers=None,
                                               hidden_size=None,
                                               fp_inputs=self.fp_inputs,
                                               fa_inputs=self.fa_inputs,
                                               out_size=None,
                                               learning_rate=self.fp_learning_rate,
                                               b_size=self.b_size,
                                               saving=self.saving,
                                               criterion=nn.MSELoss(),
                                               save_path=self.model_path,
                                               augmented=False,
                                               fp_known=self.fp_known,
                                               fp_with_constant=self.fp_constant,
                                               train_mean_X=None,
                                               train_std_X=None,
                                               train_mean_y=None,
                                               train_std_y=None,
                                               seed=self.seed)

            self.augmented_model = None

            self.wandb_config = {'fp_learning_rate': self.fp_learning_rate,
                                 'incomplete': self.incomplete,
                                 'fp_inputs': self.fp_inputs,
                                 'problem': self.problem,
                                 'problem_setting': self.problem_setting,
                                 'fp_term': self.fp_term,
                                 'b_size': self.b_size,
                                 'standardize': self.standardize,
                                 'exp_id': self.exp_id,
                                 'fp_only': self.fp_only,
                                 'fa_only': self.fa_only,
                                 'fa_mlp': self.fa_mlp,
                                 'fp_with_constant': self.fp_constant,
                                 'epochs': self.n_iters}

        elif self.model_name == 'fp_only_with_constant':
            self.fp_learning_rate = hyperparameter[0]
            self.b_size = hyperparameter[1]
            self.model_path = os.path.join(MOD_OUTPUT_PREFIX,
                                           'fp_only_with_constant',
                                           'incomplete' if self.incomplete else 'complete',
                                           'extrapolation' if self.extrapolation else 'interpolation',
                                           str(self.fp_learning_rate),
                                           str(self.b_size),
                                           'exp_' + str(self.exp_id))
            self.res_path = os.path.join(RES_OUTPUT_PREFIX,
                                         'fp_only_with_constant',
                                         'incomplete' if self.incomplete else 'complete',
                                         'extrapolation' if self.extrapolation else 'interpolation',
                                         str(self.fp_learning_rate),
                                         str(self.b_size))

            self.known_model = ParametricModel(problem=self.problem,
                                               term=self.fp_term,
                                               incomplete=self.incomplete,
                                               exp_id=self.exp_id,
                                               num_layers=None,
                                               hidden_size=None,
                                               fp_inputs=self.fp_inputs,
                                               fa_inputs=self.fa_inputs,
                                               out_size=None,
                                               learning_rate=self.fp_learning_rate,
                                               b_size=self.b_size,
                                               saving=self.saving,
                                               criterion=nn.MSELoss(),
                                               save_path=self.model_path,
                                               augmented=False,
                                               fp_known=self.fp_known,
                                               fp_with_constant=self.fp_constant,
                                               train_mean_X=None,
                                               train_std_X=None,
                                               train_mean_y=None,
                                               train_std_y=None,
                                               seed=self.seed)

            self.augmented_model = None

            self.wandb_config = {'fp_learning_rate': self.fp_learning_rate,
                                 'incomplete': self.incomplete,
                                 'fp_inputs': self.fp_inputs,
                                 'problem': self.problem,
                                 'problem_setting': self.problem_setting,
                                 'fp_term': self.fp_term,
                                 'b_size': self.b_size,
                                 'standardize': self.standardize,
                                 'exp_id': self.exp_id,
                                 'fp_only': self.fp_only,
                                 'fa_only': self.fa_only,
                                 'fa_mlp': self.fa_mlp,
                                 'fp_with_constant': self.fp_constant,
                                 'epochs': self.n_iters}

        elif self.model_name == 'fp_with_constant_then_mlp':
            self.fp_learning_rate = hyperparameter[0]
            self.num_layers = hyperparameter[1]
            self.hidden_size = hyperparameter[2]
            self.b_size = hyperparameter[3]
            self.mlp_learning_rate = hyperparameter[4]
            self.model_path = os.path.join(MOD_OUTPUT_PREFIX,
                                           'fp_with_constant_then_mlp',
                                           'incomplete' if self.incomplete else 'complete',
                                           'filter' if self.filter_fa else 'full',
                                           'extrapolation' if self.extrapolation else 'interpolation',
                                           str(self.fp_learning_rate),
                                           str(self.num_layers),
                                           str(self.hidden_size),
                                           str(self.mlp_learning_rate),
                                           str(self.b_size),
                                           'exp_' + str(self.exp_id))
            self.res_path = os.path.join(RES_OUTPUT_PREFIX,
                                         'fp_with_constant_then_mlp',
                                         'incomplete' if self.incomplete else 'complete',
                                         'filter' if self.filter_fa else 'full',
                                         'extrapolation' if self.extrapolation else 'interpolation',
                                         str(self.fp_learning_rate),
                                         str(self.num_layers),
                                         str(self.hidden_size),
                                         str(self.mlp_learning_rate),
                                         str(self.b_size))

            self.known_model = ParametricModel(problem=self.problem,
                                               term=self.fp_term,
                                               incomplete=self.incomplete,
                                               exp_id=self.exp_id,
                                               fp_inputs=self.fp_inputs,
                                               fa_inputs=self.fa_inputs,
                                               out_size=None,
                                               learning_rate=self.fp_learning_rate,
                                               num_layers=None,
                                               hidden_size=None,
                                               b_size=self.b_size,
                                               saving=True,
                                               criterion=nn.MSELoss(),
                                               save_path=self.model_path,
                                               augmented=False,
                                               fp_known=self.fp_known,
                                               fp_with_constant=self.fp_constant,
                                               train_mean_X=None,
                                               train_std_X=None,
                                               train_mean_y=None,
                                               train_std_y=None,
                                               seed=self.seed)

            # WARNING: The augmented model will rely on the fitted version of self.known_model
            # which is not fitted yet (fp_pretrained should be modified a posteriori)
            self.augmented_model = ParametricModel(problem=self.problem,
                                                   term=self.fp_term,
                                                   incomplete=self.incomplete,
                                                   exp_id=self.exp_id,
                                                   fp_inputs=self.fp_inputs,
                                                   fa_inputs=self.fa_inputs,
                                                   out_size=self.out_size,
                                                   learning_rate=self.mlp_learning_rate,
                                                   num_layers=self.num_layers,
                                                   hidden_size=self.hidden_size,
                                                   b_size=self.b_size,
                                                   saving=self.saving,
                                                   criterion=nn.MSELoss(),
                                                   save_path=self.model_path,
                                                   augmented=True,
                                                   fp_pretrained=None,
                                                   train_mean_X=self.train_mean_X,
                                                   train_std_X=self.train_std_X,
                                                   train_mean_y=self.train_mean_y,
                                                   train_std_y=self.train_std_y)

            self.wandb_config = {'fp_learning_rate': self.fp_learning_rate,
                                 'mlp_learning_rate': self.mlp_learning_rate,
                                 'incomplete': self.incomplete,
                                 'fp_inputs': self.fp_inputs,
                                 'fa_inputs': self.fa_inputs,
                                 'problem': self.problem,
                                 'problem_setting': self.problem_setting,
                                 'fp_term': self.fp_term,
                                 'num_layers': self.num_layers,
                                 'hidden_size': self.hidden_size,
                                 'b_size': self.b_size,
                                 'standardize': self.standardize,
                                 'exp_id': self.exp_id,
                                 'fp_only': self.fp_only,
                                 'fa_only': self.fa_only,
                                 'fa_mlp': self.fa_mlp,
                                 'fp_with_constant': self.fp_constant,
                                 'filter_fa': self.filter_fa,
                                 'epochs': self.n_iters}

        elif self.model_name == 'fp_with_constant_then_trees':
            self.fp_learning_rate = hyperparameter[0]
            self.b_size = hyperparameter[1]
            self.N = hyperparameter[2]
            self.d = hyperparameter[3]
            self.learning_rate_trees = hyperparameter[4]
            self.boosting = hyperparameter[5]
            self.max_features = hyperparameter[6]
            self.min_samples_split = hyperparameter[7]

            if self.boosting:
                self.model_path = os.path.join(MOD_OUTPUT_PREFIX,
                                               'fp_with_constant_then_trees',
                                               'incomplete' if self.incomplete else 'complete',
                                               'xgboost',
                                               'filter' if self.filter_fa else 'full',
                                               'extrapolation' if self.extrapolation else 'interpolation',
                                               str(self.N),
                                               str(self.d),
                                               str(self.b_size),
                                               str(self.learning_rate_trees),
                                               'exp_' + str(self.exp_id))
                self.res_path = os.path.join(RES_OUTPUT_PREFIX,
                                             'fp_with_constant_then_trees',
                                             'incomplete' if self.incomplete else 'complete',
                                             'xgboost',
                                             'filter' if self.filter_fa else 'full',
                                             'extrapolation' if self.extrapolation else 'interpolation',
                                             str(self.N),
                                             str(self.d),
                                             str(self.b_size),
                                             str(self.learning_rate_trees))
            else:
                self.model_path = os.path.join(MOD_OUTPUT_PREFIX,
                                               'fp_with_constant_then_trees',
                                               'incomplete' if self.incomplete else 'complete',
                                               'rf',
                                               'filter' if self.filter_fa else 'full',
                                               'extrapolation' if self.extrapolation else 'interpolation',
                                               str(self.N),
                                               str(self.max_features),
                                               str(self.min_samples_split),
                                               str(self.b_size),
                                               'exp_' + str(self.exp_id))
                self.res_path = os.path.join(RES_OUTPUT_PREFIX,
                                             'fp_with_constant_then_trees',
                                             'incomplete' if self.incomplete else 'complete',
                                             'rf',
                                             'filter' if self.filter_fa else 'full',
                                             'extrapolation' if self.extrapolation else 'interpolation',
                                             str(self.N),
                                             str(self.max_features),
                                             str(self.min_samples_split),
                                             str(self.b_size))

            self.known_model = ParametricModel(problem=self.problem,
                                               term=self.fp_term,
                                               incomplete=self.incomplete,
                                               exp_id=self.exp_id,
                                               fp_inputs=self.fp_inputs,
                                               fa_inputs=self.fa_inputs,
                                               out_size=None,
                                               learning_rate=self.fp_learning_rate,
                                               num_layers=None,
                                               hidden_size=None,
                                               b_size=self.b_size,
                                               saving=True,
                                               criterion=nn.MSELoss(),
                                               save_path=self.model_path,
                                               augmented=False,
                                               fp_known=self.fp_known,
                                               fp_with_constant=self.fp_constant,
                                               train_mean_X=None,
                                               train_std_X=None,
                                               train_mean_y=None,
                                               train_std_y=None,
                                               seed=self.seed)

            # WARNING: The augmented model will rely on the fitted version of self.known_model
            # which is not fitted yet (fp_pretrained should be modified a posteriori)
            self.augmented_model = Tree_Model(problem=self.problem,
                                              term=self.fp_term,
                                              N=self.N,
                                              d=self.d,
                                              learning_rate_trees=self.learning_rate_trees,
                                              max_features=self.max_features,
                                              min_samples_split=self.min_samples_split,
                                              boosting=self.boosting,
                                              criterion=nn.MSELoss(),
                                              saving=self.saving,
                                              device='cpu',
                                              save_path=self.model_path,
                                              fa_inputs=self.fa_inputs,
                                              seed=self.seed,
                                              fp_pretrained=None)

            self.wandb_config = {'fp_learning_rate': self.fp_learning_rate,
                                 'incomplete': self.incomplete,
                                 'fp_inputs': self.fp_inputs,
                                 'fa_inputs': self.fa_inputs,
                                 'problem': self.problem,
                                 'problem_setting': self.problem_setting,
                                 'fp_term': self.fp_term,
                                 'b_size': self.b_size,
                                 'learning_rate_trees': self.learning_rate_trees,
                                 'max_features': self.max_features,
                                 'min_samples_split': self.min_samples_split,
                                 'boosting': self.boosting,
                                 'standardize': self.standardize,
                                 'exp_id': self.exp_id,
                                 'fp_only': self.fp_only,
                                 'fa_only': self.fa_only,
                                 'fa_mlp': self.fa_mlp,
                                 'fp_with_constant': self.fp_constant,
                                 'N': self.N,
                                 'max_d': self.d,
                                 'filter_fa': self.filter_fa,
                                 'epochs': self.n_iters}

        if self.saving or self.model_name in ['fp_with_constant_then_mlp', 'fp_with_constant_then_trees']:
            os.makedirs(self.model_path, exist_ok=True)
            os.makedirs(self.res_path, exist_ok=True)

    def load(self, model_path):

        if self.model_name == 'fp_known_then_mlp':
            self.augmented_model.load_parametric(os.path.join(model_path, 'known.pt'), augmented=False)
            self.augmented_model.load_parametric(os.path.join(model_path, 'augmented.pt'), augmented=True)
        elif self.model_name == 'fp_known_then_trees':
            self.known_model.load_parametric(os.path.join(model_path, 'known.pt'), augmented=False)
            self.augmented_model.fp = self.known_model.model.model_phy
            self.augmented_model.load_tree_model(os.path.join(model_path, 'augmented.pkl'))
        elif self.model_name == 'fp_known_only':
            self.known_model.load_parametric(os.path.join(model_path, 'known.pt'), augmented=False)
        elif self.model_name == 'mlp_only':
            self.augmented_model.load_parametric(os.path.join(model_path, 'augmented.pt'), augmented=True)
        elif self.model_name == 'trees_only':
            self.augmented_model.load_tree_model(os.path.join(model_path, 'augmented.pkl'))
        elif self.model_name == 'fp_only_without_constant':
            self.known_model.load_parametric(os.path.join(model_path, 'known.pt'), augmented=False)
        elif self.model_name == 'fp_only_with_constant':
            self.known_model.load_parametric(os.path.join(model_path, 'known.pt'), augmented=False)
        elif self.model_name == 'fp_with_constant_then_mlp':
            self.known_model.load_parametric(os.path.join(model_path, 'known.pt'), augmented=False)
            self.augmented_model.model.model_phy = self.known_model.model.model_phy
            self.augmented_model.load_parametric(os.path.join(model_path, 'augmented.pt'), augmented=True)
        elif self.model_name == 'fp_with_constant_then_trees':
            self.known_model.load_parametric(os.path.join(model_path, 'known.pt'), augmented=False)
            self.augmented_model.fp = self.known_model.model.model_phy
            self.augmented_model.load_tree_model(os.path.join(model_path, 'augmented.pkl'))

    def compute_param_error(self, params, rel=True):
        if self.known_model is None:
            return [-1, -1]

        else:
            errors = []
            if self.problem in ['friedman1', 'corr_friedman1']:
                if self.incomplete:
                    if self.fp_term == 'first':
                        if rel:
                            errors.append((abs(params['p0'] - self.known_model.model.model_phy.p0.item()) / params['p0']).item())
                            errors.append((abs(params['p1'] - self.known_model.model.model_phy.p1.item()) / params['p1']).item())
                        else:
                            errors.append(((params['p0'] - self.known_model.model.model_phy.p0.item()) ** 2).item())
                            errors.append(((params['p1'] - self.known_model.model.model_phy.p1.item()) ** 2).item())
                    elif self.fp_term == 'second':
                        if rel:
                            errors.append((abs(params['p2'] - self.known_model.model.model_phy.p2.item()) / params['p2']).item())
                            errors.append((abs(params['p3'] - self.known_model.model.model_phy.p3.item()) / params['p3']).item())
                        else:
                            errors.append(((params['p2'] - self.known_model.model.model_phy.p2.item()) ** 2).item())
                            errors.append(((params['p3'] - self.known_model.model.model_phy.p3.item()) ** 2).item())
                    elif self.fp_term == 'third':
                        if rel:
                            errors.append((abs(params['p4'] - self.known_model.model.model_phy.p4.item()) / params['p4']).item())
                            errors.append((abs(params['p5'] - self.known_model.model.model_phy.p5.item()) / params['p5']).item())
                        else:
                            errors.append(((params['p4'] - self.known_model.model.model_phy.p4.item()) ** 2).item())
                            errors.append(((params['p5'] - self.known_model.model.model_phy.p5.item()) ** 2).item())
                else:
                    if rel:
                        errors.append((abs(params['p0'] - self.known_model.model.model_phy.p0.item()) / params['p0']).item())
                        errors.append((abs(params['p1'] - self.known_model.model.model_phy.p1.item()) / params['p1']).item())
                        errors.append((abs(params['p2'] - self.known_model.model.model_phy.p2.item()) / params['p2']).item())
                        errors.append((abs(params['p3'] - self.known_model.model.model_phy.p3.item()) / params['p3']).item())
                        errors.append((abs(params['p4'] - self.known_model.model.model_phy.p4.item()) / params['p4']).item())
                        errors.append((abs(params['p5'] - self.known_model.model.model_phy.p5.item()) / params['p5']).item())
                    else:
                        errors.append(((params['p0'] - self.known_model.model.model_phy.p0.item()) ** 2).item())
                        errors.append(((params['p1'] - self.known_model.model.model_phy.p1.item()) ** 2).item())
                        errors.append(((params['p2'] - self.known_model.model.model_phy.p2.item()) ** 2).item())
                        errors.append(((params['p3'] - self.known_model.model.model_phy.p3.item()) ** 2).item())
                        errors.append(((params['p4'] - self.known_model.model.model_phy.p4.item()) ** 2).item())
                        errors.append(((params['p5'] - self.known_model.model.model_phy.p5.item()) ** 2).item())

            elif self.problem in ['linear_data', 'overlap_data']:
                if rel:
                    errors.append((abs(params['beta'] - self.known_model.model.model_phy.beta.item()) / abs(params['beta'])).item())
                else:
                    errors.append(((params['beta'] - self.known_model.model.model_phy.beta.item()) ** 2).item())

            return errors

    def train(self, train_x, train_y, valid_x, valid_y, wandb_logging):

        if wandb_logging:
            wandb.init(project="baselines",
                       entity="yannclaes",
                       config=self.wandb_config)
            wandb.define_metric('epoch')
            wandb.define_metric('*', step_metric='epoch')

        if self.fa_only:
            if self.fa_mlp:
                best_loss = self.augmented_model.train(train_x,
                                                       train_y,
                                                       valid_x,
                                                       valid_y,
                                                       wandb_logging)
            else:
                self.augmented_model.train(train_x, train_y)
                best_loss = self.augmented_model.validate(valid_x, valid_y)

        elif self.fp_only:
            best_loss = self.known_model.train(train_x,
                                               train_y,
                                               valid_x,
                                               valid_y,
                                               wandb_logging)

        else:
            _ = self.known_model.train(train_x,
                                       train_y,
                                       valid_x,
                                       valid_y,
                                       wandb_logging)

            # Load best model and train augmented model
            self.known_model.load_parametric(self.known_model.save_path, augmented=False)

            if self.fa_mlp:
                self.augmented_model.model.model_phy = self.known_model.model.model_phy
                best_loss = self.augmented_model.train(train_x,
                                                       train_y,
                                                       valid_x,
                                                       valid_y,
                                                       wandb_logging)
            else:
                self.augmented_model.fp = self.known_model.model.model_phy
                self.augmented_model.train(train_x, train_y)
                best_loss = self.augmented_model.validate(valid_x, valid_y)

        return best_loss

    def validate(self, valid_x, valid_y):

        if self.augmented_model is not None:
            return self.augmented_model.validate(valid_x, valid_y)
        else:
            return self.known_model.validate(valid_x, valid_y)

    def predict(self, test_x, test_y=None):

        if self.augmented_model is not None:
            pred_x = self.augmented_model.predict(test_x)
        else:
            pred_x = self.known_model.predict(test_x)

        torch.save(pred_x, os.path.join(self.res_path, 'predictions.pt'))
        torch.save(test_y, os.path.join(self.res_path, 'targets.pt'))
        return pred_x

