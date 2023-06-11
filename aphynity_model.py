import os
import math
import wandb
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.linalg import norm
from torch.optim.lr_scheduler import CosineAnnealingLR

from models import MLP, Forecaster
from utils import make_known_model, get_batch


class APHYNITY():

    def __init__(self, hyperparameter, lambda_loss, problem, problem_setting,
        fp_term, exp_id, fp_inputs, fa_inputs, out_size, saving, training=False,
        fp_warmup=False, fp_control=False, jointly=True, seed=None,
        train_mean_X=None, train_std_X=None, train_mean_y=None,
        train_std_y=None, filter_fa=False, extrapolation=False,
        criterion=nn.MSELoss(), device='cpu', n_iters=3000):

        self.incomplete = hyperparameter[0]
        self.lambda_loss = lambda_loss
        self.fp_learning_rate = hyperparameter[1]
        self.lambda_rate = hyperparameter[2]

        self.problem = problem
        self.problem_setting = problem_setting
        self.fp_term = fp_term
        self.exp_id = exp_id
        self.saving = saving
        self.training = training
        self.device = device
        self.criterion = criterion
        self.fp_inputs = fp_inputs
        self.fa_inputs = fa_inputs
        self.out_size = out_size
        self.train_mean_X = train_mean_X
        self.train_std_X = train_std_X
        self.train_mean_y = train_mean_y
        self.train_std_y = train_std_y
        self.filter_fa = filter_fa
        self.extrapolation = extrapolation

        self.standardize = self.train_mean_X is not None
        self.fp_warmup = fp_warmup
        self.fp_control = fp_control
        self.jointly = jointly
        self.seed = seed

        self.n_iters = n_iters
        self.num_layers = hyperparameter[3]
        self.hidden_size = hyperparameter[4]
        self.b_size = hyperparameter[5]
        self.mlp_learning_rate = hyperparameter[6]

        # Get problem config
        self.lambda_j = 1.0
        self.lambda_fp = 1.0
        self.geometric_coeff = math.exp((math.log(0.01) - math.log(self.lambda_fp)) / self.n_iters)

        if self.saving:

            # Initialize model and results paths
            MOD_OUTPUT_PREFIX = self.problem + '/' + self.problem_setting + '/' + self.fp_term + '/models/aphynity'
            RES_OUTPUT_PREFIX = self.problem + '/' + self.problem_setting + '/' + self.fp_term + '/results/aphynity'

            self.model_path = os.path.join(MOD_OUTPUT_PREFIX,
                                           'incomplete' if self.incomplete else 'complete',
                                           'fp_warmup' if self.fp_warmup else 'fp_random',
                                           'fp_control' if self.fp_control else 'fp_relax',
                                           'jointly' if self.jointly else 'alternating',
                                           'filter' if self.filter_fa else 'full',
                                           'extrapolation' if self.extrapolation else 'interpolation',
                                           'lambda_loss' if self.lambda_loss else 'mse_loss',
                                           str(self.lambda_rate),
                                           str(self.num_layers),
                                           str(self.hidden_size),
                                           str(self.fp_learning_rate),
                                           str(self.mlp_learning_rate),
                                           str(self.b_size),
                                           'exp_' + str(self.exp_id))

            self.res_path = os.path.join(RES_OUTPUT_PREFIX,
                                         'incomplete' if self.incomplete else 'complete',
                                         'fp_warmup' if self.fp_warmup else 'fp_random',
                                         'fp_control' if self.fp_control else 'fp_relax',
                                         'jointly' if self.jointly else 'alternating',
                                         'filter' if self.filter_fa else 'full',
                                         'extrapolation' if self.extrapolation else 'interpolation',
                                         'lambda_loss' if self.lambda_loss else 'mse_loss',
                                         str(self.lambda_rate),
                                         str(self.num_layers),
                                         str(self.hidden_size),
                                         str(self.fp_learning_rate),
                                         str(self.mlp_learning_rate),
                                         str(self.b_size),
                                         'exp_' + str(self.exp_id))

            # Create model and results directories
            os.makedirs(self.model_path, exist_ok=True)
            os.makedirs(self.res_path, exist_ok=True)

            self.model_path = os.path.join(self.model_path, 'augmented.pt')

        # Initialize models
        self.known_model = make_known_model(self.problem,
                                            self.fp_term,
                                            self.incomplete,
                                            constant=self.fp_warmup,
                                            real_params=None,
                                            seed=self.seed).to(self.device)

        self.augmented_model = MLP(in_size=len(self.fa_inputs),
                                   num_layers=self.num_layers,
                                   hidden_size=self.hidden_size,
                                   out_size=self.out_size,
                                   train_mean_X=None,
                                   train_std_X=None,
                                   train_mean_y=self.train_mean_y,
                                   train_std_y=self.train_std_y).to(self.device)

        self.model = Forecaster(self.known_model,
                                self.augmented_model,
                                is_augmented=True,
                                jointly=self.jointly)

        # Define optimizer and scheduler
        if self.jointly:
            self.optim = optim.Adam(self.model.parameters(), lr=self.fp_learning_rate)
            self.scheduler = CosineAnnealingLR(self.optim, T_max=self.n_iters)
        else:
            self.fp_optim = optim.Adam(self.known_model.parameters(), lr=self.fp_learning_rate)
            self.fp_scheduler = CosineAnnealingLR(self.fp_optim, T_max=self.n_iters)
            self.fa_optim = optim.Adam(self.augmented_model.parameters(), lr=self.mlp_learning_rate)
            self.fa_scheduler = CosineAnnealingLR(self.fa_optim, T_max=self.n_iters)
            self.train_fp = True

    def load(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def compute_fp_distance(self, train_x, train_y):
        with torch.no_grad():
            fp_pred = self.known_model(train_x).squeeze()

        # This definition of the norm is only valid given that y has a single dimension
        distance = nn.MSELoss()(fp_pred, train_y).item()
        return distance

    def compute_norm_fa(self, states):
        norm_arg = self.augmented_model.get_derivatives(states[:, self.fa_inputs])
        norm_fa = (norm(norm_arg) ** 2) / len(states)
        return norm_fa

    def compute_param_error(self, params, rel=True):
        errors = []
        if self.problem in ['friedman1', 'corr_friedman1']:
            if self.incomplete:
                if self.fp_term == 'first':
                    if rel:
                        errors.append((abs(params['p0'] - self.known_model.p0.item()) / params['p0']).item())
                        errors.append((abs(params['p1'] - self.known_model.p1.item()) / params['p1']).item())
                    else:
                        errors.append(((params['p0'] - self.known_model.p0.item()) ** 2).item())
                        errors.append(((params['p1'] - self.known_model.p1.item()) ** 2).item())
                elif self.fp_term == 'second':
                    if rel:
                        errors.append((abs(params['p2'] - self.known_model.p2.item()) / params['p2']).item())
                        errors.append((abs(params['p3'] - self.known_model.p3.item()) / params['p3']).item())
                    else:
                        errors.append(((params['p2'] - self.known_model.p2.item()) ** 2).item())
                        errors.append(((params['p3'] - self.known_model.p3.item()) ** 2).item())
                elif self.fp_term == 'third':
                    if rel:
                        errors.append((abs(params['p4'] - self.known_model.p4.item()) / params['p4']).item())
                        errors.append((abs(params['p5'] - self.known_model.p5.item()) / params['p5']).item())
                    else:
                        errors.append(((params['p4'] - self.known_model.p4.item()) ** 2).item())
                        errors.append(((params['p5'] - self.known_model.p5.item()) ** 2).item())
            else:
                if rel:
                    errors.append((abs(params['p0'] - self.known_model.p0.item()) / params['p0']).item())
                    errors.append((abs(params['p1'] - self.known_model.p1.item()) / params['p1']).item())
                    errors.append((abs(params['p2'] - self.known_model.p2.item()) / params['p2']).item())
                    errors.append((abs(params['p3'] - self.known_model.p3.item()) / params['p3']).item())
                    errors.append((abs(params['p4'] - self.known_model.p4.item()) / params['p4']).item())
                    errors.append((abs(params['p5'] - self.known_model.p5.item()) / params['p5']).item())
                else:
                    errors.append(((params['p0'] - self.known_model.p0.item()) ** 2).item())
                    errors.append(((params['p1'] - self.known_model.p1.item()) ** 2).item())
                    errors.append(((params['p2'] - self.known_model.p2.item()) ** 2).item())
                    errors.append(((params['p3'] - self.known_model.p3.item()) ** 2).item())
                    errors.append(((params['p4'] - self.known_model.p4.item()) ** 2).item())
                    errors.append(((params['p5'] - self.known_model.p5.item()) ** 2).item())

        elif self.problem in ['linear_data', 'overlap_data']:
            if rel:
                errors.append((abs(params['beta'] - self.known_model.beta.item()) / abs(params['beta'])).item())
            else:
                errors.append(((params['beta'] - self.known_model.beta.item()) ** 2).item())

        return errors

    def fp_warmup_forward_epoch(self, X, y, train=False):

        n_batches = int(np.ceil(X.shape[0] / self.b_size))
        permutation = torch.randperm(X.shape[0])
        total_loss, total_mse, = 0., 0.
        for index in range(0, X.shape[0], self.b_size):

            batch_x, batch_y = get_batch(X,
                                         y,
                                         permutation[index:index+self.b_size],
                                         self.device)

            pred_x = self.known_model(batch_x).to(self.device)

            loss = self.criterion(batch_y, pred_x)

            if train:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                if self.problem in ['friedman1', 'corr_friedman1']:
                    w = self.model.model_phy.p3.data
                    w = w.clamp(0, 1)
                    self.model.model_phy.p3.data = w

            total_loss += loss.item() / n_batches
            total_mse += loss.item() / n_batches

        return total_loss, total_mse

    def forward_epoch(self, X, y, train=False):

        n_batches = int(np.ceil(X.shape[0] / self.b_size))
        permutation = torch.randperm(X.shape[0])
        total_loss, total_mse, = 0., 0.
        for index in range(0, X.shape[0], self.b_size):

            batch_x, batch_y = get_batch(X,
                                         y,
                                         permutation[index:index+self.b_size],
                                         self.device)

            if self.jointly:
                pred_x = self.model(batch_x, batch_x[:, self.fa_inputs]).to(self.device)
                pred_x_phy = self.model.model_phy(batch_x).to(self.device)
            else:
                pred_x = self.model(batch_x, batch_x[:, self.fa_inputs], train_fp=self.train_fp).to(self.device)
                if self.train_fp:
                    pred_x_phy = self.model.model_phy(batch_x).to(self.device)
                else:
                    with torch.no_grad():
                        pred_x_phy = self.model.model_phy(batch_x).to(self.device)

            loss_traj = self.criterion(batch_y, pred_x)
            if self.lambda_loss:
                norm_arg = self.model.model_aug.get_derivatives(batch_x[:, self.fa_inputs])
                loss_fa = (norm(norm_arg) ** 2) / len(batch_x)

                if self.fp_control:
                    loss_fp = self.lambda_fp * self.criterion(batch_y, pred_x_phy)
                    loss = self.lambda_j * loss_traj + loss_fa + loss_fp
                else:
                    loss = self.lambda_j * loss_traj + loss_fa
            else:
                if self.fp_control:
                    loss_fp = self.lambda_fp * self.criterion(batch_y, pred_x_phy)
                    loss = loss_traj + loss_fp
                else:
                    loss = loss_traj

            if train:
                if self.jointly:
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                else:
                    if self.train_fp:
                        self.fp_optim.zero_grad()
                        loss.backward()
                        self.fp_optim.step()
                    else:
                        self.fa_optim.zero_grad()
                        loss.backward()
                        self.fa_optim.step()

                    self.train_fp = not self.train_fp

                if self.problem in ['friedman1', 'corr_friedman1']:
                    w = self.model.model_phy.p3.data
                    w = w.clamp(0, 1)
                    self.model.model_phy.p3.data = w

            total_loss += loss.item() / n_batches
            total_mse += loss_traj.item() / n_batches

        return total_loss, total_mse

    def train(self, train_x, train_y, valid_x, valid_y, wandb_logging):

        # If training mode and wandb_logging, initialize the corresponding run
        if self.training:
            if wandb_logging:
                config = {"problem": self.problem,
                          "problem_setting": self.problem_setting,
                          "fp_term": self.fp_term,
                          "incomplete": self.incomplete,
                          "fp_inputs": self.fp_inputs,
                          "fa_inputs": self.fa_inputs,
                          "lambda_loss": self.lambda_loss,
                          "lambda_rate": self.lambda_rate,
                          "fp_learning_rate": self.fp_learning_rate,
                          "mlp_learning_rate": self.mlp_learning_rate,
                          "num_layers": self.num_layers,
                          "hidden_size": self.hidden_size,
                          "b_size": self.b_size,
                          "standardize": self.standardize,
                          "fp_warmup": self.fp_warmup,
                          "fp_control": self.fp_control,
                          "jointly": self.jointly,
                          "filter_fa": self.filter_fa,
                          "exp_id": self.exp_id}

                wandb.init(project="aphynity", entity="yannclaes", config=config)
                wandb.define_metric('fp_warmup/epoch')
                wandb.define_metric('fp_warmup/*', step_metric='fp_warmup/epoch')
                wandb.define_metric('train/epoch')
                wandb.define_metric('train/*', step_metric='train/epoch')

        best_loss = np.inf

        # Warmup phase
        if self.fp_warmup:

            self.optim = optim.Adam(self.model.model_phy.parameters(), lr=self.fp_learning_rate)
            self.scheduler = CosineAnnealingLR(self.optim, T_max=self.n_iters)

            for n in range(self.n_iters):
                tr_loss, tr_mse = self.fp_warmup_forward_epoch(train_x, train_y, train=True)
                self.scheduler.step()

                with torch.no_grad():

                    # Loss of trained model on whole training set
                    tr_loss, tr_mse = self.fp_warmup_forward_epoch(train_x, train_y)
                    val_loss, val_mse = self.fp_warmup_forward_epoch(valid_x, valid_y)

                    # Save model
                    if val_mse < best_loss:
                        if self.saving:
                            torch.save(self.model.state_dict(), self.model_path)
                        best_loss = val_mse

                        if wandb_logging:
                            wandb.run.summary['best fp_warmup val_mse'] = val_mse
                            wandb.run.summary['best fp_warmup epoch'] = n+1

                    # Compute training and validation norm
                    tr_norm_fa = self.compute_norm_fa(train_x)
                    val_norm_fa = self.compute_norm_fa(valid_x)

                    # Compute distance(fp, f)
                    distance = self.compute_fp_distance(train_x, train_y)

                    if wandb_logging:
                        log_dict = {f"fp_warmup/train_loss": tr_loss,
                                    f"fp_warmup/valid_loss": val_loss,
                                    f"fp_warmup/train_mse": tr_mse,
                                    f"fp_warmup/valid_mse": val_mse,
                                    f"fp_warmup/learning rate": self.optim.param_groups[0]['lr'],
                                    f"fp_warmup/tr_norm_fa": tr_norm_fa,
                                    f"fp_warmup/val_norm_fa": val_norm_fa,
                                    f"fp_warmup/d(fp, f)": distance,
                                    f"fp_warmup/epoch": n+1}

                        if self.problem in ['friedman1', 'corr_friedman1', 'linear_data', 'overlap_data']:
                            params = []
                            if self.problem in ['friedman1', 'corr_friedman1']:
                                if self.incomplete:
                                    if self.fp_term == 'first':
                                        params_name = [f'fp_warmup/p0', f'fp_warmup/p1']
                                        params.append(self.model.model_phy.p0.item())
                                        params.append(self.model.model_phy.p1.item())
                                    elif self.fp_term == 'second':
                                        params_name = [f'fp_warmup/p2', f'fp_warmup/p3']
                                        params.append(self.model.model_phy.p2.item())
                                        params.append(self.model.model_phy.p3.item())
                                    elif self.fp_term == 'third':
                                        params_name = [f'fp_warmup/p4', f'fp_warmup/p5']
                                        params.append(self.model.model_phy.p4.item())
                                        params.append(self.model.model_phy.p5.item())
                                else:
                                    params_name = [f'fp_warmup/p0', f'fp_warmup/p1', f'fp_warmup/p2', f'fp_warmup/p3', f'fp_warmup/p4', f'fp_warmup/p5']
                                    params.append(self.model.model_phy.p0.item())
                                    params.append(self.model.model_phy.p1.item())
                                    params.append(self.model.model_phy.p2.item())
                                    params.append(self.model.model_phy.p3.item())
                                    params.append(self.model.model_phy.p4.item())
                                    params.append(self.model.model_phy.p5.item())

                            elif self.problem in ['linear_data', 'overlap_data']:
                                params.append(self.model.model_phy.beta.item())
                                params_name = [f'fp_warmup/beta']

                            param_dict = {}
                            for param, name in zip(params, params_name):
                                param_dict[name] = param
                            log_dict.update(param_dict)

                        wandb.log(log_dict)

                    print('Finished fp_warmup epoch {:04d}/{:04d} | Train loss : {:.6f} | Validation loss: {:.6f}'.format(n+1, self.n_iters, tr_loss, val_loss))

            # Load best Fp model
            self.model.load_state_dict(torch.load(self.model_path))

            # Re-initialize optimizer and scheduler
            if self.jointly:
                self.optim = optim.Adam(self.model.parameters(), lr=self.fp_learning_rate)
                self.scheduler = CosineAnnealingLR(self.optim, T_max=self.n_iters)
            best_loss = np.inf

        for n in range(self.n_iters):

            # Training phase
            tr_loss, tr_mse = self.forward_epoch(train_x, train_y, train=True)
            if self.jointly:
                self.scheduler.step()
            else:
                self.fp_scheduler.step()
                self.fa_scheduler.step()

            with torch.no_grad():

                # Loss of trained model on whole training set
                tr_loss, tr_mse = self.forward_epoch(train_x, train_y)
                val_loss, val_mse = self.forward_epoch(valid_x, valid_y)

                # Save model
                if val_mse < best_loss:
                    if self.saving:
                        torch.save(self.model.state_dict(), self.model_path)
                    best_loss = val_mse

                    if wandb_logging:
                        wandb.run.summary['best val_mse'] = val_mse
                        wandb.run.summary['best epoch'] = n+1

                # Compute training and validation norm
                tr_norm_fa = self.compute_norm_fa(train_x)
                val_norm_fa = self.compute_norm_fa(valid_x)

                # Compute distance(fp, f)
                distance = self.compute_fp_distance(train_x, train_y)

                if wandb_logging:
                    if self.jointly:
                        log_dict = {f"train/train_loss": tr_loss,
                                    f"train/valid_loss": val_loss,
                                    f"train/train_mse": tr_mse,
                                    f"train/valid_mse": val_mse,
                                    f"train/learning rate": self.optim.param_groups[0]['lr'],
                                    f"train/tr_norm_fa": tr_norm_fa,
                                    f"train/val_norm_fa": val_norm_fa,
                                    f"train/d(fp, f)": distance,
                                    f"train/lambda": self.lambda_j,
                                    f"train/lambda_fp": self.lambda_fp,
                                    f"train/epoch": n+1}
                    else:
                        log_dict = {f"train/train_loss": tr_loss,
                                    f"train/valid_loss": val_loss,
                                    f"train/train_mse": tr_mse,
                                    f"train/valid_mse": val_mse,
                                    f"train/fp_learning rate": self.fp_optim.param_groups[0]['lr'],
                                    f"train/fa_learning rate": self.fa_optim.param_groups[0]['lr'],
                                    f"train/tr_norm_fa": tr_norm_fa,
                                    f"train/val_norm_fa": val_norm_fa,
                                    f"train/d(fp, f)": distance,
                                    f"train/lambda": self.lambda_j,
                                    f"train/lambda_fp": self.lambda_fp,
                                    f"train/epoch": n+1}

                    if self.problem in ['friedman1', 'corr_friedman1', 'linear_data', 'overlap_data']:
                        params = []
                        if self.problem in ['friedman1', 'corr_friedman1']:
                            if self.incomplete:
                                if self.fp_term == 'first':
                                    params_name = [f'train/p0', f'train/p1']
                                    params.append(self.model.model_phy.p0.item())
                                    params.append(self.model.model_phy.p1.item())
                                elif self.fp_term == 'second':
                                    params_name = [f'train/p2', f'train/p3']
                                    params.append(self.model.model_phy.p2.item())
                                    params.append(self.model.model_phy.p3.item())
                                elif self.fp_term == 'third':
                                    params_name = [f'train/p4', f'train/p5']
                                    params.append(self.model.model_phy.p4.item())
                                    params.append(self.model.model_phy.p5.item())
                            else:
                                params_name = [f'train/p0', f'train/p1', f'train/p2', f'train/p3', f'train/p4', f'train/p5']
                                params.append(self.model.model_phy.p0.item())
                                params.append(self.model.model_phy.p1.item())
                                params.append(self.model.model_phy.p2.item())
                                params.append(self.model.model_phy.p3.item())
                                params.append(self.model.model_phy.p4.item())
                                params.append(self.model.model_phy.p5.item())

                        elif self.problem in ['linear_data', 'overlap_data']:
                            params.append(self.model.model_phy.beta.item())
                            params_name = [f'train/beta']

                        param_dict = {}
                        for param, name in zip(params, params_name):
                            param_dict[name] = param
                        log_dict.update(param_dict)

                    wandb.log(log_dict)

            print('Finished epoch {:04d}/{:04d} | Train loss : {:.6f} | Validation loss: {:.6f}'.format(n+1, self.n_iters, tr_loss, val_loss))

            # Update lambda
            if self.lambda_loss:
                self.lambda_j += self.lambda_rate * tr_mse
            if self.fp_control:
                self.lambda_fp *= self.geometric_coeff

        return best_loss

    def validate(self, valid_x, valid_y):

        # Validation phase
        val_loss, val_mse = self.forward_epoch(valid_x, valid_y)
        return val_mse

    def predict(self, test_x):

        pred_x = self.model(test_x, test_x[:, self.fa_inputs]).to(self.device)
        torch.save(pred_x, os.path.join(self.res_path, 'predictions.pt'))
        torch.save(test_y, os.path.join(self.res_path, 'targets.pt'))
