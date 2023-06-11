import os
import math
import wandb
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.linalg import norm
from xgboost import XGBRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from utils import make_known_model, mse_objective, save_model, load_model


class GenericTrees():

    def __init__(self, N, d, learning_rate_trees, incomplete, max_features,
        min_samples_split, lambda_loss, problem, problem_setting, fp_term,
        boosting=True, fp_warmup=True, fp_control=False, fp_inputs=[0, 1],
        fa_inputs=[2, 3, 4, 5, 6, 7, 8, 9], seed=None):

        self.N = N
        self.d = d
        self.learning_rate_trees = learning_rate_trees
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.lambda_loss = lambda_loss
        self.problem = problem
        self.problem_setting = problem_setting
        self.fp_term = fp_term
        self.incomplete = incomplete
        self.boosting = boosting
        self.fp_warmup = fp_warmup
        self.fp_control = fp_control
        self.fp_inputs = fp_inputs
        self.fa_inputs = fa_inputs
        self.seed = seed

        # Initialize problem configuration
        self.lambda_j = 1
        self.lambda_j_init = 1
        self.lambda_fp_init = 0.1
        self.lambda_fp = 0.1

        # Initialize physical and augmented models
        self.model_phy = make_known_model(self.problem,
                                          self.fp_term,
                                          self.incomplete,
                                          constant=self.fp_warmup,
                                          real_params=None,
                                          seed=self.seed)
        self.model_aug = DummyRegressor(strategy='constant', constant=0)

    def compute_fp_distance(self, train_x, train_y):
        with torch.no_grad():
            fp_pred = self.model_phy(train_x).squeeze()

        # This definition of the norm is only valid given that y has a single dimension
        distance = nn.MSELoss()(fp_pred, train_y).item()
        return distance

    def compute_param_error(self, params, rel=True):
        errors = []
        if self.problem in ['friedman1', 'corr_friedman1']:
            if self.incomplete:
                if self.fp_term == 'first':
                    if rel:
                        errors.append((abs(params['p0'] - self.model_phy.p0.item()) / params['p0']).item())
                        errors.append((abs(params['p1'] - self.model_phy.p1.item()) / params['p1']).item())
                    else:
                        errors.append(((params['p0'] - self.model_phy.p0.item()) ** 2).item())
                        errors.append(((params['p1'] - self.model_phy.p1.item()) ** 2).item())
                elif self.fp_term == 'second':
                    if rel:
                        errors.append((abs(params['p2'] - self.model_phy.p2.item()) / params['p2']).item())
                        errors.append((abs(params['p3'] - self.model_phy.p3.item()) / params['p3']).item())
                    else:
                        errors.append(((params['p2'] - self.model_phy.p2.item()) ** 2).item())
                        errors.append(((params['p3'] - self.model_phy.p3.item()) ** 2).item())
                elif self.fp_term == 'third':
                    if rel:
                        errors.append((abs(params['p4'] - self.model_phy.p4.item()) / params['p4']).item())
                        errors.append((abs(params['p5'] - self.model_phy.p5.item()) / params['p5']).item())
                    else:
                        errors.append(((params['p4'] - self.model_phy.p4.item()) ** 2).item())
                        errors.append(((params['p5'] - self.model_phy.p5.item()) ** 2).item())
            else:
                if rel:
                    errors.append((abs(params['p0'] - self.model_phy.p0.item()) / params['p0']).item())
                    errors.append((abs(params['p1'] - self.model_phy.p1.item()) / params['p1']).item())
                    errors.append((abs(params['p2'] - self.model_phy.p2.item()) / params['p2']).item())
                    errors.append((abs(params['p3'] - self.model_phy.p3.item()) / params['p3']).item())
                    errors.append((abs(params['p4'] - self.model_phy.p4.item()) / params['p4']).item())
                    errors.append((abs(params['p5'] - self.model_phy.p5.item()) / params['p5']).item())
                else:
                    errors.append(((params['p0'] - self.model_phy.p0.item()) ** 2).item())
                    errors.append(((params['p1'] - self.model_phy.p1.item()) ** 2).item())
                    errors.append(((params['p2'] - self.model_phy.p2.item()) ** 2).item())
                    errors.append(((params['p3'] - self.model_phy.p3.item()) ** 2).item())
                    errors.append(((params['p4'] - self.model_phy.p4.item()) ** 2).item())
                    errors.append(((params['p5'] - self.model_phy.p5.item()) ** 2).item())

        elif self.problem in ['linear_data', 'overlap_data']:
            if rel:
                errors.append((abs(params['beta'] - self.model_phy.beta.item()) / abs(params['beta'])).item())
            else:
                errors.append(((params['beta'] - self.model_phy.beta.item()) ** 2).item())

        return errors

    def train_epoch(self, X, y, criterion, optim, train_phy,
        previous_preds=None):

        phy_term = torch.zeros(len(X))

        pred_x_phy = self.model_phy(X)

        if previous_preds is None:
            pred_x_aug = torch.from_numpy(self.model_aug.predict(X[:, self.fa_inputs])).float()
        else:
            pred_x_aug = previous_preds

        pred_x = pred_x_phy + pred_x_aug

        # Record physical term at time step n
        if previous_preds is None:
            phy_term = pred_x_phy
        else:
            phy_term = pred_x

        # Train physical part
        if train_phy:
            loss_traj = criterion(y, pred_x)

            if self.lambda_loss:
                loss_traj = self.lambda_j * loss_traj
                if self.fp_control:
                    loss_fp = self.lambda_fp * criterion(y, pred_x_phy)
                    loss = loss_traj + loss_fp
                else:
                    loss = loss_traj
            else:
                loss_traj = loss_traj
                if self.fp_control:
                    loss_fp = self.lambda_fp * criterion(y, pred_x_phy)
                    loss = loss_traj + loss_fp
                else:
                    loss = loss_traj

            optim.zero_grad()
            loss.backward()
            optim.step()

            if self.problem in ['friedman1', 'corr_friedman1']:
                w = self.model_phy.p3.data
                w = w.clamp(0, 1)
                self.model_phy.p3.data = w

        # Train augmented part
        else:
            if self.boosting:
                _model_aug = XGBRegressor(n_estimators=self.N,
                                          max_depth=self.d,
                                          learning_rate=self.learning_rate_trees,
                                          n_jobs=-1,
                                          random_state=self.seed,
                                          objective=mse_objective(phy_term.detach().cpu().numpy(),
                                                                  self.lambda_j,
                                                                  self.lambda_loss))
                _model_aug.fit(X[:, self.fa_inputs], y)

            else:
                _model_aug = RandomForestRegressor(n_estimators=self.N,
                                                   max_features=self.max_features,
                                                   min_samples_split=self.min_samples_split,
                                                   random_state=self.seed,
                                                   n_jobs=-1)

                _model_aug.fit(X[:, self.fa_inputs], y - phy_term.detach())


        # Return new augmented model (only defined when train_phy is False)
        if not train_phy:
            return _model_aug
        else:
            return

    def valid_epoch(self, X, y, criterion, previous_preds=None,
        incrementrees=False):

        with torch.no_grad():

            pred_x_phy = self.model_phy(X)

            if previous_preds is None:
                if incrementrees:
                    pred_x_aug = torch.zeros(len(X))
                    for m in self.model_aug:
                        pred_x_aug += torch.from_numpy(m.predict(X[:, self.fa_inputs])).float()
                else:
                    pred_x_aug = torch.from_numpy(self.model_aug.predict(X[:, self.fa_inputs])).float()
            else:
                pred_x_aug = previous_preds

            pred_x = pred_x_phy + pred_x_aug

            loss_traj = criterion(y, pred_x)
            loss_fa = (norm(pred_x_aug) ** 2) / len(pred_x_aug)
            loss_fp = self.lambda_fp * criterion(y, pred_x_phy)

            # Lambda loss setting will not be considered with Random Forests
            if self.lambda_loss:
                if self.fp_control:
                    loss = self.lambda_j * loss_traj + loss_fa + loss_fp
                else:
                    loss = self.lambda_j * loss_traj + loss_fa
            else:
                if self.fp_control:
                    loss = loss_traj + loss_fp
                else:
                    loss = loss_traj

        return loss.item(), loss_traj.item()

    def test_epoch(self, X, y, res_path, incrementrees=False):

        with torch.no_grad():

            pred_x_phy = self.model_phy(X)
            if incrementrees:
                pred_x_aug = torch.zeros(len(X))
                for m in self.model_aug:
                    pred_x_aug += torch.from_numpy(m.predict(X[:, self.fa_inputs])).float()
            else:
                pred_x_aug = torch.from_numpy(self.model_aug.predict(X[:, self.fa_inputs])).float()

            pred_x = pred_x_phy + pred_x_aug

        return pred_x

class APHYNITrees(GenericTrees):

    def __init__(self, hyperparameter, lambda_loss, problem, problem_setting,
        fp_term, boosting, exp_id, saving, training=False, fp_warmup=False,
        fp_control=False, fp_inputs=[0, 1], fa_inputs=[2, 3, 4, 5, 6, 7, 8, 9],
        filter_fa=False, extrapolation=False, seed=None, n_iters=3000):

        super().__init__(hyperparameter[0],
                         hyperparameter[1],
                         hyperparameter[5],
                         hyperparameter[2],
                         hyperparameter[6],
                         hyperparameter[7],
                         lambda_loss,
                         problem,
                         problem_setting,
                         fp_term,
                         boosting,
                         fp_warmup,
                         fp_control,
                         fp_inputs,
                         fa_inputs,
                         seed)
        self.train_aug = hyperparameter[3]
        self.learning_rate = hyperparameter[4]
        self.lambda_rate = hyperparameter[8]
        self.filter_fa = filter_fa
        self.extrapolation = extrapolation

        self.exp_id = exp_id
        self.saving = saving
        self.training = training
        self.n_iters = n_iters

        if self.lambda_fp_init == 0:
            self.geometric_coeff_fp = 1.
        else:
            self.geometric_coeff_fp = math.exp((math.log(0.001) - math.log(self.lambda_fp)) / self.n_iters)

        if self.saving:

            # Initialize model and results paths
            MOD_PREFIX = self.problem + '/' + self.problem_setting + '/' + self.fp_term + '/models/aphynitrees'
            RES_PREFIX = self.problem + '/' + self.problem_setting + '/' + self.fp_term + '/results/aphynitrees'
            if self.incomplete:
                if self.boosting:
                    self.model_path = os.path.join(MOD_PREFIX,
                                                   'incomplete',
                                                   'fp_warmup' if self.fp_warmup else 'fp_random',
                                                   'fp_control' if self.fp_control else 'fp_relax',
                                                   'xgboost',
                                                   'filter' if self.filter_fa else 'full',
                                                   'extrapolation' if self.extrapolation else 'interpolation',
                                                   str(self.N),
                                                   str(self.d),
                                                   str(self.train_aug),
                                                   str(self.learning_rate),
                                                   str(self.learning_rate_trees))
                    self.res_path = os.path.join(RES_PREFIX,
                                                 'incomplete',
                                                 'fp_warmup' if self.fp_warmup else 'fp_random',
                                                 'fp_control' if self.fp_control else 'fp_relax',
                                                 'xgboost',
                                                 'filter' if self.filter_fa else 'full',
                                                 'extrapolation' if self.extrapolation else 'interpolation',
                                                 str(self.N),
                                                 str(self.d),
                                                 str(self.train_aug),
                                                 str(self.learning_rate),
                                                 str(self.learning_rate_trees))
                else:
                    self.model_path = os.path.join(MOD_PREFIX,
                                                   'incomplete',
                                                   'fp_warmup' if self.fp_warmup else 'fp_random',
                                                   'fp_control' if self.fp_control else 'fp_relax',
                                                   'rf',
                                                   'filter' if self.filter_fa else 'full',
                                                   'extrapolation' if self.extrapolation else 'interpolation',
                                                   str(self.N),
                                                   str(self.max_features),
                                                   str(self.min_samples_split),
                                                   str(self.train_aug),
                                                   str(self.learning_rate))
                    self.res_path = os.path.join(RES_PREFIX,
                                                 'incomplete',
                                                 'fp_warmup' if self.fp_warmup else 'fp_random',
                                                 'fp_control' if self.fp_control else 'fp_relax',
                                                 'rf',
                                                 'filter' if self.filter_fa else 'full',
                                                 'extrapolation' if self.extrapolation else 'interpolation',
                                                 str(self.N),
                                                 str(self.max_features),
                                                 str(self.min_samples_split),
                                                 str(self.train_aug),
                                                 str(self.learning_rate))
            else:
                if self.boosting:
                    self.model_path = os.path.join(MOD_PREFIX,
                                                   'complete',
                                                   'fp_warmup' if self.fp_warmup else 'fp_random',
                                                   'fp_control' if self.fp_control else 'fp_relax',
                                                   'xgboost',
                                                   'filter' if self.filter_fa else 'full',
                                                   'extrapolation' if self.extrapolation else 'interpolation',
                                                   str(self.N),
                                                   str(self.d),
                                                   str(self.train_aug),
                                                   str(self.learning_rate),
                                                   str(self.learning_rate_trees))
                    self.res_path = os.path.join(RES_PREFIX,
                                                 'complete',
                                                 'fp_warmup' if self.fp_warmup else 'fp_random',
                                                 'fp_control' if self.fp_control else 'fp_relax',
                                                 'xgboost',
                                                 'filter' if self.filter_fa else 'full',
                                                 'extrapolation' if self.extrapolation else 'interpolation',
                                                 str(self.N),
                                                 str(self.d),
                                                 str(self.train_aug),
                                                 str(self.learning_rate),
                                                 str(self.learning_rate_trees))
                else:
                    self.model_path = os.path.join(MOD_PREFIX,
                                                   'complete',
                                                   'fp_warmup' if self.fp_warmup else 'fp_random',
                                                   'fp_control' if self.fp_control else 'fp_relax',
                                                   'rf',
                                                   'filter' if self.filter_fa else 'full',
                                                   'extrapolation' if self.extrapolation else 'interpolation',
                                                   str(self.N),
                                                   str(self.max_features),
                                                   str(self.min_samples_split),
                                                   str(self.train_aug),
                                                   str(self.learning_rate))
                    self.res_path = os.path.join(RES_PREFIX,
                                                 'complete',
                                                 'fp_warmup' if self.fp_warmup else 'fp_random',
                                                 'fp_control' if self.fp_control else 'fp_relax',
                                                 'rf',
                                                 'filter' if self.filter_fa else 'full',
                                                 'extrapolation' if self.extrapolation else 'interpolation',
                                                 str(self.N),
                                                 str(self.max_features),
                                                 str(self.min_samples_split),
                                                 str(self.train_aug),
                                                 str(self.learning_rate))

            if self.lambda_loss:
                self.model_path = os.path.join(self.model_path,
                                               'lambda_loss',
                                               str(self.lambda_rate))
                self.res_path = os.path.join(self.res_path,
                                             'lambda_loss',
                                             str(self.lambda_rate))
            else:
                self.model_path = os.path.join(self.model_path, 'mse_loss')
                self.res_path = os.path.join(self.res_path, 'mse_loss')

            # Add experiment id information
            self.model_path = os.path.join(self.model_path, 'exp_' + str(self.exp_id))
            self.res_path = os.path.join(self.res_path, 'exp_' + str(self.exp_id))

            self.model_phy_path = os.path.join(self.model_path, 'physical.pt')
            self.model_aug_path = self.model_path

            # Create model and results directories
            os.makedirs(self.model_path, exist_ok=True)
            os.makedirs(self.res_path, exist_ok=True)

    def compute_norm_fa(self, states):
        norm_arg = torch.from_numpy(self.model_aug.predict(states[:, self.fa_inputs])).float()
        norm_fa = 1/len(states) * norm(norm_arg) ** 2
        return norm_fa

    def load(self, model_phy_path, model_aug_path):
        self.model_phy.load_state_dict(torch.load(model_phy_path))
        self.model_aug = load_model(os.path.join(model_aug_path, 'augmented.pkl'))

    def train(self, train_x, train_y, valid_x, valid_y, wandb_logging):

        # If training mode and wandb_logging, initialize the corresponding run
        if self.training:
            if wandb_logging:
                wandb_config = {"problem": self.problem,
                                "problem_setting": self.problem_setting,
                                "fp_term": self.fp_term,
                                "incomplete": self.incomplete,
                                "fp_inputs": self.fp_inputs,
                                "fa_inputs": self.fa_inputs,
                                "lambda_loss": self.lambda_loss,
                                "boosting": self.boosting,
                                "fp_warmup": self.fp_warmup,
                                "fp_control": self.fp_control,
                                "N": self.N,
                                "max_d": self.d,
                                "train_aug": self.train_aug,
                                "learning_rate": self.learning_rate,
                                "learning_rate_trees": self.learning_rate_trees,
                                "max_features": self.max_features,
                                "min_samples_split": self.min_samples_split,
                                "lambda_rate": self.lambda_rate,
                                "filter_fa": self.filter_fa,
                                "exp_id": self.exp_id}

                wandb.init(project="aphynitrees",
                           entity="yannclaes",
                           config=wandb_config)
                wandb.define_metric("train/epoch")
                wandb.define_metric("train/*", step_metric="train/epoch")
                wandb.define_metric("fp_warmup/epoch")
                wandb.define_metric("fp_warmup/*", step_metric="fp_warmup/epoch")

        # Initialize models
        self.model_aug.fit(train_x[:, self.fa_inputs], train_y)

        # Initialize optimizer
        optimizer = optim.Adam(self.model_phy.parameters(),
                               lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=int((1 - 1/self.train_aug) * self.n_iters))
        mse = nn.MSELoss()
        best_loss = np.inf

        # Compute loss and logMSE on 0-th epoch
        tr_loss, tr_mse = self.valid_epoch(train_x, train_y, mse)
        val_loss, val_mse = self.valid_epoch(valid_x, valid_y, mse)

        # Compute ||F_a|| on train and valid datasets
        tr_norm_fa = self.compute_norm_fa(train_x)
        val_norm_fa = self.compute_norm_fa(valid_x)

        # Compute distance(fp, f)
        distance = self.compute_fp_distance(train_x, train_y)

        print('Finished init epoch | Train loss : {:.6f} | Validation loss: {:.6f}'.format(tr_loss, val_loss))

        if wandb_logging:
            log_dict = {f"fp_warmup/train_loss_{self.exp_id}": tr_loss,
                        f"fp_warmup/train_mse_{self.exp_id}": tr_mse,
                        f"fp_warmup/valid_mse_{self.exp_id}": val_mse,
                        f"fp_warmup/valid_loss_{self.exp_id}": val_loss,
                        f"fp_warmup/learning rate_{self.exp_id}": optimizer.param_groups[0]['lr'],
                        f"fp_warmup/d(fp, f)": distance,
                        f"fp_warmup/epoch": 0}

            if self.problem in ['friedman1', 'corr_friedman1', 'linear_data', 'overlap_data']:
                params = []
                if self.problem in ['friedman1', 'corr_friedman1']:
                    if self.incomplete:
                        if self.fp_term == 'first':
                            params_name = [f'fp_warmup/p0', f'fp_warmup/p1']
                            params.append(self.model_phy.p0.item())
                            params.append(self.model_phy.p1.item())
                        elif self.fp_term == 'second':
                            params_name = [f'fp_warmup/p2', f'fp_warmup/p3']
                            params.append(self.model_phy.p2.item())
                            params.append(self.model_phy.p3.item())
                        elif self.fp_term == 'third':
                            params_name = [f'fp_warmup/p4', f'fp_warmup/p5']
                            params.append(self.model_phy.p4.item())
                            params.append(self.model_phy.p5.item())
                    else:
                        params_name = [f'fp_warmup/p0', f'fp_warmup/p1', f'fp_warmup/p2', f'fp_warmup/p3', f'fp_warmup/p4', f'fp_warmup/p5']
                        params.append(self.model_phy.p0.item())
                        params.append(self.model_phy.p1.item())
                        params.append(self.model_phy.p2.item())
                        params.append(self.model_phy.p3.item())
                        params.append(self.model_phy.p4.item())
                        params.append(self.model_phy.p5.item())

                elif self.problem in ['linear_data', 'overlap_data']:
                    params.append(self.model_phy.beta.item())
                    params_name = [f'fp_warmup/beta']

                param_dict = {}
                for param, name in zip(params, params_name):
                    param_dict[name] = param
                log_dict.update(param_dict)


            wandb.log(log_dict)

        # Warmup phase
        if self.fp_warmup:

            self.lambda_fp = 0.
            self.lambda_j = 1.
            scheduler = CosineAnnealingLR(optimizer, T_max=self.n_iters)

            train_phy = True
            for n in range(self.n_iters):
                _model_aug = self.train_epoch(train_x,
                                              train_y,
                                              mse,
                                              optimizer,
                                              train_phy)
                scheduler.step()

                # Loss of trained model on whole training set
                tr_loss, tr_mse = self.valid_epoch(train_x, train_y, mse)
                val_loss, val_mse = self.valid_epoch(valid_x, valid_y, mse)

                # Compute distance(fp, f)
                distance = self.compute_fp_distance(train_x, train_y)

                # Save model
                if val_mse < best_loss:
                    if self.saving:
                        torch.save(self.model_phy.state_dict(), self.model_phy_path)
                    best_loss = val_mse

                    if wandb_logging:
                        wandb.run.summary['best fp_warmup val_mse'] = val_mse
                        wandb.run.summary['best fp_warmup epoch'] = n+1

                if wandb_logging:
                    log_dict = {f"fp_warmup/train_loss": tr_loss,
                                f"fp_warmup/train_mse": tr_mse,
                                f"fp_warmup/valid_mse": val_mse,
                                f"fp_warmup/valid_loss": val_loss,
                                f"fp_warmup/learning rate": optimizer.param_groups[0]['lr'],
                                f"fp_warmup/d(fp, f)": distance,
                                f"fp_warmup/epoch": n+1}

                    if self.problem in ['friedman1', 'corr_friedman1', 'linear_data', 'overlap_data']:
                        params = []
                        if self.problem in ['friedman1', 'corr_friedman1']:
                            if self.incomplete:
                                if self.fp_term == 'first':
                                    params_name = [f'fp_warmup/p0', f'fp_warmup/p1']
                                    params.append(self.model_phy.p0.item())
                                    params.append(self.model_phy.p1.item())
                                elif self.fp_term == 'second':
                                    params_name = [f'fp_warmup/p2', f'fp_warmup/p3']
                                    params.append(self.model_phy.p2.item())
                                    params.append(self.model_phy.p3.item())
                                elif self.fp_term == 'third':
                                    params_name = [f'fp_warmup/p4', f'fp_warmup/p5']
                                    params.append(self.model_phy.p4.item())
                                    params.append(self.model_phy.p5.item())
                            else:
                                params_name = [f'fp_warmup/p0', f'fp_warmup/p1', f'fp_warmup/p2', f'fp_warmup/p3', f'fp_warmup/p4', f'fp_warmup/p5']
                                params.append(self.model_phy.p0.item())
                                params.append(self.model_phy.p1.item())
                                params.append(self.model_phy.p2.item())
                                params.append(self.model_phy.p3.item())
                                params.append(self.model_phy.p4.item())
                                params.append(self.model_phy.p5.item())

                        elif self.problem in ['linear_data', 'overlap_data']:
                            params.append(self.model_phy.beta.item())
                            params_name = [f'fp_warmup/beta']

                        param_dict = {}
                        for param, name in zip(params, params_name):
                            param_dict[name] = param
                        log_dict.update(param_dict)

                    wandb.log(log_dict)

                print('Finished fp_warmup epoch {:04d}/{:04d} | Train loss : {:.6f} | Validation loss: {:.6f}'.format(n+1, self.n_iters, tr_loss, val_loss))

            # Load best Fp model
            self.model_phy.load_state_dict(torch.load(self.model_phy_path))

            # Re-initialize optimizer and scheduler
            optimizer = optim.Adam(self.model_phy.parameters(),
                                   lr=self.learning_rate)
            scheduler = CosineAnnealingLR(optimizer, T_max=int((1 - 1/self.train_aug) * self.n_iters))
            self.lambda_fp = self.lambda_fp_init
            self.lambda_j = self.lambda_j_init
            best_loss = np.inf

        # Compute loss and logMSE on 0-th epoch after warmup
        tr_loss, tr_mse = self.valid_epoch(train_x, train_y, mse)
        val_loss, val_mse = self.valid_epoch(valid_x, valid_y, mse)

        # Compute ||F_a|| on train and valid datasets
        tr_norm_fa = self.compute_norm_fa(train_x)
        val_norm_fa = self.compute_norm_fa(valid_x)

        # Compute distance(fp, f)
        distance = self.compute_fp_distance(train_x, train_y)

        print('Finished init epoch | Train loss : {:.6f} | Validation loss: {:.6f}'.format(tr_loss, val_loss))

        if wandb_logging:
            log_dict = {f"train/train_loss": tr_loss,
                        f"train/train_mse": tr_mse,
                        f"train/valid_mse": val_mse,
                        f"train/valid_loss": val_loss,
                        f"train/learning rate": optimizer.param_groups[0]['lr'],
                        f"train/d(fp, f)": distance,
                        f"train/epoch": 0}

            if self.problem in ['friedman1', 'corr_friedman1', 'linear_data', 'overlap_data']:
                params = []
                if self.problem in ['friedman1', 'corr_friedman1']:
                    if self.incomplete:
                        if self.fp_term == 'first':
                            params_name = [f'train/p0', f'train/p1']
                            params.append(self.model_phy.p0.item())
                            params.append(self.model_phy.p1.item())
                        elif self.fp_term == 'second':
                            params_name = [f'train/p2', f'train/p3']
                            params.append(self.model_phy.p2.item())
                            params.append(self.model_phy.p3.item())
                        elif self.fp_term == 'third':
                            params_name = [f'train/p4', f'train/p5']
                            params.append(self.model_phy.p4.item())
                            params.append(self.model_phy.p5.item())
                    else:
                        params_name = [f'train/p0', f'train/p1', f'train/p2', f'train/p3', f'train/p4', f'train/p5']
                        params.append(self.model_phy.p0.item())
                        params.append(self.model_phy.p1.item())
                        params.append(self.model_phy.p2.item())
                        params.append(self.model_phy.p3.item())
                        params.append(self.model_phy.p4.item())
                        params.append(self.model_phy.p5.item())

                elif self.problem in ['linear_data', 'overlap_data']:
                    params.append(self.model_phy.beta.item())
                    params_name = [f'train/beta']

                param_dict = {}
                for param, name in zip(params, params_name):
                    param_dict[name] = param
                log_dict.update(param_dict)

            wandb.log(log_dict)

        for n in range(self.n_iters):

            # Training phase
            train_phy = n % self.train_aug != 0
            _model_aug = self.train_epoch(train_x,
                                          train_y,
                                          mse,
                                          optimizer,
                                          train_phy)

            # Update augmented model
            if not train_phy:
                self.model_aug = _model_aug
            else:
                scheduler.step()

            # Loss of trained model on whole training set
            tr_loss, tr_mse = self.valid_epoch(train_x, train_y, mse)
            val_loss, val_mse = self.valid_epoch(valid_x, valid_y, mse)

            # Compute ||F_a|| on train and valid datasets
            tr_norm_fa = self.compute_norm_fa(train_x)
            val_norm_fa = self.compute_norm_fa(valid_x)

            # Compute distance(fp, f)
            distance = self.compute_fp_distance(train_x, train_y)

            # Save model
            if val_mse < best_loss:
                if self.saving:
                    torch.save(self.model_phy.state_dict(), self.model_phy_path)
                    save_model(self.model_aug, self.model_aug_path, 'augmented')
                best_loss = val_mse

                if wandb_logging:
                    wandb.run.summary['best val_mse'] = val_mse
                    wandb.run.summary['best epoch'] = n+1

            if wandb_logging:
                log_dict = {f"train/train_loss": tr_loss,
                            f"train/train_mse": tr_mse,
                            f"train/valid_mse": val_mse,
                            f"train/valid_loss": val_loss,
                            f"train/learning rate": optimizer.param_groups[0]['lr'],
                            f"train/tr_norm_fa": tr_norm_fa,
                            f"train/valid_norm_fa": val_norm_fa,
                            f"train/d(fp, f)": distance,
                            f"train/epoch": n+1}

                if self.problem in ['friedman1', 'corr_friedman1', 'linear_data', 'overlap_data']:
                    params = []
                    if self.problem in ['friedman1', 'corr_friedman1']:
                        if self.incomplete:
                            if self.fp_term == 'first':
                                params_name = [f'train/p0', f'train/p1']
                                params.append(self.model_phy.p0.item())
                                params.append(self.model_phy.p1.item())
                            elif self.fp_term == 'second':
                                params_name = [f'train/p2', f'train/p3']
                                params.append(self.model_phy.p2.item())
                                params.append(self.model_phy.p3.item())
                            elif self.fp_term == 'third':
                                params_name = [f'train/p4', f'train/p5']
                                params.append(self.model_phy.p4.item())
                                params.append(self.model_phy.p5.item())
                        else:
                            params_name = [f'train/p0', f'train/p1', f'train/p2', f'train/p3', f'train/p4', f'train/p5']
                            params.append(self.model_phy.p0.item())
                            params.append(self.model_phy.p1.item())
                            params.append(self.model_phy.p2.item())
                            params.append(self.model_phy.p3.item())
                            params.append(self.model_phy.p4.item())
                            params.append(self.model_phy.p5.item())

                    elif self.problem in ['linear_data', 'overlap_data']:
                        params.append(self.model_phy.beta.item())
                        params_name = [f'train/beta']

                    param_dict = {}
                    for param, name in zip(params, params_name):
                        param_dict[name] = param
                    log_dict.update(param_dict)

                if self.lambda_loss:
                    log_dict.update({f'train/lambda': self.lambda_j})
                if self.fp_control:
                    log_dict.update({f'train/lambda_fp': self.lambda_fp})

                wandb.log(log_dict)

            print('Finished epoch {:04d}/{:04d} | Train loss : {:.6f} | Validation loss: {:.6f}'.format(n+1, self.n_iters, tr_loss, val_loss))

            # Update lambda
            if self.lambda_loss:
                self.lambda_j += self.lambda_rate * tr_mse
            if self.fp_control:
                self.lambda_fp *= self.geometric_coeff_fp

        return best_loss


    def validate(self, valid_x, valid_y):

        mse = nn.MSELoss()
        val_loss, val_mse = self.valid_epoch(valid_x, valid_y, mse)
        return val_mse

    def predict(self, test_x, test_y=None):
        return self.test_epoch(test_x, test_y, self.res_path)
