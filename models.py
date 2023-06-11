import torch
import torch.nn as nn


class friedman_model(nn.Module):

    def __init__(self, term, incomplete, params=None, constant=False, seed=0):

        super(friedman_model, self).__init__()

        if seed is None:
            seed = 0
        else:
            torch.manual_seed(seed)

        self.term = term
        self.incomplete = incomplete
        self.constant = constant

        self.real = params
        if params is not None:
            if self.term == 'first':
                self.p0 = nn.Parameter(params['p0'], requires_grad=False)
                self.p1 = nn.Parameter(params['p1'], requires_grad=False)
            elif self.term == 'second':
                self.p2 = nn.Parameter(params['p2'], requires_grad=False)
                self.p3 = nn.Parameter(params['p3'], requires_grad=False)
            elif self.term == 'third':
                self.p4 = nn.Parameter(params['p4'], requires_grad=False)
                self.p5 = nn.Parameter(params['p5'], requires_grad=False)

            if not self.incomplete:
                if self.term == 'first':
                    self.p2 = nn.Parameter(params['p2'], requires_grad=False)
                    self.p3 = nn.Parameter(params['p3'], requires_grad=False)
                    self.p4 = nn.Parameter(params['p4'], requires_grad=False)
                    self.p5 = nn.Parameter(params['p5'], requires_grad=False)
                elif self.term == 'second':
                    self.p0 = nn.Parameter(params['p0'], requires_grad=False)
                    self.p1 = nn.Parameter(params['p1'], requires_grad=False)
                    self.p4 = nn.Parameter(params['p4'], requires_grad=False)
                    self.p5 = nn.Parameter(params['p5'], requires_grad=False)
                elif self.term == 'third':
                    self.p0 = nn.Parameter(params['p0'], requires_grad=False)
                    self.p1 = nn.Parameter(params['p1'], requires_grad=False)
                    self.p2 = nn.Parameter(params['p2'], requires_grad=False)
                    self.p3 = nn.Parameter(params['p3'], requires_grad=False)
        else:
            self.p0 = nn.Parameter(torch.rand(1))
            self.p1 = nn.Parameter(torch.rand(1))
            self.p2 = nn.Parameter(torch.rand(1))
            self.p3 = nn.Parameter(torch.rand(1))
            self.p4 = nn.Parameter(torch.rand(1))
            self.p5 = nn.Parameter(torch.rand(1))

        if self.constant:
            self.c = nn.Parameter(torch.rand(1))
        else:
            self.c = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x):

        if self.incomplete:
            if self.term == 'first':
                y = self.p0 * torch.sin(self.p1 * x[:, 0] * x[:, 1])
            elif self.term == 'second':
                y = self.p2 * (x[:, 2] - self.p3) ** 2
            elif self.term == 'third':
                y = self.p4 * x[:, 3] + self.p5 * x[:, 4]
        else:
            y = self.p0 * torch.sin(self.p1 * x[:, 0] * x[:, 1]) + self.p2 * (x[:, 2] - self.p3) ** 2  + self.p4 * x[:, 3] + self.p5 * x[:, 4]

        if self.constant:
            y  = y + self.c
            return y
        else:
            return y

class linear_model(nn.Module):

    def __init__(self, constant, params=None, seed=0):

        super().__init__()

        if seed is None:
            seed = 0
        else:
            torch.manual_seed(seed)

        if params is not None:
            self.beta = nn.Parameter(params['beta'], requires_grad=False)
        else:
            self.beta = nn.Parameter(torch.rand(1))

        self.constant = constant

        if self.constant:
            self.c = nn.Parameter(torch.rand(1))
        else:
            self.c = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x):

        y = self.beta * x[:, 0]

        if self.constant:
            y = y + self.c
            return y
        else:
            return y

class overlap_model(nn.Module):

    def __init__(self, constant, params=None, seed=0):

        super().__init__()

        if seed is None:
            seed = 0
        else:
            torch.manual_seed(seed)

        if params is not None:
            self.beta = nn.Parameter(params['beta'], requires_grad=False)
        else:
            self.beta = nn.Parameter(torch.rand(1))

        self.constant = constant

        if self.constant:
            self.c = nn.Parameter(torch.rand(1))
        else:
            self.c = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x):

        y = self.beta * x[:, 0] ** 2

        if self.constant:
            y = y + self.c
            return y
        else:
            return y

class power_plant_model(nn.Module):

    def __init__(self, constant, params=None, seed=0):

        super().__init__()

        if seed is None:
            seed = 0
        else:
            torch.manual_seed(seed)

        if params is not None:
            self.p0 = nn.Parameter(params['p0'], requires_grad=False)
        else:
            self.p0 = nn.Parameter(torch.rand(1))

        self.constant = constant

        if self.constant:
            self.c = nn.Parameter(torch.rand(1))
        else:
            self.c = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x):

        y = self.p0 * x[:, 0]

        if self.constant:
            y = y + self.c
            return y
        else:
            return y

class concrete_model(nn.Module):

    def __init__(self, constant, params=None, seed=0):

        super().__init__()

        if seed is None:
            seed = 0
        else:
            torch.manual_seed(seed)

        if params is not None:
            self.p0 = nn.Parameter(params['p0'], requires_grad=False)
        else:
            self.p0 = nn.Parameter(torch.rand(1))

        self.constant = constant

        if self.constant:
            self.c = nn.Parameter(torch.rand(1))
        else:
            self.c = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x):

        y = self.p0 * x[:, 0]

        if self.constant:
            y = y + self.c
            return y
        else:
            return y

class MLP(nn.Module):

    def __init__(self, in_size, num_layers, hidden_size, out_size, train_mean_X,
        train_std_X, train_mean_y, train_std_y):
        super().__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.train_mean_X = train_mean_X
        self.train_std_X = train_std_X
        self.train_mean_y = train_mean_y
        self.train_std_y = train_std_y

        # Initialize sequential module
        layers = []

        if num_layers == 0:
            layers.append(nn.Linear(self.in_size, self.out_size))

        else:
            previous_size = self.in_size
            for _ in range(num_layers):
                layers.append(nn.Linear(previous_size, hidden_size))
                layers.append(nn.ReLU())
                previous_size = hidden_size
            layers.append(nn.Linear(previous_size, self.out_size))

        self.net = nn.Sequential(*layers).double()

    def forward(self, x):
        if self.train_mean_X is not None:
            x = (x - self.train_mean_X) / self.train_std_X

        x = self.net(x)

        if self.train_mean_y is not None:
            x = x * self.train_std_y + self.train_mean_y

        x = x.squeeze()
        return x

    def get_derivatives(self, x):
        x = self.forward(x)
        return x

class Forecaster(nn.Module):
    def __init__(self, model_phy, model_aug, is_augmented, jointly):
        super().__init__()

        self.model_phy = model_phy
        self.model_aug = model_aug
        self.is_augmented = is_augmented
        self.jointly = jointly

    def forward(self, x_phy, x_aug, train_fp=False):
        if self.jointly:
            if self.model_phy is not None:
                res_phy = self.model_phy(x_phy)
            if self.is_augmented:
                res_aug = self.model_aug(x_aug)
                if self.model_phy is not None:
                    return res_phy + res_aug
                else:
                    return res_aug
            else:
                return res_phy

        else:
            if train_fp:
                res_phy = self.model_phy(x_phy)
                if self.model_aug is not None:
                    with torch.no_grad():
                        res_aug = self.model_aug(x_aug)
                    return res_phy + res_aug
                else:
                    return res_phy
            else:
                res_aug = self.model_aug(x_aug)
                if self.model_phy is not None:
                    with torch.no_grad():
                        res_phy = self.model_phy(x_phy)
                    return res_phy + res_aug
                else:
                    return res_aug

    def get_params(self):
        return self.model_phy.params
