import os
import dill
import torch

from utils import make_friedman1_data, sample_parameters, make_linear_data, \
    make_overlap_data, make_power_plant_data, make_concrete_data


def create_datasets(problem, term, n_datasets, n_samples_ls, n_samples_test,
    n_features, seeding=False):

    if problem in ['friedman1', 'corr_friedman1']:
        make_dataset_fn = make_friedman1_data
        data_path = f'data/{problem}/{term}'
    elif problem in ['linear_data']:
        make_dataset_fn = make_linear_data
        data_path = f'data/{problem}'
    elif problem in ['overlap_data']:
        make_dataset_fn = make_overlap_data
        data_path = f'data/{problem}'
    elif problem == 'power_plant':
        make_dataset_fn = make_power_plant_data
        data_path = f'data/{problem}'
    elif problem == 'concrete':
        make_dataset_fn = make_concrete_data
        data_path = f'data/{problem}'

    # Create data directory
    os.makedirs(data_path, exist_ok=True)

    # Generate n_datasets datasets for training (1 per parameter sample)
    selected_pairs = {}
    for i in range(n_datasets):

        seed = i+1 if seeding else None

        # Check if already on disk
        train_path = os.path.join(data_path, str(i))
        os.makedirs(train_path, exist_ok=True)

        train_X_path = os.path.join(train_path, 'learning_set_X.pt')
        train_y_path = os.path.join(train_path, 'learning_set_y.pt')
        test_X_path = os.path.join(train_path, 'test_set_X.pt')
        test_y_path = os.path.join(train_path, 'test_set_y.pt')

        if not os.path.exists(train_X_path):

            if problem == 'friedman1':
                params = sample_parameters(problem, n_features, seed=seed)
                dataset = make_dataset_fn(params, n_samples_ls, n_samples_test, seed=seed, term=term, problem=problem)
            elif problem == 'corr_friedman1':
                params = sample_parameters(problem, n_features, seed=seed)
                dataset = make_dataset_fn(params, n_samples_ls, n_samples_test, seed=seed, term=term, problem=problem, selected_pairs=selected_pairs)
                selected_pairs = dataset[-1]
            elif problem == 'linear_data':
                params = {'beta': -0.5, 'gamma': 1}
                dataset = make_dataset_fn(n_samples_ls, n_samples_test, seed=seed)
            elif problem == 'overlap_data':
                params = {'beta': 0.2, 'gamma': 1.5, 'delta': 1}
                dataset = make_dataset_fn(n_samples_ls, n_samples_test, seed=seed, no_covariance=False)
            elif problem in ['power_plant', 'protein', 'concrete']:
                dataset = make_dataset_fn(n_samples_ls, n_samples_test, seed=seed)
            learning_set, test_set = dataset[0], dataset[1]

            # Save train and validation datasets
            torch.save(torch.from_numpy(learning_set[0]), os.path.join(train_X_path))
            torch.save(torch.from_numpy(learning_set[1]), os.path.join(train_y_path))
            torch.save(torch.from_numpy(test_set[0]), os.path.join(test_X_path))
            torch.save(torch.from_numpy(test_set[1]), os.path.join(test_y_path))

            # Save corresponding simulation parameters
            if problem in ['friedman1', 'corr_friedman1', 'linear_data', 'overlap_data']:
                with open(os.path.join(train_path, 'params.pkl'), 'wb') as f:
                    dill.dump(params, f)


if __name__ == '__main__':

    # create_datasets('friedman1', 'entire', 10, n_samples_ls=600, n_samples_test=600, n_features=10, seeding=True)
    # create_datasets('corr_friedman1', 'entire', 10, n_samples_ls=600, n_samples_test=600, n_features=10, seeding=True)
    # create_datasets('linear_data', term=None, n_datasets=10, n_samples_ls=100, n_samples_test=600, n_features=2, seeding=True)
    # create_datasets('overlap_data', term=None, n_datasets=10, n_samples_ls=100, n_samples_test=600, n_features=2, seeding=True)
    # create_datasets('power_plant', term=None, n_datasets=10, n_samples_ls=200, n_samples_test=9000, n_features=4, seeding=True)
    create_datasets('concrete', term=None, n_datasets=10, n_samples_ls=200, n_samples_test=830, n_features=9, seeding=True)
