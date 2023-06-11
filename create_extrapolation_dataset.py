import os
import dill
import torch

from utils import make_power_plant_data, make_concrete_data

def create_datasets(problem, n_datasets, y_low, y_high, seeding=False):

    if problem == 'power_plant':
        make_dataset_fn = make_power_plant_data
    elif problem == 'concrete':
        make_dataset_fn = make_concrete_data
    data_path = f'data_extrapolation/{problem}'

    # Create data directory
    os.makedirs(data_path, exist_ok=True)

    # Generate n_datasets datasets for training (1 per parameter sample)
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

            dataset = make_dataset_fn(y_low=y_low, y_high=y_high, seed=seed)
            learning_set, test_set = dataset[0], dataset[1]

            # Save train and validation datasets
            torch.save(torch.from_numpy(learning_set[0]), os.path.join(train_X_path))
            torch.save(torch.from_numpy(learning_set[1]), os.path.join(train_y_path))
            torch.save(torch.from_numpy(test_set[0]), os.path.join(test_X_path))
            torch.save(torch.from_numpy(test_set[1]), os.path.join(test_y_path))


if __name__ == '__main__':

    create_datasets('power_plant', n_datasets=10, y_low=445, y_high=445, seeding=True)
    create_datasets('concrete', n_datasets=10, y_low=25, y_high=25, seeding=True)

