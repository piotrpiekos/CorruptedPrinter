import torch

import numpy as np
import os

from src.models.simple_ml import Oracle, Naive, DirectRandomForest, DirectLightGBM
from src.Simulator import RealSimulator
from src.models.LSTM_incontext import LSTMencoder_incontext

import pandas as pd

np.random.seed(1)


def find_the_best_next(state, model, next_optimal):
    from itertools import product

    cur_min = 1999999999

    controls = np.array(list(product(range(-3, 4), range(-3, 4))))
    inputs = np.concatenate((np.tile(state, (49, 1)), controls), axis=1)
    next_states = model.batch_call(inputs)

    dist = ((next_states - next_optimal) ** 2).sum(axis=1)
    opt_control = controls[np.argmin(dist)]
    cur_min = dist.min()
    chosen_next_state = next_states[np.argmin(dist)]

    return opt_control, cur_min, chosen_next_state


def find_the_best_next_slow(state, model, next_optimal):
    from itertools import product

    cur_min = 1999999999

    for control in product(range(-3, 4), range(-3, 4)):
        next_imagined_state = model(state, control)

        dist = np.square(next_optimal[0] - next_imagined_state[0]) + np.square(next_optimal[1] - next_imagined_state[1])
        if dist < cur_min:
            cur_min = dist
            opt_control = control
            chosen_next_state = next_imagined_state
        # print(dist, control)

    return opt_control, cur_min, chosen_next_state


def find_the_best_next_appr(state, model, next_optimal):
    from itertools import product
    state, next_optimal = np.array(state), np.array(next_optimal)

    control = (1, 1)
    next_imagined_state = model(state, control)
    diff = next_imagined_state - state

    ratio = (next_optimal - state) / diff
    control = ratio.round()
    control = np.clip(np.where(diff != 0, np.nan_to_num(control), 0), -3, 3)
    next_state = state + control * diff
    dist = ((next_state - next_optimal) ** 2).sum()

    return control, dist, next_state


def select_current_trajectory(traj_id, all_trajectories):
    return traj_id


def print_with_world_model(starting_point, optimal_states, model):
    cur_imagined_point = starting_point
    selected_controls = []

    for i in range(optimal_states.shape[0]):
        next_optimal = optimal_states[i]
        if fast:
            chosen_control, _, cur_imagined_point = find_the_best_next(cur_imagined_point, model, next_optimal)
        else:
            chosen_control, _, cur_imagined_point = find_the_best_next_slow(cur_imagined_point, model, next_optimal)
        selected_controls.append(chosen_control)
    return selected_controls


def print_with_given_controls(starting_point, controls, corr_id, real_simulation):
    cur_point = starting_point
    printed_points = [tuple(cur_point)]
    for ctrl in controls:
        cur_point = real_simulation.get_next(cur_point, ctrl, corr_id)
        printed_points.append(cur_point)
    return printed_points


def RMSE(optimal_points, printed_points):
    MSE = ((optimal_points[:, 0] - printed_points[:, 0]) ** 2 + (
            optimal_points[:, 1] - printed_points[:, 1]) ** 2).mean()

    return np.sqrt(MSE)


def evaluate_model(model, split='val'):
    """
    calculates the metric on the dataset for evaluation
    :param model: function that serves as a world model, model: position x control ｜ helper_trajectory -> position_prediction
    :return: MSE over the validation data
    """
    # load the data
    data_directory = os.path.join('data', 'real',
                                  'validation' if split == 'val' else 'testing')

    true_simulation = RealSimulator()

    T_trajectory_selector = select_current_trajectory

    losses = []
    for test_type in os.listdir(data_directory):
        for file in os.listdir(os.path.join(data_directory, test_type)):
            file_path = os.path.join(data_directory, test_type, file)

            print(file_path)
            data = np.genfromtxt(file_path, delimiter=',')

            all_trajectories = np.unique(data[:, 3])
            print(all_trajectories.shape)
            for traj_id in all_trajectories:
                if traj_id % 10 == 0:
                    print(traj_id)

                cur_trajectory = data[data[:, 3] == traj_id]
                starting_point = cur_trajectory[0, [4, 9]]

                T_trajectory = T_trajectory_selector(traj_id, all_trajectories)
                model.adapt_to_traj(data[data[:, 3] == T_trajectory])

                # calculate the controls that the model will select
                next_optimal_states = cur_trajectory[:, [7, 12]]
                selected_controls = print_with_world_model(starting_point, next_optimal_states, model)
                corr_id = cur_trajectory[0, [1, 2]]

                # calculate what the model will actually print
                printed_points = print_with_given_controls(starting_point, selected_controls, corr_id, true_simulation)

                optimal_states = np.concatenate([np.expand_dims(starting_point, 0), next_optimal_states])
                # compare it to the original figure (Optimal trajectory)

                losses.append(RMSE(optimal_states, np.array(printed_points)))

            break
        break

    return np.mean(losses)


def evaluate_model2(model, split='val'):
    """
    calculates the metric on the dataset for evaluation
    :param model: function that serves as a world model, model: position x control ｜ helper_trajectory -> position_prediction
    :return: MSE over the validation data
    """
    data_directory = os.path.join('data', 'corrupted_pulleys', 'validation' if split == 'val' else 'testing')
    # data_directory = os.path.join('data', 'real_data',
    #                              'validation' if split == 'val' else 'testing')

    true_simulation = RealSimulator()

    T_trajectory_selector = select_current_trajectory

    losses = []
    test_type = 'ext_2'
    files_directory = os.path.join('data', 'corrupted_pulleys', 'validation', 'ext_2')
    # files_directory = os.path.join('data', 'real_data', 'validation', 'ext_1')

    os.path.join(data_directory, test_type)

    print(len(os.listdir(os.path.join(data_directory, test_type))), ' files')
    for file in os.listdir(files_directory)[:1]:
        file_path = os.path.join(data_directory, test_type, file)

        print(file_path)
        data = np.genfromtxt(file_path, delimiter=',')

        all_trajectories = np.unique(data[:, 3])
        print(all_trajectories.shape)
        opt_trajs = []
        printed_trajs = []
        for traj_id in all_trajectories:
            if traj_id % 10 == 0:
                print(traj_id)

            cur_trajectory = data[data[:, 3] == traj_id]
            starting_point = cur_trajectory[0, [4, 9]]

            T_trajectory = T_trajectory_selector(traj_id, all_trajectories)
            model.adapt_to_traj(data[data[:, 3] == T_trajectory])

            # calculate the controls that the model will select
            next_optimal_states = cur_trajectory[:, [7, 12]]
            selected_controls = print_with_world_model(starting_point, next_optimal_states, model)
            corr_id = cur_trajectory[0, [1, 2]]

            # calculate what the model will actually print
            printed_points = print_with_given_controls(starting_point, selected_controls, corr_id, true_simulation)

            optimal_states = np.concatenate([np.expand_dims(starting_point, 0), next_optimal_states])
            # compare it to the original figure (Optimal trajectory)
            opt_trajs.append(optimal_states)
            printed_trajs.append(np.array(printed_points))

        losses.append(RMSE(np.concatenate(opt_trajs), np.concatenate(printed_trajs)))

    return np.mean(losses)


def evaluate_model_adapt_to_body(model, method, split='val'):
    """
    calculates the metric on the dataset for evaluation
    :param model: function that serves as a world model, model: position x control ｜ helper_trajectory -> position_prediction
    :return: MSE over the validation data
    """
    true_simulation = RealSimulator()

    losses = []

    data_for_csv = {'names': [], 'errors': []}
    for file in files:
        file_path = os.path.join(files_directory, file)
        print(file_path)
        data = np.genfromtxt(file_path, delimiter=',')
        if len(data) == 0:
            continue

        all_trajectories_counts = np.unique(data[:, 3], return_counts=True)
        all_trajectories = all_trajectories_counts[0]
        longest_traj_id = all_trajectories[all_trajectories_counts[1].argmax()]
        print('longest traj length: ', all_trajectories_counts[1].max())

        longest_traj = data[data[:, 3] == longest_traj_id]
        model.adapt_to_traj(longest_traj)

        print(all_trajectories.shape)

        opt_trajs = []
        printed_trajs = []
        #for traj_id in np.random.choice(all_trajectories, 3, replace=False):
        for traj_id in all_trajectories:

            cur_trajectory = data[data[:, 3] == traj_id]
            starting_point = cur_trajectory[0, [4, 9]]

            # T_trajectory = T_trajectory_selector(traj_id, all_trajectories)
            # model.adapt_to_traj(data[data[:, 3] == T_trajectory])

            # calculate the controls that the model will select
            next_optimal_states = cur_trajectory[:, [7, 12]]
            selected_controls = print_with_world_model(starting_point, next_optimal_states, model)
            corr_id = cur_trajectory[0, [1, 2]]

            # calculate what the model will actually print
            printed_points = print_with_given_controls(starting_point, selected_controls, corr_id, true_simulation)

            optimal_states = np.concatenate([np.expand_dims(starting_point, 0), next_optimal_states])
            # compare it to the original figure (Optimal trajectory)
            opt_trajs.append(optimal_states)
            printed_trajs.append(np.array(printed_points))

        err = RMSE(np.concatenate(opt_trajs), np.concatenate(printed_trajs))
        losses.append(err)
        data_for_csv['names'].append(file)
        data_for_csv['errors'].append(err)

    pd.DataFrame.from_dict(data_for_csv).to_csv(f'data/results/{method}_dist_with_names.csv')
    np.savetxt('data/results/distribution2.csv', np.array(losses))
    return np.mean(losses)


def print_3dbody(model, body_path):
    """
    calculates the metric on the dataset for evaluation
    :param model: function that serves as a world model, model: position x control ｜ helper_trajectory -> position_prediction
    :return: MSE over the validation data
    """
    # load the data
    true_simulation = RealSimulator()

    T_trajectory_selector = select_current_trajectory

    losses = []

    data = np.genfromtxt(body_path, delimiter=',')

    all_trajectories_counts = np.unique(data[:, 3], return_counts=True)
    all_trajectories = all_trajectories_counts[0]
    longest_traj_id = all_trajectories[all_trajectories_counts[1].argmax()]

    longest_traj = data[data[:, 3] == longest_traj_id]
    model.adapt_to_traj(longest_traj)
    print(all_trajectories.shape)

    trajectories = []
    for traj_id in all_trajectories:
        if traj_id % 10 == 0:
            print(traj_id)

        cur_trajectory = data[data[:, 3] == traj_id]
        starting_point = cur_trajectory[0, [4, 9]]

        # T_trajectory = T_trajectory_selector(traj_id, all_trajectories)
        # model.adapt_to_traj(data[data[:, 3] == T_trajectory])

        # calculate the controls that the model will select
        next_optimal_states = cur_trajectory[:, [7, 12]]
        selected_controls = print_with_world_model(starting_point, next_optimal_states, model)
        corr_id = cur_trajectory[0, [1, 2]]
        # calculate what the model will actually print
        printed_points = print_with_given_controls(starting_point, selected_controls, corr_id, true_simulation)

        optimal_states = np.concatenate([np.expand_dims(starting_point, 0), next_optimal_states])
        # compare it to the original figure (Optimal trajectory)

        printed_with_id = np.c_[np.ones(len(printed_points)) * traj_id, np.array(printed_points)]

        trajectories.append(printed_with_id)
        losses.append(RMSE(optimal_states, np.array(printed_points)))

    res = np.concatenate(trajectories)
    print(res.shape)
    np.savetxt('data/results/rf_100139_1_dense.csv', res)

    return np.mean(losses)


def load_lstm(path):
    model = LSTMencoder_incontext(256, 512)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model



test_type = 'ext_2'
files_directory = os.path.join('data', 'corrupted_pulleys', 'validation', test_type)
files = np.random.choice(os.listdir(files_directory), 100, replace=False)

# evaluate_model(DirectRandomForest())
#
fast = False
print('Naive: ', evaluate_model_adapt_to_body(Naive(), 'naive'))
print('Naive: ', evaluate_model_adapt_to_body(Naive(), 'naive'))
print('Oracle: ', evaluate_model_adapt_to_body(Oracle(), 'oracle'))
fast = True
print('RF: ', evaluate_model_adapt_to_body(DirectRandomForest(), 'rf'))

# print('lgbm: ', evaluate_model_adapt_to_body(DirectLightGBM()))

# model = load_lstm('models/ibex_long.pth')
# with torch.no_grad():
#    print('LSTM: ', evaluate_model2(model))

# bodypath = os.path.join('data', 'corrupted_pulleys_and_loose_belt', 'training', '100139_1.csv')
# bodypath = os.path.join('data', 'corrupted_pulleys', 'training', '100139_pulley_dense.csv')
# lstm_model = load_lstm('models/ibex_big.pth')
# with torch.no_grad():
#    print(print_3dbody(lstm_model, bodypath))
# print(print_3dbody(Naive(), bodypath))
# print(print_3dbody(DirectRandomForest(), bodypath))

# print(print_3dbody(Oracle(), bodypath))
