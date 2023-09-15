import numpy as np
import os

from src.models.simple_ml import Oracle, Naive, DirectRandomForest
from src.Simulator import RealSimulator


def find_the_best_next(state, model, next_optimal):
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

    ratio = (next_optimal - state)/diff
    control = ratio.round()
    control = np.clip(np.where(diff!=0, np.nan_to_num(control), 0), -3, 3)
    next_state = state + control * diff
    dist =((next_state - next_optimal)**2).sum()

    return control, dist, next_state


def select_current_trajectory(traj_id, all_trajectories):
    return traj_id


def print_with_world_model(starting_point, optimal_states, model):
    cur_imagined_point = starting_point
    selected_controls = []

    for i in range(optimal_states.shape[0]):
        next_optimal = optimal_states[i]
        chosen_control, _, cur_imagined_point = find_the_best_next_appr(cur_imagined_point, model, next_optimal)
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
    data_directory = os.path.join('data', 'real_data', 'validation' if split == 'val' else 'test')

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
                if len(cur_trajectory) < 300:
                    continue

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


# evaluate_model(DirectRandomForest())
#
print('Naive: ', evaluate_model(Naive()))
#print('Oracle: ', evaluate_model(Oracle()))
print('RF: ', evaluate_model(DirectRandomForest()))
