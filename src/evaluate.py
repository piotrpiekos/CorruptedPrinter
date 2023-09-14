import numpy as np
import os


def find_the_best_next(state, model, next_optimal):
    from itertools import product

    cur_min = 1999999999

    for control in product(range(-3, 4), range(-3, 4)):
        next_imagined_state = model(state, control)

        dist = np.abs(next_optimal[0] - next_imagined_state[0]) + np.abs(next_optimal[1] - next_imagined_state[1])
        if dist < cur_min:
            cur_min = dist
            opt_control = control
            chosen_next_state = next_imagined_state
        # print(dist, control)

    return opt_control, cur_min, chosen_next_state


def select_current_trajectory(traj_id, all_trajectories):
    return traj_id


def print_with_world_model(starting_point, optimal_states, model):
    cur_imagined_point = starting_point
    selected_controls = []
    for i in range(optimal_states.shape[0]):
        next_optimal = optimal_states[i]
        chosen_control, _, cur_imagined_point = find_the_best_next(cur_imagined_point, model, next_optimal)
    return selected_controls


def print_with_given_controls(starting_point, controls, corr_id, real_simulation):
    cur_point = starting_point
    printed_points = [cur_point]
    for ctrl in controls:
        cur_point = real_simulation(cur_point, ctrl, corr_id)
        printed_points.append(cur_point)
    return printed_points



def evaluate_model(model, split='val'):
    """
    calculates the metric on the dataset for evaluation
    :param model: function that serves as a world model, model: position x control ｜ helper_trajectory -> position_prediction
    :return: MSE over the validation data
    """
    # load the data
    data_directory = os.path.join('data', 'real_data', 'validation' if split == 'val' else 'test')

    true_simulation = 'todo' # todo: implement like in the notebook

    T_trajectory_selector = select_current_trajectory
    for test_type in os.listdir(data_directory):
        for file in os.listdir(os.path.join(data_directory, test_type)):
            file_path = os.path.join(data_directory, test_type, file)
            data = np.genfromtxt(file_path, delimiter=',')

            all_trajectories = data[:, 3].unique()
            for traj_id in all_trajectories:
                cur_trajectory = data[data[:, 3] == traj_id]
                starting_point = cur_trajectory[0, [4, 9]]

                T_trajectory = T_trajectory_selector(traj_id, all_trajectories)
                model.adapt_to_traj(data[data[:, 3] == T_trajectory])

                # calculate the controls that the model will select
                next_optimal_states = cur_trajectory[:, [7, 12]]
                selected_controls = print_with_world_model(starting_point, next_optimal_states, model)
                corr_id = cur_trajectory[0, [1, 2]]

                # calculate what the model will actually print
                print_with_given_controls(starting_point, selected_controls, corr_id, true_simulation)

                # compare it to the original figure
                # todo


evaluate_model(2)
