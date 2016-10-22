# Helper functions for normal_logistic_ga.py

import numpy as np


def evaluate_value_fn(current_beta, current_mu, x_batch, z_batch, batch_size,
                      true_data_mean):
    """
    Evaluates value function.

    Args:
        current_beta: The latest beta parameter value.
        current_mu: The latest mu parameter value.
        x_batch: Random sample, with replacement, from true data.
        z_batch: Random sample, with replacement, from noise data.
        batch_size: Number of points to sample from data.
        true_data_mean: Value of x in N(x, 1), distribution of true data.

    Returns:
         value: Value of value function.
    """
    value = 0.0
    for i in xrange(batch_size):
        value += safe_ln(discriminate_datapoint(x_batch[i], current_beta))
        value += safe_ln(1 - discriminate_datapoint(
            generate_from_noise(z_batch[i], current_mu), current_beta))
        # REGULARIZATION
        #value += abs(current_mu - true_data_mean)
    value /= batch_size
    return value


def evaluate_value_fn_second_term(current_beta, current_mu, z_batch,
                                  batch_size, true_data_mean):
    """
    Evaluates only the second term of the value function.

    Args:
        current_beta: The latest beta parameter value.
        current_mu: The latest mu parameter value.
        z_batch: Random sample, with replacement, from noise data.
        batch_size: Number of points to sample from data.
        true_data_mean: Value of x in N(x, 1), distribution of true data.

    Returns:
         value: Value of value function.
    """
    value = 0.0
    for i in xrange(batch_size):
        value += safe_ln(1 - discriminate_datapoint(
            generate_from_noise(z_batch[i], current_mu), current_beta))
        # REGULARIZATION
        #value += abs(current_mu - true_data_mean)
    value /= batch_size
    return value


def safe_ln(x, minval=0.0000000001):
    return np.log(x.clip(min=minval))


def discriminate_datapoint(datapoint, current_beta):
    """
    Produces probability of being true data, using logistic model.

    Args:
        datapoint: True or generated datapoint.
        current_beta: The latest beta parameter value.

    Returns:
        probability_true_data: Probability of being from the true data.
    """
    probability_true_data = (
        1 / (1 + np.exp(-1 * (datapoint * current_beta[1] + current_beta[0]))))
    return probability_true_data


def generate_from_noise(noise, current_mu):
    """
    Generates from normal model, using noise and current mu.

    Args:
        noise: Standard Gaussian noise, in dimension of original data.
        current_mu: The latest mu parameter value.

    Returns:
        generated_datapoint: Single datapoint from generative model.
    """
    generated_datapoint = noise + current_mu
    return generated_datapoint


# NOT CURRENTLY BEING USED.
def sample_batch(batch_size, data):
    """
    Generates batch from full data.

    Args:
        batch_size: Number of points to sample from data.
        data: Numpy array of data (e.g. true or generated).

    Returns:
         batch: Numpy array of size (batch_size, 2), where each element [x, y]
           indicates value x of class y.
    """
    num_elements = data.shape[0]
    if batch_size > num_elements:
        sys.exit("Batch size exceed data size.")
    batch_indices = np.random.choice(range(num_elements), batch_size)
    batch = [data[i] for i in batch_indices]
    batch = np.array(batch)
    return batch