import numpy as np
import itertools
import sys


def initialize_data(data_size):
    """
    Generates true data, x, and initial noise data, z.

    Args:
        data_size: Number of data points in original true dataset.

    Returns:
        x_labeled: True data Numpy array of size (data_size, 2), where each
          element [x, 1] indicates value x of class 1.
        z_labeled: Noise data Numpy array of size (data_size, 2), where each
          element [z, 0] indicates value x of class 0.
    """
    mu, sigma = 5, 1
    x = np.random.normal(mu, sigma, data_size)
    z = np.random.normal(0, 1, data_size)
    x_labeled = np.array([[i, 1] for i in x])  # True data gets class 1.
    z_labeled = np.array([[j, 0] for j in z])  # Fake data gets class 0.
    return x_labeled, z_labeled


def initialize_model_params(data_size):
    """
    Initializes model parameters beta_0 and mu_0, for logistic and normal
    models, respectively.

    Args:
        None

    Returns:
        beta_0: Scalar, representing the initial logistic model parameter.
        mu_0: Scalar, representing the initial normal model mean parameter.
    """
    beta_0 = 0
    mu_0 = 0
    return beta_0, mu_0


def update_discriminator(x_labeled, z_labeled, current_beta, current_mu,
                         batch_size):
    """
    Performs one gradient update on the logistic beta parameter.

    Args:
        x_labeled: True data array of size (data_size, 2), where each element
          [x, 1] indicates value x of class 1.
        z_labeled: Noise data array of size (data_size, 2), where each element
           [z, 0] indicates value x of class 0.
        current_beta: The latest beta parameter value.
        current_mu: The latest mu parameter value.
        batch_size: Number of points to sample from data.

    Returns:
        updated_beta: The beta parameter value, after gradient update.
    """
    x_batch = [i[0] for i in sample_batch(batch_size, x_labeled)]
    z_batch = [i[0] for i in sample_batch(batch_size, z_labeled)]
    beta_gradient = 0
    for i in xrange(batch_size):
        beta_gradient += (
            (x_batch[i] / (1 + np.exp(x_batch[i] * current_beta))) -
            ((current_mu + z_batch[i]) / (1 + np.exp(-current_beta * (
                current_mu + z_batch[i])))))
        beta_gradient += 1e-3*(current_mu - 5)
    beta_gradient /= batch_size
    updated_beta = current_beta + beta_gradient
    return updated_beta


def update_generator(z_labeled, current_beta, current_mu, batch_size):
    """
    Performs one gradient update on the normal mu parameter.

    Args:
        z_labeled: Noise data array of size (data_size, 2), where each element
           [z, 0] indicates value x of class 0.
        current_beta: The latest beta parameter value.
        current_mu: The latest mu parameter value.
        batch_size: Number of points to sample from data.

    Returns:
        updated_mu: The mu parameter value, after gradient update.
    """
    z_batch = [i[0] for i in sample_batch(batch_size, z_labeled)]
    mu_gradient = 0
    for i in xrange(batch_size):
        mu_gradient -= (-current_beta / (1 + np.exp(-current_beta * (
            current_mu + z_batch[i]))))
    mu_gradient /= batch_size
    updated_mu = current_mu + mu_gradient
    return updated_mu


def generate_from_noise(noise, current_mu):
    """
    Generates from normal model, using noise and current mu.

    Args:
        noise: Standard Gaussian noise, in dimension of original data.
        current_mu: The latest mu parameter value.

    Returns:
        generated_datapoint: Single datapoint from generative model.
    """
    generated_datapoint = current_mu + noise
    return generated_datapoint


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


def run_one_learning_iteration(num_consecutive_discriminator_updates,
                               x_labeled, z_labeled, current_beta,
                               current_mu, batch_size):
    """
    Performs several discriminator updates, and one generator update.

    Args:
        num_consecutive_discriminator_updates: Count for discriminator updates.
        x_labeled: True data array of size (data_size, 2), where each element
          [x, 1] indicates value x of class 1.
        z_labeled: Noise data array of size (data_size, 2), where each element
           [z, 0] indicates value x of class 0.
        current_beta: The latest beta parameter value.
        current_mu: The latest mu parameter value.
        batch_size: Number of points to sample from data.

    Returns:
        current_beta: The latest beta parameter value.
        current_mu: The latest mu parameter value.
    """
    # Perform several discriminator updates.
    for _ in itertools.repeat(None, num_consecutive_discriminator_updates):
        current_beta = update_discriminator(x_labeled, z_labeled, current_beta,
                                            current_mu, batch_size)
    # Perform one generator update.
    current_mu = update_generator(z_labeled, current_beta, current_mu,
                                  batch_size)
    return current_beta, current_mu


def main():
    data_size = 100
    batch_size = 80
    num_consecutive_discriminator_updates = 5
    num_learning_iterations = 50

    x_labeled, z_labeled = initialize_data(data_size)
    current_beta, current_mu = initialize_model_params(data_size)

    for _ in itertools.repeat(None, num_learning_iterations):
        print 'Current mu, beta: {}, {}'.format(current_mu, current_beta)
        current_beta, current_mu = run_one_learning_iteration(
            num_consecutive_discriminator_updates, x_labeled, z_labeled,
            current_beta, current_mu, batch_size)

    print 'Current mu, beta: {}, {}'.format(current_mu, current_beta)

if __name__ == "__main__":
    main()

"""
import matplotlib.pyplot as plt

# True data x.
count, bins, ignored = plt.hist(x, 30, normed=True)
plt.plot(bins, 1/ (sigma * np.sqrt(2 * np.pi)) *
         np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)),
         linewidth=2, color='r')

count_z, bins_z, ignored_z = plt.hist(z, 30, normed=True)
plt.plot(bins_z, 1 / (1 * np.sqrt(2 * np.pi)) *
         np.exp(- (bins_z - 0) ** 2 / (2 * 1 ** 2)),
         linewidth=2, color='b')
"""
