import numpy as np
import itertools
import sys
import pdb
import matplotlib.pyplot as plt


def initialize_model_params(data_size):
    """
    Initializes model parameters beta_0 and mu_0, for logistic and normal
    models, respectively.

    Args:
        None.

    Returns:
        beta_0: Numpy array, representing the initial logistic model parameter.
        mu_0: Scalar, representing the initial normal model mean parameter.
    """
    beta_0 = np.array([0., 0.])
    mu_0 = 0.
    return beta_0, mu_0


def update_discriminator(current_beta, current_mu, batch_size):
    """
    Performs one gradient update on the logistic beta parameter.

    Args:
        current_beta: The latest beta parameter value.
        current_mu: The latest mu parameter value.
        batch_size: Number of points to sample from data.

    Returns:
        current_beta: The beta parameter value, after gradient update.
    """
    x_batch = np.random.normal(5, 1, batch_size)
    z_batch = np.random.normal(0, 1, batch_size)
    beta_gradient = np.array([0., 0.])

    finished = False
    count = 0
    max_count = 1000
    eps = 1e-4
    while not finished and count < max_count:
        for i in xrange(batch_size):
            beta_gradient[0] += (
                (1 / (1 + np.exp(
                    x_batch[i] * current_beta[1] + current_beta[0]))) -
                (1 / (1 + np.exp(-1 *
                                 (current_beta[0] + current_beta[1] * (
                                     current_mu + z_batch[i]))))))
            beta_gradient[1] += (
                (x_batch[i] / (1 + np.exp(
                    x_batch[i] * current_beta[1] + current_beta[0]))) -
                ((current_mu + z_batch[i]) / (1 + np.exp(-1 *
                                                         (current_beta[0] +
                                                          current_beta[1] * (
                                                          current_mu +
                                                          z_batch[i]))))))

        beta_gradient /= batch_size
        updated_beta = current_beta + beta_gradient

        value_fn_current_beta = evaluate_value_fn(current_beta, current_mu,
                                                  x_batch, z_batch, batch_size)
        value_fn_updated_beta = evaluate_value_fn(updated_beta, current_mu,
                                                  x_batch, z_batch, batch_size)

        if (value_fn_updated_beta - value_fn_current_beta) < eps:
            finished = True
        else:
            current_beta = updated_beta
        count += 1
        if count % 100 == 0:
            print 'Discriminator update count: {}'.format(count)
    return current_beta


def update_generator(current_beta, current_mu, batch_size):
    """
    Performs one gradient update on the normal mu parameter.

    Args:
        current_beta: The latest beta parameter value.
        current_mu: The latest mu parameter value.
        batch_size: Number of points to sample from data.

    Returns:
        updated_mu: The mu parameter value, after gradient update.
    """
    eps = 1e-5
    z_batch = np.random.normal(0, 1, batch_size)
    mu_gradient = 0
    for i in xrange(batch_size):
        mu_gradient += (-current_beta[1] / (
            1 + np.exp(
                -1 * (current_beta[0] + current_beta[1] * (current_mu +
                                                           z_batch[i])))))
    mu_gradient /= batch_size
    #updated_mu = current_mu - mu_gradient
    #return updated_mu

    # Calibrate step for mu.
    finished = False
    count = 0
    max_count = 10
    while not finished and count < max_count:
        value_fn_current_mu = evaluate_value_fn_second_term(
            current_beta, current_mu, z_batch, batch_size)
        value_fn_updated_mu = evaluate_value_fn_second_term(
            current_beta, current_mu - mu_gradient, z_batch, batch_size)
        print 'Delta on value: {}'.format(value_fn_updated_mu -
                                          value_fn_current_mu)
        if value_fn_updated_mu >= value_fn_current_mu:
            mu_gradient /= 2
        else:
            current_mu -= mu_gradient
            finished = True
        count += 1
    return current_mu


def evaluate_value_fn(current_beta, current_mu, x_batch, z_batch, batch_size):
    """
    Evaluates value function.

    Args:
        current_beta: The latest beta parameter value.
        current_mu: The latest mu parameter value.
        x_batch: Random sample, with replacement, from true data.
        z_batch: Random sample, with replacement, from noise data.
        batch_size: Number of points to sample from data.

    Returns:
         value: Value of value function.
    """
    value = 0.0
    for i in xrange(batch_size):
        value += np.log(discriminate_datapoint(x_batch[i], current_beta))
        value += np.log(1 - discriminate_datapoint(
            generate_from_noise(z_batch[i], current_mu), current_beta))
    value /= batch_size
    return value


def evaluate_value_fn_second_term(current_beta, current_mu, z_batch,
                                  batch_size):
    """
    Evaluates only the second term of the value function.

    Args:
        current_beta: The latest beta parameter value.
        current_mu: The latest mu parameter value.
        z_batch: Random sample, with replacement, from noise data.
        batch_size: Number of points to sample from data.

    Returns:
         value: Value of value function.
    """
    value = 0.0
    for i in xrange(batch_size):
        value += np.log(1 - discriminate_datapoint(
            generate_from_noise(z_batch[i], current_mu), current_beta))
    value /= batch_size
    return value


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


def run_one_learning_iteration(current_beta, current_mu, batch_size):
    """
    Performs several discriminator updates, and one generator update.

    Args:
        current_beta: The latest beta parameter value.
        current_mu: The latest mu parameter value.
        batch_size: Number of points to sample from data.

    Returns:
        updated_beta: The latest beta parameter value.
        updated_mu: The latest mu parameter value.
    """
    # Perform several discriminator updates.
    updated_beta = update_discriminator(current_beta, current_mu, batch_size)

    # Perform one generator update.
    updated_mu = update_generator(current_beta, current_mu, batch_size)

    print 'Updated mu, beta: {}, {}'.format(updated_mu, updated_beta)
    return updated_beta, updated_mu


def graph_results(mu_0, current_beta, current_mu, data_size):
    """
    Shows results.

    Args:
        mu_0: The initial normal model mean parameter.
        current_beta: The latest beta parameter value.
        current_mu: The latest mu parameter value.
        data_size: Number of data points in original true dataset.

    Returns:
        None
    """
    x = np.random.normal(5, 1, data_size)
    z = np.random.normal(0, 1, data_size)

    plt.style.use('ggplot')
    fig = plt.figure(1)
    fig.suptitle('Fixed-scale Univariate Normal Generator, Logistic '
                 'Discriminator', size=14)

    ax1 = plt.subplot(211)
    plt.title(r'Data (gray): $N(5, 1)$. Discriminator (blue).', size=10)

    # Histogram and density of true data.
    count_x, bins_x, ignored_x = plt.hist(x, 30, normed=True, color='gray')
    plt.plot(bins_x, 1 / (1 * np.sqrt(2 * np.pi)) *
             np.exp(- (bins_x - 5) ** 2 / (2 * 1 ** 2)),
             linewidth=1, color='gray')
    # Histogram and density of noise data.
    count_z, bins_z, ignored_z = plt.hist(z, 30, normed=True, color='blue')
    plt.plot(bins_z, 1 / (1 * np.sqrt(2 * np.pi)) *
             np.exp(- (bins_z - mu_0) ** 2 / (2 * 1 ** 2)),
             linewidth=1, color='blue')

    plt.subplot(212, sharex=ax1)

    # Histogram and density of true data.
    _, bins_x, _ = plt.hist(x, 30, normed=True, color='gray')
    plt.plot(bins_x, 1 / (1 * np.sqrt(2 * np.pi)) *
             np.exp(- (bins_x - 5) ** 2 / (2 * 1 ** 2)),
             linewidth=1, color='gray')
    # Histogram and density of generated data.
    generated_x = np.random.normal(current_mu, 1, data_size)
    _, bins_generated_x, _ = plt.hist(generated_x, 30, normed=True,
                                      color='blue')
    plt.plot(bins_generated_x, 1 / (1 * np.sqrt(2 * np.pi)) *
             np.exp(- (bins_generated_x - current_mu) ** 2 / (2 * 1 ** 2)),
             linewidth=1, color='blue')

    # Discriminator.
    x_space = np.linspace(-10, 15, 50)
    plt.plot(x_space, 1 / (1 + np.exp(-1 * (current_beta[0] + current_beta[1]
                                            * x_space))), color="blue")

    plt.show()


def main():
    data_size = 1000
    batch_size = 800
    num_learning_iterations = 100

    beta_0, mu_0 = initialize_model_params(data_size)
    current_beta, current_mu = beta_0, mu_0

    for _ in itertools.repeat(None, num_learning_iterations):
        current_beta, current_mu = run_one_learning_iteration(current_beta,
                                                              current_mu,
                                                              batch_size)

    graph_results(mu_0, current_beta, current_mu, data_size)


if __name__ == "__main__":
    main()
