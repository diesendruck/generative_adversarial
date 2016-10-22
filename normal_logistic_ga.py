import itertools
import sys
import pdb
import time
import datetime
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from normal_logistic_ga_utils import evaluate_value_fn, discriminate_datapoint,\
    safe_ln, evaluate_value_fn_second_term, generate_from_noise


def initialize_model_params():
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


def update_discriminator(current_beta, current_mu, batch_size, true_data_mean):
    """
    Performs one gradient update on the logistic beta parameter.

    Args:
        current_beta: The latest beta parameter value.
        current_mu: The latest mu parameter value.
        batch_size: Number of points to sample from data.
        true_data_mean: Value of x in N(x, 1), distribution of true data.

    Returns:
        updated_beta: The beta parameter value, after gradient update.
        value_delta: Change of value function from current to updated beta.
    """
    # Sample batches.
    x_batch = np.random.normal(true_data_mean, 1, batch_size)
    z_batch = np.random.normal(0, 1, batch_size)
    beta_gradient = np.array([0., 0.])

    # Compute stochastic gradient using batches.
    for i in xrange(batch_size):
        beta_gradient[0] += (
            (1 / (1 + np.exp(current_beta[0] + current_beta[1] *
                             x_batch[i]))) -
            (1 / (1 + np.exp(-1 *
                             (current_beta[0] + current_beta[1] * (
                                 current_mu + z_batch[i]))))))
        beta_gradient[1] += (
            (x_batch[i] / (1 + np.exp(
                current_beta[0] + current_beta[1] * x_batch[i]))) -
            (generate_from_noise(z_batch[i], current_mu) / (
                1 + np.exp(-1 * (current_beta[0] + current_beta[1] * (
                    current_mu + z_batch[i]))))))
    beta_gradient /= batch_size

    # Check to see that gradient moved in the correct direction.
    candidate_beta = current_beta + beta_gradient
    value_fn_current_beta = evaluate_value_fn(
        current_beta, current_mu, x_batch, z_batch, batch_size,
        true_data_mean)
    value_fn_candidate_beta = evaluate_value_fn(
        candidate_beta, current_mu, x_batch, z_batch, batch_size,
        true_data_mean)
    value_delta = value_fn_candidate_beta - value_fn_current_beta
    current_beta = candidate_beta

    return current_beta, value_delta


def update_generator(current_beta, current_mu, batch_size, true_data_mean,
                     global_iteration):
    """
    Performs one gradient update on the normal mu parameter.

    Args:
        current_beta: The latest beta parameter value.
        current_mu: The latest mu parameter value.
        batch_size: Number of points to sample from data.
        true_data_mean: Value of x in N(x, 1), distribution of true data.
        global_iteration: Count of total global iterations.

    Returns:
        updated_mu: The mu parameter value, after gradient update.
        value_delta: Change of value function from current to updated mu.
    """
    z_batch = np.random.normal(0, 1, batch_size)
    mu_gradient = 0
    for i in xrange(batch_size):
        mu_gradient += (-current_beta[1] / (
            1 + np.exp(
                -1 * (current_beta[0] + current_beta[1] * (
                    generate_from_noise(z_batch[i], current_mu))))))
        # REGULARIZATION
        # mu_gradient += np.sign(current_mu - true_data_mean)
    mu_gradient /= batch_size

    # Calibrate step for mu.
    finished = False
    count = 0
    max_count = 1
    while not finished and count < max_count:
        count += 1

        candidate_mu = current_mu - 0.5*np.sign(mu_gradient)*(
            .99**global_iteration)
        #candidate_mu = current_mu - mu_gradient
        value_fn_current_mu = evaluate_value_fn_second_term(
            current_beta, current_mu, z_batch, batch_size, true_data_mean)
        value_fn_candidate_mu = evaluate_value_fn_second_term(
            current_beta, candidate_mu, z_batch, batch_size,
            true_data_mean)
        value_delta = value_fn_candidate_mu - value_fn_current_mu

        current_mu = candidate_mu
        finished = True
        # Try scaling back the step size, to see if a fraction will reduce
        # value function.
        # if value_delta >= 0:
        #     print '    ...tried with mu: {} - {}, but not better...'.format(
        #         round(current_mu, 2), round(mu_gradient, 2))
        #     mu_gradient *= 0.8
        # else:
        #     print '    Good step with mu: {} - {}.'.format(
        #         round(current_mu, 2), round(mu_gradient, 2))
        #     current_mu = candidate_mu
        #     finished = True

    return current_mu, value_delta


def compute_val_loglik(current_beta, current_mu, true_data_mean, batch_size):
    """
    Visualize relationship between value function and pseudo-likelihood.

    Args:
        current_beta: The latest beta parameter value.
        current_mu: The latest mu parameter value.
        true_data_mean: Value of x in N(x, 1), distribution of true data.
        batch_size: Size of data and noise batches.

    Returns:
        Tuple of (1) value function value and (2) log likelihood relative to
          true distribution, for the generated data.
    """
    x_batch = np.random.normal(true_data_mean, 1, batch_size)
    z_batch = np.random.normal(0, 1, batch_size)
    value_generated_data = evaluate_value_fn(current_beta, current_mu, x_batch,
                                             z_batch, batch_size,
                                             true_data_mean)
    loglik_generated_data = 0
    for z in z_batch:
        generated = generate_from_noise(z, current_mu)
        loglik_generated_data += \
            norm.logpdf(generated, loc=true_data_mean, scale=1)
    return value_generated_data, loglik_generated_data


def run_one_learning_iteration(
        current_beta, current_mu, batch_size, true_data_mean, max_k_updates,
        max_mu_updates, global_iteration, vals_logliks):
    """
    Performs several discriminator updates, and one generator update.

    Args:
        current_beta: The latest beta parameter value.
        current_mu: The latest mu parameter value.
        batch_size: Number of points to sample from data.
        true_data_mean: Value of x in N(x, 1), distribution of true data.
        max_k_updates: See name.
        max_mu_updates: See name.
        global_iteration: Index of global iterations.
        vals_logliks: Array of values and log likelihoods for each (mu,
          beta) pair.

    Returns:
        current_beta: The latest beta parameter value.
        current_mu: The latest mu parameter value.
        vals_logliks: Array of values and log likelihoods for each (mu,
          beta) pair.
    """
    count = 0
    while count < max_k_updates:
        count += 1
        updated_beta, value_delta = update_discriminator(
            current_beta, current_mu, batch_size, true_data_mean)
        current_beta = updated_beta

    count = 0
    while count < max_mu_updates:
        count += 1
        updated_mu, value_delta = update_generator(current_beta, current_mu,
                                                   batch_size, true_data_mean,
                                                   global_iteration)
        current_mu = updated_mu

    val_loglik = compute_val_loglik(current_beta, current_mu, true_data_mean,
                                    batch_size)
    vals_logliks.append(val_loglik)

    print 'UPDATING BETA...\n    beta PRE: {}'.format(current_beta)
    print '    beta POST: {}'.format(updated_beta)
    print 'UPDATING MU...\n    mu PRE: {}'.format(current_mu)
    print '    mu POST: {}'.format(updated_mu)
    return current_beta, current_mu, vals_logliks


def graph_results(mu_0, true_data_mean, current_beta, current_mu, data_size,
                  x, iter, file_timestamp):
    """
    Shows status of discriminator and generator, relative to true data.

    Assumes the figure has already been defined.

    Args:
        mu_0: The initial normal model mean parameter.
        true_data_mean: Value of x in N(x, 1), distribution of true data.
        current_beta: The latest beta parameter value.
        current_mu: The latest mu parameter value.
        data_size: Number of data points in original true dataset.
        x: The original data set.
        iter: Iteration, passed for filenaming.
        file_timestamp: Datetime used for timestamping all figure outputs.

    Returns:
        None
    """
    # Plotting settings.
    plt.style.use('ggplot')
    fig = plt.figure()
    fig.suptitle(
        'Fixed-scale Univariate Normal Generator, Logistic '
        'Discriminator', size=14)

    # Histogram and density of TRUE data.
    ax1 = fig.add_subplot(211)
    plt.title(r'Data (gray). Discriminator (blue).', size=10)
    count_x, bins_x, ignored_x = plt.hist(x, 30, normed=True, color='gray')
    plt.plot(bins_x, 1 / (1 * np.sqrt(2 * np.pi)) *
             np.exp(- (bins_x - true_data_mean) ** 2 / (2 * 1 ** 2)),
             linewidth=1, color='gray')
    # Histogram and density of NOISE data.
    z = np.random.normal(0, 1, data_size)
    count_z, bins_z, ignored_z = plt.hist(z, 30, normed=True, color='blue')
    plt.plot(bins_z, 1 / (1 * np.sqrt(2 * np.pi)) *
             np.exp(- (bins_z - mu_0) ** 2 / (2 * 1 ** 2)),
             linewidth=1, color='blue')

    ax2 = fig.add_subplot(212, sharex=ax1)

    _, bins_x, _ = ax2.hist(x, 30, normed=True, color='gray')
    plt.plot(bins_x, 1 / (1 * np.sqrt(2 * np.pi)) *
             np.exp(- (bins_x - true_data_mean) ** 2 / (2 * 1 ** 2)),
             linewidth=1, color='gray')

    # Histogram and density of GENERATED data.
    generated_x = np.random.normal(current_mu, 1, data_size)
    _, bins_generated_x, _ = ax2.hist(generated_x, 30, normed=True,
                                      color='blue')
    plt.plot(bins_generated_x, 1 / (1 * np.sqrt(2 * np.pi)) *
             np.exp(- (bins_generated_x - current_mu) ** 2 / (2 * 1 ** 2)),
             linewidth=1, color='blue')

    # Discriminator.
    x_space = np.linspace(- 5, true_data_mean + 5, 50)
    plt.plot(x_space,
             1 / (1 + np.exp(-1 * (current_beta[0] + current_beta[1]
                                   * x_space))), color="blue")

    plt.savefig('{}-fig{}.png'.format(file_timestamp, iter))


def graph_traceplots(true_data_mean, traceplot_beta, traceplot_mu,
                     file_timestamp):
    """
    Shows traceplots.

    Args:
        true_data_mean: Value of x in N(x, 1), distribution of true data.
        traceplot_beta: Collection of sampled beta values.
        traceplot_mu: Collection of sampled mu values.
        file_timestamp: Datetime used for timestamping all figure outputs.

    Returns:
        None
    """
    display = False
    plt.style.use('ggplot')
    fig = plt.figure()
    fig.suptitle('Fixed-scale Univariate Normal Generator, Logistic '
                 'Discriminator: Traceplots', size=14)

    ax1 = plt.subplot(311)
    plt.title(r'Mu', size=10)
    plt.plot(traceplot_mu, color='blue')
    plt.axhline(y=true_data_mean, xmin=0, xmax=1, hold=None,
                color='gray', label='True mean')
    ax1.legend(loc='best')

    ax2 = plt.subplot(312)
    plt.title(r'Mu Steps', size=10)
    plt.plot(np.diff(traceplot_mu), color='red')

    ax3 = plt.subplot(313)
    plt.title(r'Beta', size=10)
    plt.plot(np.array(traceplot_beta)[:, 0], label='b0')
    plt.plot(np.array(traceplot_beta)[:, 1], label='b1')
    ax3.legend(loc='best')

    plt.savefig('{}-traceplot'.format(file_timestamp))
    if display:
        plt.show()


def graph_vals_logliks(vals_logliks):
    """
    Plot the value function value and the log likelihood of the generated
    data, for the mu-beta pair of each global iteration.

    Args:
        vals_logliks: Array of values and log likelihoods for each (mu,
          beta) pair.

    Returns:
        None
    """
    plt.style.use('ggplot')
    fig = plt.figure()
    fig.suptitle('Value vs. Log Likelihood for Generated Data Against True '
                 'Data Distribution', size=14)
    vals = [v for (v, l) in vals_logliks]
    logliks = [l for (v, l) in vals_logliks]
    t = np.arange(len(vals))
    plt.scatter(vals, logliks, c=t, cmap='cool', s=50)
    plt.xlabel('Value')
    plt.ylabel('Log Likelihood')


def main():
    start = time.time()
    file_timestamp = '{:%Y%m%d-%H%M%S}'.format(datetime.datetime.now())

    # Adjustable procedural parameters.
    true_data_mean = 8
    data_size = 1000
    batch_size = 10
    num_training_iterations = 200
    max_k_updates = 200
    max_mu_updates = 1
    display = True

    # Full data, and initial parameters.
    x = np.random.normal(true_data_mean, 1, data_size)
    beta_0, mu_0 = initialize_model_params()
    current_beta, current_mu = beta_0, mu_0
    traceplot_beta = []
    traceplot_mu = []
    vals_logliks = []

    for iter in range(num_training_iterations):
        print '\n\n----------Global learning iter {}----------'.format(iter)
        # graph_results(mu_0, true_data_mean, current_beta, current_mu, data_size,
        #               x, iter, file_timestamp)

        current_beta, current_mu, vals_logliks = run_one_learning_iteration(
            current_beta, current_mu, batch_size, true_data_mean, max_k_updates,
            max_mu_updates, iter, vals_logliks)
        traceplot_beta.append(current_beta)
        traceplot_mu.append(current_mu)

    graph_traceplots(true_data_mean, traceplot_beta, traceplot_mu,
                     file_timestamp)
    graph_vals_logliks(vals_logliks)

    print '\n\nTime Elapsed: {}'.format(time.time() - start)
    if display:
        plt.show(block=True)

if __name__ == "__main__":
    main()