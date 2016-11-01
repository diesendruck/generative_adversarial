import itertools
import sys
import pdb
import time
import datetime
import numpy as np
from scipy.stats import norm
from collections import namedtuple
import matplotlib.pyplot as plt
from binary_classifier_nn import run_binary_classifier_nn
from sbm_gan_utils import rbern, sym_matrix, evaluate_value_fn, \
    discriminate_datapoint, safe_ln, evaluate_value_fn_second_term, \
    generate_from_noise


COLORS = {'lightgreen': '#BDDFB3',
          'blue': '#2BAA9C',
          'black': '#2F2E2E',
          'red': '#85403B',
          'green': '#456F3F',
          'std_red': 'red'}


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

        # Choose gradient step method.
        option = 1
        if option == 1:
            candidate_mu = current_mu - 0.1*np.sign(mu_gradient)*(
                .99**global_iteration)
        elif option == 2:
            candidate_mu = current_mu - mu_gradient
        value_fn_current_mu = evaluate_value_fn_second_term(
            current_beta, current_mu, z_batch, batch_size, true_data_mean)
        value_fn_candidate_mu = evaluate_value_fn_second_term(
            current_beta, candidate_mu, z_batch, batch_size,
            true_data_mean)
        value_delta = value_fn_candidate_mu - value_fn_current_mu

        current_mu = candidate_mu
        finished = True

        step_scaling = 0
        if step_scaling:
            # Try scaling back the step size, to see if a fraction will reduce
            # value function.
            if value_delta >= 0:
                print '    ...tried with mu: {} - {}, but not better...'.format(
                    round(current_mu, 2), round(mu_gradient, 2))
                mu_gradient *= 0.8
            else:
                print '    Good step with mu: {} - {}.'.format(
                    round(current_mu, 2), round(mu_gradient, 2))
                current_mu = candidate_mu
                finished = True

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
        max_mu_updates, global_iteration, optimal_vals_logliks_mus,
        within_run_vals_logliks):
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
        optimal_vals_logliks_mus: Array of values, log likelihoods, and mus for
          each (mu, beta) pair.
        within_run_vals_logliks: Array of arrays, where each element is a
          list of value-loglikelihood pairs, for a grid of mus and a fixed
          beta, for one global iteration.

    Returns:
        current_beta: The latest beta parameter value.
        current_mu: The latest mu parameter value.
        optimal_vals_logliks_mus: Array of values, log likelihoods, and mus for
          each (mu, beta) pair.
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
    val_loglik_mu = val_loglik + (current_mu,)
    optimal_vals_logliks_mus.append(val_loglik_mu)

    num_within_run_graphs = 30
    within_run_vals_logliks.append([
        compute_val_loglik(current_beta, m, true_data_mean, batch_size) for m
        in np.linspace(-3, true_data_mean + 3, num_within_run_graphs)])


    print 'UPDATING BETA...\n    beta PRE: {}'.format(current_beta)
    print '    beta POST: {}'.format(updated_beta)
    print 'UPDATING MU...\n    mu PRE: {}'.format(current_mu)
    print '    mu POST: {}'.format(updated_mu)
    return current_beta, current_mu, optimal_vals_logliks_mus, \
           within_run_vals_logliks


def graph_results(mu_0, true_data_mean, current_beta, current_mu, data_size,
                  x, global_iteration, file_timestamp):
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
        global_iteration: Iteration, passed for filenaming.
        file_timestamp: Datetime used for timestamping all figure outputs.

    Returns:
        None
    """
    # Plotting settings.
    plt.style.use('ggplot')
    fig = plt.figure()
    fig.suptitle(
        'Fixed-scale Univariate Normal Generator, Logistic '
        'Discriminator:\nIteration {}'.format(global_iteration), size=14)

    # Histogram and density of TRUE data.
    ax1 = fig.add_subplot(211)
    count_x, bins_x, ignored_x = plt.hist(x, 30, normed=True, color=COLORS[
        'lightgreen'], label='Data')
    plt.plot(bins_x, 1 / (1 * np.sqrt(2 * np.pi)) *
             np.exp(- (bins_x - true_data_mean) ** 2 / (2 * 1 ** 2)),
             linewidth=1, color=COLORS['lightgreen'])
    # Histogram and density of NOISE data.
    z = np.random.normal(0, 1, data_size)
    count_z, bins_z, ignored_z = plt.hist(z, 30, normed=True,
                                          color=COLORS['blue'], label='Noise')
    plt.plot(bins_z, 1 / (1 * np.sqrt(2 * np.pi)) *
             np.exp(- (bins_z - mu_0) ** 2 / (2 * 1 ** 2)),
             linewidth=1, color=COLORS['blue'])
    ax1.legend(loc='upper left')

    ax2 = fig.add_subplot(212, sharex=ax1)

    _, bins_x, _ = ax2.hist(x, 30, normed=True, color=COLORS['lightgreen'],
                            label='Data')
    plt.plot(bins_x, 1 / (1 * np.sqrt(2 * np.pi)) *
             np.exp(- (bins_x - true_data_mean) ** 2 / (2 * 1 ** 2)),
             linewidth=1, color=COLORS['lightgreen'])

    # Histogram and density of GENERATED data.
    generated_x = np.random.normal(current_mu, 1, data_size)
    _, bins_generated_x, _ = ax2.hist(generated_x, 30, normed=True,
                                      color=COLORS['blue'], label='Generated')
    plt.plot(bins_generated_x, 1 / (1 * np.sqrt(2 * np.pi)) *
             np.exp(- (bins_generated_x - current_mu) ** 2 / (2 * 1 ** 2)),
             linewidth=1, color=COLORS['blue'])

    # Discriminator.
    x_space = np.linspace(- 5, true_data_mean + 5, 50)
    plt.plot(x_space,
             1 / (1 + np.exp(-1 * (current_beta[0] + current_beta[1]
                                   * x_space))), color=COLORS['black'],
             label='Discriminator')

    ax2.legend(loc='upper left')
    plt.savefig('{}-fig{}.png'.format(file_timestamp, global_iteration))


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
    plt.plot(traceplot_mu, color=COLORS['blue'])
    plt.axhline(y=true_data_mean, xmin=0, xmax=1, hold=None,
                color='gray', label='True mean')
    ax1.legend(loc='upper left')

    ax2 = plt.subplot(312)
    plt.title(r'Mu Steps', size=10)
    plt.plot(np.diff(traceplot_mu), color=COLORS['blue'])

    ax3 = plt.subplot(313)
    plt.title(r'Beta', size=10)
    plt.plot(np.array(traceplot_beta)[:, 0], color=COLORS['blue'],
             label='b0')
    plt.plot(np.array(traceplot_beta)[:, 1], color=COLORS['black'],
             label='b1')
    ax3.legend(loc='upper left')

    plt.savefig('{}-traceplot'.format(file_timestamp))
    if display:
        plt.show()


def graph_vals_logliks(optimal_vals_logliks_mus, within_run_vals_logliks):
    """
    For each global iteration, plot the value function value and the log
    likelihood of the generated data, for that iteration's mu-beta pair.
    Also, within each iteration, plot the value-loglikelihood values for the
    current beta, along a grid of mu values.

    Args:
        optimal_vals_logliks: Array of values, log likelihoods, and mus for
        each (mu,
          beta) pair.
        within_run_vals_logliks: Array of arrays, where each element is a
        list of value-loglikelihood pairs, and current_mu, for a grid of mus
          and a fixed beta, for one global iteration.

    Returns:
        None
    """
    # Plot single graph for optimal mu-beta pairs.
    plt.style.use('ggplot')
    fig = plt.figure()
    fig.suptitle('Value vs. Log Likelihood for Generated Data Against True '
                 'Data Distribution: Optimal Pairs', size=14)
    ax = fig.add_subplot(111)
    vals = [v for (v, l, m) in optimal_vals_logliks_mus]
    logliks = [l for (v, l, m) in optimal_vals_logliks_mus]
    mus = [m for (v, l, m) in optimal_vals_logliks_mus]
    t = np.arange(len(vals))
    sc = ax.scatter(vals, logliks, c=t, cmap='cool', s=50)
    for i, txt in enumerate(mus):
        ax.annotate(round(txt, 2), (vals[i], logliks[i]))
    cb = plt.colorbar(sc)
    cb.set_label('Iter', labelpad=-31, y=1.05, rotation=0)
    plt.xlabel('Value')
    plt.ylabel('Log Likelihood')

    # Plot within-run, gridded, mu-beta graphs for each iteration.
    fig = plt.figure()
    fig.suptitle('Value vs. Log Likelihood: Gridded Pairs, Per Iteration',
                 size=14)
    num_runs = len(optimal_vals_logliks_mus)
    dims = np.ceil(np.sqrt(num_runs))
    for run_index in range(num_runs):
        optimal_val = optimal_vals_logliks_mus[run_index][0]
        optimal_loglik = optimal_vals_logliks_mus[run_index][1]
        optimal_mu = optimal_vals_logliks_mus[run_index][2]

        ax = plt.subplot(dims, dims, run_index + 1)
        plt.title(r'Iter: {}'.format(run_index), size=10)
        vals_logliks = within_run_vals_logliks[run_index]
        vals = [v for (v, l) in vals_logliks]
        logliks = [l for (v, l) in vals_logliks]
        ax.scatter(vals, logliks, c=COLORS['lightgreen'], s=30)
        ax.scatter(optimal_val, optimal_loglik, c='red', s=50)
        ax.annotate(round(optimal_mu, 2), (optimal_val, optimal_loglik))
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        if run_index == 0:
            plt.xlabel('Value')
            plt.ylabel('Log Likelihood')
            plt.setp(ax.get_xticklabels(), visible=True)
            plt.setp(ax.get_yticklabels(), visible=False)


def sample_adjacency_matrix_from_SBM(true_SBM, loops=False, directed=False):
    """
    Samples an adjacency matrix from a stochastic block model with a certain
    cluster affinity matrix and cluster distribution.

    Args:
      true_SBM: NamedTuple, with fields B, PI, K, and n, which represent
        cluster affinity matrix, cluster distribution, number of clusters, and
        number of nodes.
      loops: Boolean for whether or not to allow self-links (i.e. non-zero on
        diagonal of adjacency matrix).
      directed: Boolean for whether or not the graph has directed edges.

    Returns:
      adjacency_matrix: A 2D array, representing the link network between the
        num_nodes nodes.
    """
    B = true_SBM.affinity
    PI = true_SBM.distribution
    K = true_SBM.num_clusters
    n = true_SBM.num_nodes

    if len(B) != len(PI):
        raise ValueError('Dimensions of cluster affinity matrix and cluster '
                         'distributions do no match.')
    if not np.isclose(sum(np.absolute(PI)), 1.0, atol=1e-5):
        raise ValueError('Cluster distributions must be on [0, 1] and sum to '
                         '1.')

    assignments = np.random.choice(K, n, replace=True, p=PI)

    adjacency_matrix = []
    for row_index in range(n):
        adjacency_matrix_row = []
        for col_index in range(n):
            class_i = assignments[row_index]
            class_j = assignments[col_index]
            ij_prob = B[class_i][class_j]
            ij_link = rbern(ij_prob)
            adjacency_matrix_row.append(ij_link)
        adjacency_matrix.append(adjacency_matrix_row)

    if not directed:
        adjacency_matrix = sym_matrix(adjacency_matrix)
    if not loops:
        np.fill_diagonal(adjacency_matrix, 0)
    return adjacency_matrix


def sample_random_adjacency_matrix(num_nodes):
    """
    Samples a random network, with zero-diagonal, and symmetric links.

    Args:
      num_nodes: Number of nodes in each network.

    Returns:
      rand_adjacency_matrix: A 2D array of links among num_nodes nodes.
    """
    rand_adjacency_matrix = sym_matrix(
        [[rbern(0.5) for _ in range(num_nodes)] for _ in range(num_nodes)])
    np.fill_diagonal(rand_adjacency_matrix, 0)
    return rand_adjacency_matrix


def initialize_data_with_labels(num_nodes):
    """
    Creates a balanced, shuffled, labeled data set of rows from adjacency
    matrices from two network models: a known SBM and a random network.

    Args:
      num_nodes: Number of nodes in each network.

    Returns:
      chunk: Named tuple, with a chunk of data (adjacency matrix rows) and their
        associated labels.
    """
    SBM_info = namedtuple('SBM_info', ['affinity', 'distribution',
                                       'num_clusters', 'num_nodes'])
    affinity = [[0.9, 0.1, 0.1],
                [0.1, 0.9, 0.1],
                [0.1, 0.1, 0.9]]
    distribution = np.asarray([0.8, 0.1, 0.1])
    num_clusters = len(distribution)
    verbose = 0
    if verbose:
        print ('Affinity:\n{}\nDistribution: {}\nnum_clusters: {}\nnum_nodes: {'
               '}\n\n').format(affinity, distribution, num_clusters, num_nodes)

    true_SBM = SBM_info(affinity, distribution, num_clusters, num_nodes)
    true_adjacency_matrix = sample_adjacency_matrix_from_SBM(true_SBM)
    rand_adjacency_matrix = sample_random_adjacency_matrix(num_nodes)

    display = False
    if display:
        fig, axes = plt.subplots(2, 1)
        axes.flatten()
        axes[0].imshow(true_adjacency_matrix, interpolation='none', cmap='GnBu')
        axes[0].set_title('True SBM')
        axes[1].imshow(rand_adjacency_matrix, interpolation='none', cmap='GnBu')
        axes[1].set_title('Random Undirected Linkages')
        plt.show()

    true_rows_with_labels = np.array(
        [[row, np.array([0., 1.])] for row in true_adjacency_matrix])
    rand_rows_with_labels = np.array(
        [[row, np.array([1., 0.])] for row in rand_adjacency_matrix])
    data_with_labels = np.vstack((true_rows_with_labels, rand_rows_with_labels))
    np.random.shuffle(data_with_labels)

    DataLabels = namedtuple('DataLabels', ['data', 'labels'])
    chunk = DataLabels(np.vstack(data_with_labels[:, 0]),
                       np.vstack(data_with_labels[:, 1]))
    return chunk


def main():
    start = time.time()
    file_timestamp = '{:%Y%m%d-%H%M%S}'.format(datetime.datetime.now())

    data_samples = []
    label_samples = []
    for _ in range(2500):
        train_data_with_labels = initialize_data_with_labels(num_nodes=10)
        data_samples.append(train_data_with_labels.data)
        label_samples.append(train_data_with_labels.labels)
    DataLabels = namedtuple('DataLabels', ['data', 'labels'])
    train_data_with_labels = DataLabels(np.vstack(data_samples),
                                        np.vstack(label_samples))

    data_samples = []
    label_samples = []
    for _ in range(500):
        test_data_with_labels = initialize_data_with_labels(num_nodes=10)
        data_samples.append(test_data_with_labels.data)
        label_samples.append(test_data_with_labels.labels)
    DataLabels = namedtuple('DataLabels', ['data', 'labels'])
    test_data_with_labels = DataLabels(np.vstack(data_samples),
                                        np.vstack(label_samples))

    TrainTestData = namedtuple('TrainTestData', ['train', 'test'])
    data = TrainTestData(train_data_with_labels, test_data_with_labels)

    run_binary_classifier_nn(data)
    print '\n\nTime Elapsed: {}'.format(time.time() - start)

    #discriminator = train_binary_classifier_nn(true_adjacency_matrix,
    #                                           rand_adjacency_matrix)

    # Adjustable procedural parameters.
    # true_data_mean = 6
    # data_size = 1000
    # batch_size = 20
    # num_training_iterations = 1
    # max_k_updates = 1000
    # max_mu_updates = 1
    # display = True
    #
    # # Full data, and initial parameters.
    # x = np.random.normal(true_data_mean, 1, data_size)
    # beta_0, mu_0 = initialize_model_params()
    # current_beta, current_mu = beta_0, mu_0
    # traceplot_beta = []
    # traceplot_mu = []
    # vals_logliks = []
    # within_run_vals_logliks = []

    # for global_iteration in range(num_training_iterations):
    #     print '\n\n-----Global learning iter {}-----'.format(global_iteration)
    #
    #     # graph_results(mu_0, true_data_mean, current_beta, current_mu, data_size,
    #     #               x, global_iteration, file_timestamp)
    #
    #     current_beta, current_mu, vals_logliks, within_run_vals_logliks = \
    #         run_one_learning_iteration(current_beta, current_mu, batch_size,
    #                                    true_data_mean, max_k_updates,
    #                                    max_mu_updates, global_iteration,
    #                                    vals_logliks, within_run_vals_logliks)
    #     traceplot_beta.append(current_beta)
    #     traceplot_mu.append(current_mu)
    #
    # graph_traceplots(true_data_mean, traceplot_beta, traceplot_mu,
    #                  file_timestamp)
    # graph_vals_logliks(vals_logliks, within_run_vals_logliks)
    #
    # print '\n\nTime Elapsed: {}'.format(time.time() - start)
    # if display:
    #     plt.show(block=True)

if __name__ == "__main__":
    main()