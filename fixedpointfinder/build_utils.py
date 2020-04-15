import autograd.numpy as np
import numdifftools as nd

# TODO: could also be turned into class object


class DynamicalSystemsBuilder(object):

    def __init__(self, recurrent_layer_weights, n_recurrent_units,
                 to_recurrent_layer_weights=None, from_recurrent_layer_weights=None):

        self.recurrent_layer_weights = recurrent_layer_weights
        self.n_recurrent_units = n_recurrent_units
        self.to_recurrent_layer_weights = to_recurrent_layer_weights
        self.from_recurrent_layer_weights = from_recurrent_layer_weights


    def build_ds(self, inputs):
        pass


class RnnDsBuilder(DynamicalSystemsBuilder):

    def __init__(self, recurrent_layer_weights, n_recurrent_units,
                 to_recurrent_layer_weights=None, from_recurrent_layer_weights=None):

        super().__init__(recurrent_layer_weights, n_recurrent_units,
                 to_recurrent_layer_weights, from_recurrent_layer_weights)

        self.weights, self.inputweights, self.b = recurrent_layer_weights[1], recurrent_layer_weights[0], \
                                   recurrent_layer_weights[2]
        self.n_hidden = n_recurrent_units

    def build_sequential_ds(self, inputs):
        def fun(x):
            return 1 / self.n_hidden * np.sum((- x + np.tanh(x @ self.weights + inputs @ self.inputweights + self.b)) ** 2)

        return fun

    def build_joint_ds(self, inputs):
        def fun(x):
            return np.mean(
                1 / self.n_hidden * np.sum(((- x + np.tanh(x @ self.weights + inputs @ self.inputweights + self.b)) ** 2), axis=1))

        return fun

    def build_velocity_fun(self, inputs):
        def fun(x):
            return 0.5 * np.sum(((- x + np.tanh(x @ self.weights + inputs @ self.inputweights + self.b)) ** 2), axis=1)

        return fun

    def build_jacobian_fun(self):
        jac_fun = lambda x: - np.eye(self.n_hidden, self.n_hidden) + self.weights * (1 - np.tanh(x) ** 2)

        return jac_fun


class GruDsBuilder(DynamicalSystemsBuilder):

    def __init__(self, recurrent_layer_weights, n_recurrent_units,
                 to_recurrent_layer_weights=None, from_recurrent_layer_weights=None):

        super().__init__(recurrent_layer_weights, n_recurrent_units,
                         to_recurrent_layer_weights, from_recurrent_layer_weights)

        weights, self.n_hidden = self.recurrent_layer_weights, n_recurrent_units

        z, r, h = np.arange(0, self.n_hidden), np.arange(self.n_hidden, 2 * self.n_hidden), \
                  np.arange(2 * self.n_hidden, 3 * self.n_hidden)
        self.W_z, self.W_r, self.W_h = weights[0][:, z], weights[0][:, r], weights[0][:, h]
        self.U_z, self.U_r, self.U_h = weights[1][:, z], weights[1][:, r], weights[1][:, h]
        self.b_z, self.b_r, self.b_h = weights[2][0, z], weights[2][0, r], weights[2][0, h]

    def build_joint_ds(self, input):

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        z_projection_b = input @ self.W_z + self.b_z
        r_projection_b = input @ self.W_r + self.b_r
        g_projection_b = input @ self.W_h + self.b_h

        z_fun = lambda x: sigmoid(x @ self.U_z + z_projection_b)
        r_fun = lambda x: sigmoid(x @ self.U_r + r_projection_b)
        g_fun = lambda x: np.tanh(r_fun(x) * (x @ self.U_h) + g_projection_b)

        def fun(x):
            return np.mean(1 / self.n_hidden * np.sum(((- x + z_fun(x) * x + (1 - z_fun(x)) * g_fun(x)) ** 2), axis=1))

        return fun

    def build_sequential_ds(self, input):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        z_projection_b = input @ self.W_z + self.b_z
        r_projection_b = input @ self.W_r + self.b_r
        g_projection_b = input @ self.W_h + self.b_h

        z_fun = lambda x: sigmoid(x @ self.U_z + z_projection_b)
        r_fun = lambda x: sigmoid(x @ self.U_r + r_projection_b)
        g_fun = lambda x: np.tanh(r_fun(x) * (x @ self.U_h) + g_projection_b)

        fun = lambda x: 1/self.n_hidden * np.sum((- x + (z_fun(x) * x) + ((1 - z_fun(x)) * g_fun(x))) ** 2)

        return fun

    def build_velocity_fun(self, input):

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        z_projection_b = input @ self.W_z + self.b_z
        r_projection_b = input @ self.W_r + self.b_r
        g_projection_b = input @ self.W_h + self.b_h

        z_fun = lambda x: sigmoid(x @ self.U_z + z_projection_b)
        r_fun = lambda x: sigmoid(x @ self.U_r + r_projection_b)
        g_fun = lambda x: np.tanh(r_fun(x) * (x @ self.U_h) + g_projection_b)

        fun = lambda x: 1/self.n_hidden * np.sum(((- x + z_fun(x) * x + (1 - z_fun(x)) * g_fun(x)) ** 2), axis=1)

        return fun

    def build_jacobian_fun(self, input):

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        z_projection_b = input @ self.W_z + self.b_z
        r_projection_b = input @ self.W_r + self.b_r
        g_projection_b = input @ self.W_h + self.b_h

        z_fun = lambda x: sigmoid(x @ self.U_z + z_projection_b)
        r_fun = lambda x: sigmoid(x @ self.U_r + r_projection_b)
        g_fun = lambda x: np.tanh(r_fun(x) * (x @ self.U_h) + g_projection_b)

        def dynamical_system(x):
            return - x + z_fun(x) * x + (1 - z_fun(x)) * g_fun(x)
        jac_fun = nd.Jacobian(dynamical_system)

        return jac_fun


def build_lstm_ds(self, input, method: str = 'joint'):
    weights, n_hidden = self.recurrent_layer_weights, self.n_recurrent_units
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    W, U, b = weights[0], weights[1], weights[2]

    W_i, W_f, W_c, W_o = W[:, :n_hidden], W[:, n_hidden:2 * n_hidden], \
                         W[:, 2 * n_hidden:3 * n_hidden], W[:, 3 * n_hidden:]
    U_i, U_f, U_c, U_o = U[:, :n_hidden], U[:, n_hidden:2 * n_hidden], \
                         U[:, 2 * n_hidden:3 * n_hidden], U[:, 3 * n_hidden:]
    b_i, b_f, b_c, b_o = b[:n_hidden], b[n_hidden:2 * n_hidden], \
                         b[2 * n_hidden:3 * n_hidden], b[3 * n_hidden:]

    f_projection_b = np.matmul(input, W_f) + b_f
    i_projection_b = np.matmul(input, W_i) + b_i
    o_projection_b = np.matmul(input, W_o) + b_o
    c_projection_b = np.matmul(input, W_c) + b_c

    f_fun = lambda x: sigmoid(np.matmul(x, U_f) + f_projection_b)
    i_fun = lambda x: sigmoid(np.matmul(x, U_i) + i_projection_b)
    o_fun = lambda x: sigmoid(np.matmul(x, U_o) + o_projection_b)
    c_fun = lambda c, h: f_fun(h) * c + i_fun(h) * np.tanh((np.matmul(h, U_c) + c_projection_b))

    if method == 'joint':
        def h_fun(x):
            c, h = x[:, n_hidden:], x[:, :n_hidden]
            return o_fun(h) * np.tanh(c_fun(c, h))

        def cfun(x):
            c, h = x[:, n_hidden:], x[:, :n_hidden]
            return c_fun(c, h)

        def fun(x):
            return np.mean(0.5 * np.sum(np.square(np.hstack((h_fun(x), cfun(x))) - x), axis=1))

    elif method == 'sequential':
        def h_fun(x):
            c, h = x[n_hidden:], x[:n_hidden]
            return o_fun(h) * np.tanh(c_fun(c, h))

        def cfun(x):
            c, h = x[n_hidden:], x[:n_hidden]
            return c_fun(c, h)

        def fun(x):
            return 0.5 * np.sum(np.square(np.hstack((h_fun(x), cfun(x))) - x))

    elif method == 'velocity':
        def h_fun(x):
            c, h = x[:, n_hidden:], x[:, :n_hidden]
            return o_fun(h) * np.tanh(c_fun(c, h))

        def cfun(x):
            c, h = x[:, n_hidden:], x[:, :n_hidden]
            return c_fun(c, h)

        def fun(x):
            return 0.5 * np.sum(np.square(np.hstack((h_fun(x), cfun(x))) - x), axis=1)
    else:
        raise ValueError('Method argument to build function must be one of '
                         '[joint, sequential, velocity] but was', method)

    def dynamical_system(x):
        return np.hstack((h_fun(x), cfun(x))) - x

    jac_fun = nd.Jacobian(dynamical_system)

    return fun, jac_fun

def build_circular_gru_ds(self, input, method: str = 'joint'):
    weights, n_hidden = self.recurrent_layer_weights, self.n_recurrent_units
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    z, r, h = np.arange(0, n_hidden), np.arange(n_hidden, 2 * n_hidden), np.arange(2 * n_hidden, 3 * n_hidden)
    W_z, W_r, W_h = weights[0][:, z], weights[0][:, r], weights[0][:, h]
    U_z, U_r, U_h = weights[1][:, z], weights[1][:, r], weights[1][:, h]
    b_z, b_r, b_h = weights[2][0, z], weights[2][0, r], weights[2][0, h]

    z_projection_b = input @ W_z + b_z
    r_projection_b = input @ W_r + b_r
    g_projection_b = input @ W_h + b_h

    z_fun = lambda x: sigmoid(x @ U_z + z_projection_b)
    r_fun = lambda x: sigmoid(x @ U_r + r_projection_b)
    g_fun = lambda x: np.tanh(r_fun(x) * (x @ U_h) + g_projection_b)

    if method == 'joint':
        def fun(x):
            return np.mean(1/n_hidden * np.sum(((- x + z_fun(x) * x + (1 - z_fun(x)) * g_fun(x)) ** 2), axis=1))
    elif method == 'sequential':
        fun = lambda x: 1/n_hidden * np.sum((- x + (z_fun(x) * x) + ((1 - z_fun(x)) * g_fun(x))) ** 2)
    elif method == 'velocity':
        fun = lambda x: 1/n_hidden * np.sum(((- x + z_fun(x) * x + (1 - z_fun(x)) * g_fun(x)) ** 2), axis=1)
    else:
        raise ValueError('Method argument to build function must be one of '
                     '[joint, sequential, velocity] but was', method)

    def dynamical_system(x):
        return - x + z_fun(x) * x + (1 - z_fun(x)) * g_fun(x)
    jac_fun = nd.Jacobian(dynamical_system)

    return fun, jac_fun

    def build_numpy_submodelto(to_recurrent_layer_weights):
        first_layer = lambda x: np.tanh(x @ to_recurrent_layer_weights[0] + to_recurrent_layer_weights[1])
        second_layer = lambda x: np.tanh(x @ to_recurrent_layer_weights[2] + to_recurrent_layer_weights[3])
        third_layer = lambda x: np.tanh(x @ to_recurrent_layer_weights[4] + to_recurrent_layer_weights[5])
        fourth_layer = lambda x: np.tanh(x @ to_recurrent_layer_weights[6] + to_recurrent_layer_weights[7])

        return first_layer, second_layer, third_layer, fourth_layer

    def build_numpy_submodelfrom(from_recurrent_layer_weights):
        softplus = lambda x: np.log(np.exp(x) + 1)
        alpha_fun = lambda x: softplus(x @ from_recurrent_layer_weights[0] + from_recurrent_layer_weights[1])
        beta_fun = lambda x: softplus(x @ from_recurrent_layer_weights[2] + from_recurrent_layer_weights[3])

        return alpha_fun, beta_fun

    submodelfrom_weights = chiefinvesti.sub_model_from.get_weights()
    alpha_fun, beta_fun = build_numpy_submodelfrom(submodelfrom_weights)
    alphas = alpha_fun(activations_over_all_episodes)
    betas = beta_fun(activations_over_all_episodes)

def act_deterministic(alphas, betas):
    actions = (alphas - 1) / (alphas + betas - 2)

    action_max_values = chiefinvesti.env.action_space.high
    action_min_values = chiefinvesti.env.action_space.low
    action_mm_diff = action_max_values - action_min_values

    actions = np.multiply(actions, action_mm_diff) + action_min_values

    return actions
    actions = act_deterministic(alphas, betas)

def unwrapper(means, variances, states):
    unwrapped_states = []
    for state in states:
        unwrapped_states.append((state * np.sqrt(variances)) + means)

    return np.vstack(unwrapped_states)

    means = chiefinvesti.preprocessor.wrappers[0].mean[0]
    variances = chiefinvesti.preprocessor.wrappers[0].variance[0]
    states_all_episodes = np.vstack(states_all_episodes)
    unwrapped_states = unwrapper(means, variances, states_all_episodes)

    pca = skld.PCA(3)

    transformed_states = pca.fit_transform(unwrapped_states[:100, :])
    transformed_states_wrapped = pca.transform(states_all_episodes[:100, :])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.plot(transformed_states[:, 0], transformed_states[:, 1], transformed_states[:, 2])
    plt.title('unwrapped states')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.plot(transformed_states_wrapped[:, 0], transformed_states_wrapped[:, 1], transformed_states_wrapped[:, 2])
    plt.title('wrapped states')
    plt.show()


