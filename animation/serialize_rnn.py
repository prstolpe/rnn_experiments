import numpy as np


class SerializedVanilla:

    def __init__(self, recurrent_weights, n_hidden):
        self.n_hidden = n_hidden
        self.reconstructed_diagonals, self.evecs_c, self.stretch_or_rotate = self.serialize_recurrent_layer(recurrent_weights, n_hidden)

        self.evecs_c_inv = np.linalg.inv(self.evecs_c)

        self.all_diagonal, _ = self.transform_recurrent_layer(recurrent_weights)

    @staticmethod
    def translate_to_complex(x, c_inv):
        return x @ c_inv

    @staticmethod
    def translate_to_real(x, c):
        return x @ c

    @staticmethod
    def serialize_recurrent_layer(weights, n_hidden):

        evals, evecs = np.linalg.eig(weights)
        real_parts = evals.real
        img_parts = evals.imag
        evecs_c = np.real(evecs)
        reconstructed_diagonals = []
        i = 0
        stretch_or_rotate = []
        for k in range((len(weights) - np.sum(img_parts > 0))):

            if img_parts[i] > 0:
                diagonal_evals = np.zeros((n_hidden, n_hidden))
                diagonal_evals[i, i + 1] = img_parts[i]
                diagonal_evals[i + 1, i] = img_parts[i + 1]
                diagonal_evals[i, i] = real_parts[i]
                diagonal_evals[i + 1, i + 1] = real_parts[i + 1]
                evecs_c[:, i] = np.real(evecs[:, i])
                evecs_c[:, i + 1] = np.imag(evecs[:, i])
                i += 2
                stretch_or_rotate.append(False)
            elif img_parts[i] < 0:
                pass
            else:
                stretch_or_rotate.append(True)
                diagonal_evals = np.zeros((n_hidden, n_hidden))
                diagonal_evals[i, i] = real_parts[i]
                i += 1

            reconstructed_diagonals.append(diagonal_evals)

        return reconstructed_diagonals, evecs_c, stretch_or_rotate

    @staticmethod
    def transform_recurrent_layer(weights):

        evals, evecs = np.linalg.eig(weights)
        diagonal_evals = np.real(np.diag(evals))
        # real_parts = evals.real
        img_parts = evals.imag
        evecs_c = np.real(evecs)

        for i in range(len(weights)):
            if img_parts[i] > 0:
                diagonal_evals[i, i + 1] = img_parts[i]
                diagonal_evals[i + 1, i] = img_parts[i + 1]

                evecs_c[:, i] = np.real(evecs[:, i])
                evecs_c[:, i + 1] = np.imag(evecs[:, i])

        return diagonal_evals, evecs_c


class SerializedGru:

    def __init__(self, recurrent_weights, n_hidden, trans_factor):
        self.n_hidden = n_hidden
        self.trans_factor = trans_factor
        self.U_z, self.U_r, self.U_h, self.W_z, self.W_r, self.W_h, \
            self.b_z, self.b_r, self.b_h = self.split_weights(recurrent_weights)
        self.serialized = {"z_update": self.serialize_recurrent_layer(self.U_z, self.n_hidden),
                           "r_reset": self.serialize_recurrent_layer(self.U_r, self.n_hidden),
                           "h_activation": self.serialize_recurrent_layer(self.U_h, self.n_hidden)}

        # evals, _ = np.linalg.eig(self.U_h)
        # print(evals)

        self.n_evals = []
        for key in self.serialized:
            self.n_evals.append(len(self.serialized[key][0]))
        self.max_evals = np.max(self.n_evals)

        self.z_c_inv = np.linalg.inv(self.serialized["z_update"][1])
        self.r_c_inv = np.linalg.inv(self.serialized["r_reset"][1])
        self.h_c_inv = np.linalg.inv(self.serialized["h_activation"][1])

        self.b_z = self.translate_to_complex(self.b_z, self.z_c_inv)
        self.b_r = self.translate_to_complex(self.b_r, self.r_c_inv)
        self.b_h = self.translate_to_complex(self.b_h, self.h_c_inv)

    def split_weights(self, weights):
        z, r, h = np.arange(0, self.n_hidden), np.arange(self.n_hidden, 2 * self.n_hidden), \
                  np.arange(2 * self.n_hidden, 3 * self.n_hidden)
        W_z, W_r, W_h = weights[0][:, z], weights[0][:, r], weights[0][:, h]
        U_z, U_r, U_h = weights[1][:, z], weights[1][:, r], weights[1][:, h]
        b_z, b_r, b_h = weights[2][0, z], weights[2][0, r], weights[2][0, h]

        return U_z, U_r, U_h, W_z, W_r, W_h, b_z, b_r, b_h

    def translate_to_complex(self, x, c):
        return x @ (self.trans_factor * c)

    def handle_inputs(self, inputs):
        r_input = inputs @ self.W_r
        z_input = inputs @ self.W_z
        h_input = inputs @ self.W_h

        r_input_complex = r_input @ (self.trans_factor * self.r_c_inv)
        z_input_complex = z_input @ (self.trans_factor * self.z_c_inv)
        h_input_complex = h_input @ (self.trans_factor * self.h_c_inv)

        self.r_input_serial = self.serialize_inputs(self.U_r, r_input_complex)
        self.z_input_serial = self.serialize_inputs(self.U_z, z_input_complex)
        self.h_input_serial = self.serialize_inputs(self.U_h, h_input_complex)

    @staticmethod
    def serialize_recurrent_layer(weights, n_hidden):
        evals, evecs = np.linalg.eig(weights)
        real_parts = evals.real
        img_parts = evals.imag
        evecs_c = np.real(evecs)
        reconstructed_diagonals = []
        i = 0
        stretch_or_rotate = []
        for k in range((n_hidden - np.sum(img_parts > 0))):

            if img_parts[i] > 0:
                diagonal_evals = np.zeros((n_hidden, n_hidden))
                diagonal_evals[i, i + 1] = img_parts[i]
                diagonal_evals[i + 1, i] = img_parts[i + 1]
                diagonal_evals[i, i] = real_parts[i]
                diagonal_evals[i + 1, i + 1] = real_parts[i + 1]
                evecs_c[:, i] = np.real(evecs[:, i])
                evecs_c[:, i + 1] = np.imag(evecs[:, i])

                i += 2
                stretch_or_rotate.append(False)
            elif img_parts[i] < 0:
                pass

            else:
                stretch_or_rotate.append(True)
                diagonal_evals = np.zeros((n_hidden, n_hidden))
                diagonal_evals[i, i] = real_parts[i]

                i += 1

            reconstructed_diagonals.append(diagonal_evals)
        # return list with list of diagonals, evecs_complex matrix, boolean list whether eval said stretch or rotation
        return [reconstructed_diagonals, evecs_c, stretch_or_rotate]

    # serialize input
    @staticmethod
    def serialize_inputs(weights, complex_inputs):
        evals, evecs = np.linalg.eig(weights)
        real_parts = evals.real
        img_parts = np.imag(evals)
        complexgreaterzero = np.sum(img_parts > 0)
        # print(complexgreaterzero)
        serialized_inputs = []
        lh = 0
        for k in range((len(weights) - complexgreaterzero)):
            if img_parts[lh] > 0:
                input = np.zeros((len(complex_inputs), len(weights)))
                input[:, lh] = complex_inputs[:, lh]
                input[:, lh + 1] = complex_inputs[:, lh + 1]

                lh += 2

            elif img_parts[lh] < 0:
                pass
            else:
                input = np.zeros((len(complex_inputs), len(weights)))
                input[:, lh] = complex_inputs[:, lh]

                lh += 1

            serialized_inputs.append(input)

        return serialized_inputs

    @staticmethod
    def transform_recurrent_layer(weights):

        evals, evecs = np.linalg.eig(weights)
        diagonal_evals = np.real(np.diag(evals))
        # real_parts = evals.real
        img_parts = evals.imag
        evecs_c = np.real(evecs)

        for i in range(len(weights)):
            if img_parts[i] > 0:
                diagonal_evals[i, i + 1] = img_parts[i]
                diagonal_evals[i + 1, i] = img_parts[i + 1]

                evecs_c[:, i] = np.real(evecs[:, i])
                evecs_c[:, i + 1] = np.imag(evecs[:, i])

        return diagonal_evals, evecs_c






