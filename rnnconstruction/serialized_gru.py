import numpy as np

class SerializedGru:

    def __init__(self, recurrent_weights, n_hidden):
        self.n_hidden = n_hidden
        self.U_z, self.U_r, self.U_h, self.W_z, self.W_r, self.W_h = self.split_weights(recurrent_weights)
        self.serialized = []

        self.serialized.append(self.serialize_recurrent_layer(self.U_z))
        self.serialized.append(self.serialize_recurrent_layer(self.U_r))
        self.serialized.append(self.serialize_recurrent_layer(self.U_h))

        self.z_c_inv = np.linalg.inv(self.serialized[0][1])
        self.r_c_inv = np.linalg.inv(self.serialized[1][1])
        self.h_c_inv = np.linalg.inv(self.serialized[2][1])

    def split_weights(self, weights):
        z, r, h = np.arange(0, self.n_hidden), np.arange(self.n_hidden, 2 * self.n_hidden), \
                  np.arange(2 * self.n_hidden, 3 * self.n_hidden)
        W_z, W_r, W_h = weights[0][:, z], weights[0][:, r], weights[0][:, h]
        U_z, U_r, U_h = weights[1][:, z], weights[1][:, r], weights[1][:, h]
        # b_z, b_r, b_h = weights[2][0, z], weights[2][0, r], weights[2][0, h]

        return U_z, U_r, U_h, W_z, W_r, W_h

    def translate_to_complex(self):
        pass

    @staticmethod
    def serialize_recurrent_layer(weights):
        evals, evecs = np.linalg.eig(weights)
        real_parts = evals.real
        img_parts = evals.imag
        evecs_c = np.real(evecs)
        reconstructed_diagonals = []
        i = 0
        stretch_or_rotate = []
        for k in range((len(weights) - np.sum(img_parts > 0))):

            if img_parts[i] > 0:
                diagonal_evals = np.zeros((24, 24))
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
                diagonal_evals = np.zeros((24, 24))
                diagonal_evals[i, i] = real_parts[i]
                i += 1

            reconstructed_diagonals.append(diagonal_evals)

        return [reconstructed_diagonals, evecs_c, stretch_or_rotate]


