from manimlib.imports import *
import os
import pyclbr
import numpy as np
import sklearn.decomposition as skld
from fixedpointfinder.three_bit_flip_flop import Flipflopper

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ThreebitflipflopAnim(ThreeDScene):
    CONFIG = {
    "plane_kwargs" : {
        "color" : RED_B
        },
    "point_charge_loc" : 0.5*RIGHT-1.5*UP,
    }

    def setup(self):
        rnn_type = 'vanilla'
        n_hidden = 24

        flopper = Flipflopper(rnn_type=rnn_type, n_hidden=n_hidden)
        stim = flopper.generate_flipflop_trials()
        # train the model
        # flopper.train(stim, 4000, save_model=True)

        # if a trained model has been saved, it may also be loaded
        flopper.load_model()

        self.weights = flopper.model.get_layer(flopper.hps['rnn_type']).get_weights()
        self.output_weights = flopper.model.get_layer('dense').get_weights()
        # self.activations = np.vstack(flopper.get_activations(stim))
        self.outputs = np.vstack(stim['output'])
        self.activations = self.outputs @ self.output_weights[0].T

    @staticmethod
    def translate_to_complex(x, c_inv):
        return x @ c_inv

    @staticmethod
    def translate_to_real(x, c):
        return x @ c

    def serialize_recurrent_layer(self):
        evals, evecs = np.linalg.eig(self.weights[1])
        real_parts = evals.real
        img_parts = evals.imag
        evecs_c = np.real(evecs)
        reconstructed_matrices = []
        i = 0
        stretch_or_rotate = []
        for k in range((len(self.weights[1]) - np.sum(img_parts > 0))):

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

            reconstructed_matrices.append(diagonal_evals)

        return reconstructed_matrices, evecs_c, stretch_or_rotate

    def transform_recurrent_layer(self):

        evals, evecs = np.linalg.eig(self.weights[1])
        diagonal_evals = np.real(np.diag(evals))
        # real_parts = evals.real
        img_parts = evals.imag
        evecs_c = np.real(evecs)

        for i in range(len(self.weights[1])):
            if img_parts[i] > 0:
                diagonal_evals[i, i+1] = img_parts[i]
                diagonal_evals[i+1, i] = img_parts[i+1]

                evecs_c[:, i] = np.real(evecs[:, i])
                evecs_c[:, i+1] = np.imag(evecs[:, i])

        return diagonal_evals, evecs_c


    def translate_to_diagonal(self):
        evals, evecs = np.linalg.eig(self.weights[1])
        diagonal_evals = np.real(np.diag(evals))
        real_parts = evals.real
        img_parts = evals.imag
        evecs_c = np.real(evecs)
        for i in range(len(self.weights[1])):
            if img_parts > 0:
                diagonal_evals[i, i + 1] = img_parts[i]
                diagonal_evals[i + 1, i] = img_parts[i + 1]

                evecs_c[:, i] = np.real(evecs[:, i])
                evecs_c[:, i+1] = np.imag(evecs[:, i])

        return diagonal_evals, evecs_c

    def construct(self):

        diagonal_evals, c, stretch = self.serialize_recurrent_layer()
        c_inv = np.linalg.inv(c)
        all_diagonal_evals, _ = self.transform_recurrent_layer()

        pca_activations = skld.PCA(3)

        complex_activations = self.translate_to_complex(self.activations, 1 / 5 * c_inv)
        complex_activations = complex_activations @ all_diagonal_evals
        print(complex_activations.shape)
        self.activations = np.vstack(self.activations)
        activations_transformed = pca_activations.fit_transform(complex_activations)

        vector = Vector(activations_transformed[0, :])
        vector.set_color(RED_B)
        old_point = activations_transformed[0, :]

        for t in range(200):
            activation_shape = Line(activations_transformed[t, :], activations_transformed[t + 1, :])
            activation_shape.set_color(BLUE)
            self.add(activation_shape)
        # self.set_camera_orientation(phi=PI / 3, gamma=PI / 5)

        for timestep in range(10):

            fp_vector = vector
            self.add(fp_vector)
            imaginary_activations = complex_activations[timestep, :]

            for rotation in range(len(diagonal_evals)):
                single_transformation = imaginary_activations @ diagonal_evals[rotation]
                imaginary_activations = complex_activations[timestep, :] + single_transformation

                end_point = pca_activations.transform(imaginary_activations.reshape(1, -1))
                end_point = end_point[0]

                if stretch[rotation]:
                    new_vector = Vector(end_point, color=RED_B)

                    self.play(Transform(vector, new_vector), run_time=0.5)
                    self.remove(vector)
                    old_point = end_point
                    vector = new_vector
                else:
                    new_vector = Vector(end_point, color=RED_B)
                    angle = angle_between_vectors(old_point, end_point)

                    self.play(Rotate(vector, angle, ORIGIN), run_time=0.5)
                    self.remove(vector)

                    old_point = end_point
                    vector = new_vector