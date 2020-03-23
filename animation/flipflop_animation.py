from manimlib.imports import *
import os
import pyclbr
import numpy as np
import sklearn.decomposition as skld

import sys
sys.path.append("/Users/Raphael/rnn_dynamical_systems")
from fixedpointfinder.three_bit_flip_flop import Flipflopper
from rnnconstruction.serialized_gru import SerializedGru

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

    def construct(self):

        diagonal_evals, c, stretch = self.serialize_recurrent_layer()
        c_inv = np.linalg.inv(c)
        all_diagonal_evals, _ = self.transform_recurrent_layer()

        pca_activations = skld.PCA(3)

        complex_activations = self.translate_to_complex(self.activations, 1 / 5 * c_inv)
        complex_activations = complex_activations @ all_diagonal_evals
        print(complex_activations.shape)
        # self.activations = np.vstack(self.activations)
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


class SerializedGruAnim(ThreeDScene):

    def setup(self):
        rnn_type = 'gru'
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

        self.sGru = SerializedGru(self.weights, n_hidden)

        self.complex_activations = self.activations @ (1/4 * self.sGru.h_c_inv)

    def construct(self):

        pca = skld.PCA(3)

        transformed_activations = pca.fit_transform(self.complex_activations)

        shape = Polygon(*transformed_activations[:200, :], color=BLUE, width=0.1)
        vector = Vector(transformed_activations[0, :], color=RED_B)

        z_vector = (self.complex_activations[0, :] @ self.sGru.serialized[0][0][0]) * self.complex_activations[0, :]
        z_transformed = pca.transform(z_vector.reshape(1,-1))[0]
        z_arrow = Vector(z_transformed, color=GREEN, width=0.1)

        r_vector = self.complex_activations[0, :] @ self.sGru.serialized[0][1][0]
        r_transformed = pca.transform(r_vector.reshape(1, -1))[0]
        r_arrow = Vector(r_transformed, color=YELLOW, width=0.1)

        second_vector = (1-z_vector) * (r_vector * self.complex_activations[0, :]) @ self.sGru.serialized[0][0][0]
        second_transformed = pca.transform(second_vector.reshape(1, -1))[0]
        second_arrow = Arrow(z_arrow, second_transformed, color=YELLOW, width=0.1)

        self.play(ShowCreation(TextMobject("GRU vector computation").to_edge(UP)))
        self.play(ShowCreation(shape),
                  ShowCreation(vector),
                  ShowCreation(z_arrow),
                  ShowCreation(second_arrow))
                  #ShowCreation(r_arrow))

        # TextMobjects
        r_text = TextMobject("reset gate", color=YELLOW)
        z_text = TextMobject("update gate", color=GREEN)
        h_text = TextMobject("activation", color=RED_B)
        r_text.move_to(np.array([5, 0.75, 0]))
        z_text.next_to(r_text, DOWN)
        h_text.next_to(z_text, DOWN)


        self.play(ShowCreation(r_text),
                  ShowCreation(z_text),
                  ShowCreation(h_text))

        n_evals = []
        for i in range(3):
            print(len(self.sGru.serialized[0][i]))
            n_evals.append(len(self.sGru.serialized[0][i]))

        print(np.max(n_evals))
        for timestep in range(20):

            imaginary_activations = self.complex_activations[timestep, :]

            z_iterator = 0
            h_iterator = 0
            for rotation in range(np.max(n_evals)):
                r_vector = imaginary_activations @ self.sGru.serialized[0][1][rotation]
                if z_iterator < 14:
                    z_vector = imaginary_activations @ self.sGru.serialized[0][0][z_iterator]
                    z_iterator += 1
                h_vector = (r_vector * imaginary_activations) @ self.sGru.serialized[0][0][h_iterator]
                if h_iterator < 14:
                    h_iterator += 1
                # transformed_imaginary = pca.transform(imaginary_activations.reshape(1,-1))[0]
                z_transformed = pca.transform(z_vector.reshape(1, -1))[0]
                r_transformed = pca.transform(r_vector.reshape(1, -1))[0]

                new_z_arrow = #Vector(z_transformed, color=GREEN)
                new_r_arrow = #Vector(r_transformed, color=YELLOW)

                imaginary_activations = self.complex_activations[timestep, :] + z_vector * imaginary_activations + \
                                        (1 - z_vector) * h_vector
                # TODO: should vector grow from origin or tip of activation ?
                # TODO: show vectors z_vector*activation and 1-z_vector * h_vector to create vector pointing to tip
                new_point = pca.transform(imaginary_activations.reshape(1,-1))[0]
                new_vector = Vector(new_point, color=RED_B)
                self.play(Transform(z_arrow, new_z_arrow),
                          Transform(r_arrow, new_r_arrow),
                          Transform(vector, new_vector),
                          run_time=0.3)

