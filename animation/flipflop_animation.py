from manimlib.imports import *
import os
import numpy as np
import sklearn.decomposition as skld
import sys
sys.path.append("/Users/Raphael/dexterous-robot-hand")
sys.path.append("/Users/Raphael/dexterous-robot-hand/rnn_dynamical_systems")

from rnn_dynamical_systems.fixedpointfinder.three_bit_flip_flop import Flipflopper, RetrainableFlipflopper
from rnn_dynamical_systems.animation.serialize_rnn import SerializedGru, SerializedVanilla
from rnn_dynamical_systems.fixedpointfinder.FixedPointFinder import Adamfixedpointfinder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class SerializedVanillaAnim(ThreeDScene):

    def setup(self):
        rnn_type = 'vanilla'
        n_hidden = 24

        flopper = Flipflopper(rnn_type=rnn_type, n_hidden=n_hidden)
        stim = flopper.generate_flipflop_trials()
        # flopper.train(stim, 4000, save_model=True)
        # if a trained model has been saved, it may also be loaded
        flopper.load_model()

        self.weights = flopper.model.get_layer(flopper.hps['rnn_type']).get_weights()
        self.output_weights = flopper.model.get_layer('dense').get_weights()
        # self.activations = np.vstack(flopper.get_activations(stim))
        self.outputs = np.vstack(stim['output'])
        self.activations = self.outputs @ self.output_weights[0].T

        self.sVanilla = SerializedVanilla(self.weights[1], n_hidden)

        self.complex_input = (np.vstack(stim['inputs']) @ self.weights[0]) @ (1/6 * self.sVanilla.evecs_c_inv)

    def construct(self):

        pca_activations = skld.PCA(3)

        complex_activations = self.sVanilla.translate_to_complex(self.activations, 1 / 6 * self.sVanilla.evecs_c_inv)
        complex_activations = complex_activations @ self.sVanilla.all_diagonal # move forward one timestep for contraction
        activations_transformed = pca_activations.fit_transform(complex_activations)

        vector = Vector(activations_transformed[0, :], color=RED_B)
        activation_shape = Polygon(*activations_transformed[:500, :], color=BLUE, width=0.1)

        timestep = 0
        timestepscounter = TextMobject("Timestep:" , str(timestep))

        self.play(ShowCreation(TextMobject("Vanilla vector computation").to_edge(UP)),
                  ShowCreation(timestepscounter.to_edge(LEFT)),
                  ShowCreation(activation_shape),
                  ShowCreation(vector))

        for timestep in range(20):

            imaginary_activations = complex_activations[timestep, :]
            imaginary_input = self.complex_input[timestep, :]
            new_timestepscounter = TextMobject("Timestep:" , str(timestep)).to_edge(LEFT)
            self.play(Transform(timestepscounter, new_timestepscounter), run_time=0.4)

            for sub_timestep in range(len(self.sVanilla.reconstructed_diagonals)):
                single_transformation = imaginary_activations @ self.sVanilla.reconstructed_diagonals[sub_timestep]
                input_update = imaginary_input @ self.sVanilla.reconstructed_diagonals[sub_timestep]
                imaginary_activations = complex_activations[timestep, :] + single_transformation + input_update

                end_point = pca_activations.transform(imaginary_activations.reshape(1, -1))[0]

                new_vector = Vector(end_point, color=RED_B)

                self.play(Transform(vector, new_vector), run_time=0.3)



class SerializedGruAnim(ThreeDScene):

    def setup(self):
        """Function to be called by manim similar to _init_. Here the data and serialization of the GRU is set up.
        Activations are converted to complex activations using the inverse of the reordered complex eigenvalue matrix
        derived from the recurrent weight matrix for the activation named h. Additionally that transformation is scaled
         by 1/4. """
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
        self.activations = np.vstack(flopper.get_activations(stim))
        self.outputs = np.vstack(stim['output'])
        # self.activations = self.outputs @ self.output_weights[0].T # this will make for idealized activations

        self.sGru = SerializedGru(self.weights, n_hidden) # helper class to serialize gru

        self.complex_activations = self.activations @ (1/4 * self.sGru.h_c_inv) # transform activations to complex system
        # self.complex_activations = np.vstack((np.zeros(n_hidden), self.complex_activations[:-1, :]))

        inputs = np.vstack(stim['inputs'])
        self.sGru.handle_inputs(inputs)

    def display_gate_text(self):
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

    def construct(self):

        pca = skld.PCA(3)

        transformed_activations = pca.fit_transform(self.complex_activations)

        shape = Polygon(*transformed_activations[:700, :], color=BLUE, width=0.1)
        vector = Vector(transformed_activations[0, :], color=RED_B)

        z_vector = (self.complex_activations[0, :] @ self.sGru.serialized["z_update"][0][0])
        z_transformed = pca.transform(z_vector.reshape(1, -1))[0]
        z_arrow = Vector(z_transformed, color=GREEN, width=0.1)

        r_vector = self.complex_activations[0, :] @ self.sGru.serialized["r_reset"][0][0]
        r_transformed = pca.transform(r_vector.reshape(1, -1))[0]
        r_arrow = Vector(r_transformed, color=YELLOW, width=0.1)

        timestep = 0
        timestepscounter = TextMobject("Timestep:" , str(timestep))

        self.play(ShowCreation(TextMobject("GRU vector computation").to_edge(UP)),
                  ShowCreation(timestepscounter.to_edge(LEFT)))
        self.play(ShowCreation(shape),
                  ShowCreation(vector),
                  ShowCreation(z_arrow),
                  ShowCreation(r_arrow))
        self.display_gate_text()

        for timestep in range(4):

            imaginary_activations = self.complex_activations[timestep, :]
            new_timestepscounter = TextMobject("Timestep:" , str(timestep)).to_edge(LEFT)
            self.play(Transform(timestepscounter, new_timestepscounter), run_time=0.4)

            r_iterator = 0
            h_iterator = 0
            for sub_timestep in range(self.sGru.max_evals):
                z_vector = imaginary_activations @ self.sGru.serialized["z_update"][0][sub_timestep]

                if r_iterator <= self.sGru.n_evals[1]:
                    r_vector = imaginary_activations @ self.sGru.serialized["r_reset"][0][r_iterator]
                else:
                    r_vector = 0

                if r_iterator < (self.sGru.n_evals[1]-1):
                    r_iterator += 1

                h_vector = (r_vector * imaginary_activations) @ self.sGru.serialized["h_activation"][0][h_iterator]

                if h_iterator < (self.sGru.n_evals[2]-1):
                    h_iterator += 1

                z_transformed = pca.transform(z_vector.reshape(1, -1))[0]
                r_transformed = pca.transform(r_vector.reshape(1, -1))[0]

                new_z_arrow = Vector(z_transformed, color=GREEN)
                new_r_arrow = Vector(r_transformed, color=YELLOW)

                imaginary_activations = self.complex_activations[timestep, :] + z_vector * imaginary_activations + \
                                        (1 - z_vector) * h_vector

                new_point = pca.transform(imaginary_activations.reshape(1, -1))[0]
                new_vector = Vector(new_point, color=RED_B)

                self.play(Transform(z_arrow, new_z_arrow),
                          Transform(r_arrow, new_r_arrow),
                          Transform(vector, new_vector),
                          run_time=0.3)


class AnimateFlipFlopLearning(ThreeDScene):

    def setup(self):

        rnn_type = 'vanilla'
        n_hidden = 24

        flopper = RetrainableFlipflopper(rnn_type=rnn_type, n_hidden=n_hidden)
        stim = flopper.generate_flipflop_trials()

        _ = flopper.initial_train(stim, 10, True)
        self.iterations = 40
        self.collected_activations = []
        for i in range(self.iterations):
            self.collected_activations.append(np.vstack(flopper.get_activations(stim)))
            _ = flopper.continued_train(stim, 50, True)

        flopper.load_model()
        weights = flopper.model.get_layer(flopper.hps['rnn_type']).get_weights()
        activations = flopper.get_activations(stim)
        # initialize adam fpf
        fpf = Adamfixedpointfinder(weights, rnn_type,
                                   q_threshold=1e-12,
                                   epsilon=0.01,
                                   alr_decayr=0.0001,
                                   max_iters=7000)
        # sample states, i.e. a number of ICs
        states = fpf.sample_states(activations, 1000, 0.2)
        # generate corresponding input as zeros for flip flop task
        # please keep in mind that the input does not need to be zero for all tasks
        inputs = np.zeros((states.shape[0], 3))
        # find fixed points
        self.fps = fpf.find_fixed_points(states, inputs)
        self.fixedpoint_locations = self.extract_fixed_point_locations(self.fps)
        self.fps, self.x_directions = self.classify_fixedpoints(self.fps, 2)
    @staticmethod
    def extract_fixed_point_locations(fps):
        """Processing of minimisation results for pca. The function takes one fixedpoint object at a time and
        puts all coordinates in single array."""
        fixed_point_location = [fp['x'] for fp in fps]

        fixed_point_locations = np.vstack(fixed_point_location)

        return fixed_point_locations

    @staticmethod
    def classify_fixedpoints(fps, scale):
        """Function to classify fixed points. Methodology is based on
        'Nonlinear Dynamics and Chaos, Strogatz 2015'.

        Args:
            fps: Fixedpointobject containing a set of fixedpoints.
            scale: Float by which the unstable modes shall be scaled for plotting.

        Returns:
            fps: Fixedpointobject that contains 'fp_stability', i.e. information about
            the stability of the fixedpoint
            x_directions: list of matrices containing vectors of unstable modes"""

        x_directions = []
        scale = scale
        for fp in fps:

            trace = np.matrix.trace(fp['jac'])
            det = np.linalg.det(fp['jac'])
            if det > 0 and trace == 0:
                print('center has been found. Watch out for limit cycles')
            elif trace**2 - 4 * det == 0:
                print("star nodes has been found.")
            elif trace**2 - 4 * det < 0:
                print("spiral has been found")
            e_val, e_vecs = np.linalg.eig(fp['jac'])
            ids = np.argwhere(np.real(e_val) > 0)
            countgreaterzero = np.sum(e_val > 0)
            if countgreaterzero == 0:
                print('stable fixed point was found.')
                fp['fp_stability'] = 'stable fixed point'
            elif countgreaterzero > 0:
                print('saddle point was found.')
                fp['fp_stability'] = 'saddle point'
                for id in ids:
                    x_plus = fp['x'] + scale * e_val[id] * np.real(e_vecs[:, id].transpose())
                    x_minus = fp['x'] - scale * e_val[id] * np.real(e_vecs[:, id].transpose())
                    x_direction = np.vstack((x_plus, fp['x'], x_minus))
                    x_directions.append(np.real(x_direction))

        return fps, x_directions

    def construct(self):

        pca = skld.PCA(3)

        pca.fit(self.collected_activations[-1])
        self.set_camera_orientation(phi=60 * DEGREES, theta=-120 * DEGREES)

        transformed_points = pca.transform(self.fixedpoint_locations)
        # set up fixed points
        points = []
        lines = []
        for i in range(len(transformed_points)):
            if self.fps[i]['fp_stability'] == 'stable fixed point':
                points.append(Dot(transformed_points[i, :], color=GREEN, size=0.15))
            elif self.fps[i]['fp_stability'] == 'saddle point':
                points.append(Dot(transformed_points[i, :], color=RED_B, size=0.15))
                for p in range(len(self.x_directions)):
                    direction_matrix = pca.transform(self.x_directions[p])
                    lines.append(Line(direction_matrix[0, :], direction_matrix[2, :], color=RED_B, width=0.2))

        dots = VGroup(*points)
        modes = VGroup(*lines)

        self.play(ShowCreation(dots),
                  ShowCreation(modes))
        self.wait(2)
        for it in range(self.iterations):
            transformed_activations = pca.transform(self.collected_activations[it])

            line = Polygon(*transformed_activations[:300, :],
                           color=BLUE, width=0.05, opacity=0.2)
            self.add(line)
            self.wait(0.5)
            self.remove(line)


