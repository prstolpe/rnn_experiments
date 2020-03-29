# from analysis.chiefinvestigation import Chiefinvestigator
from manimlib.imports import *
import os
import sklearn.decomposition as skld
import sys
sys.path.append("/Users/Raphael//rnn_dynamical_systems")
sys.path.append("/Users/Raphael/dexterous-robot-hand")
from rnn_dynamical_systems.fixedpointfinder.three_bit_flip_flop import Flipflopper
from rnn_dynamical_systems.fixedpointfinder.FixedPointFinder import RecordingFixedpointfinder
from analysis.chiefinvestigation import Chiefinvestigator
from rnn_dynamical_systems.animation.serialize_rnn import SerializedGru, SerializedVanilla

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class AnimateActivitySingleRun(ThreeDScene):

    def setup(self):

        #agent_id = 1583404415  # 1583180664 lunarlandercont
        # agent_id = 1583256614 # reach task
        #chiefinvesti = Chiefinvestigator(agent_id)

        #layer_names = chiefinvesti.get_layer_names()
        #print(layer_names)

        #self.activations_over_episode, self.inputs_over_episode, \
        #self.actions_over_episode = chiefinvesti.get_data_over_single_run("policy_recurrent_layer",
         #                                                                 layer_names[1])

        rnn_type = 'vanilla'
        n_hidden = 24

        flopper = Flipflopper(rnn_type=rnn_type, n_hidden=n_hidden)
        stim = flopper.generate_flipflop_trials()
        # train the model
        flopper.train(stim, 4000, save_model=True)

        # if a trained model has been saved, it may also be loaded
        # flopper.load_model()

        self.weights = flopper.model.get_layer(flopper.hps['rnn_type']).get_weights()
        self.output_weights = flopper.model.get_layer('dense').get_weights()
        self.activations = np.vstack(flopper.get_activations(stim))
        self.outputs = np.vstack(stim['output'])
        # self.activations = self.outputs @ self.output_weights[0].T
        self.inputs = np.vstack(stim['inputs'])


    def construct(self):
        outputs = self.outputs
        def generate_output_mobjects(outputs):
            first_bit = TextMobject(str(outputs[0]))
            first_bit.move_to(np.array([4, 0.75, 0]))
            second_bit = TextMobject(str(outputs[1]))
            second_bit.next_to(first_bit, DOWN)
            third_bit = TextMobject(str(outputs[2]))
            third_bit.next_to(second_bit, DOWN)

            return first_bit, second_bit, third_bit
        RUN_TIME = 0.7

        pca = skld.PCA(3)
        activations_transformed = pca.fit_transform(self.activations)
        shape = Polygon(*activations_transformed[:200, :], color=BLUE, width=0.1)
        dot = Dot(activations_transformed[0, :], color=RED, size=1.2)
        self.play(ShowCreation(shape),
                  ShowCreation(dot),
                  run_time=RUN_TIME)
        first_bit, second_bit, third_bit = generate_output_mobjects(outputs[0, :])
        self.play(ShowCreation(first_bit),
                  ShowCreation(second_bit),
                  ShowCreation(third_bit))


        for i in range(20):
            new_first_bit, new_second_bit, new_third_bit = generate_output_mobjects(outputs[i, :])
            new_dot = Dot(activations_transformed[i, :], color=RED, size=1.2)
            self.play(Transform(dot, new_dot),
                      Transform(first_bit, new_first_bit),
                      Transform(second_bit, new_second_bit),
                      Transform(third_bit, new_third_bit),
                      run_time=0.2)
            self.wait(0.3)
# TODO: add input and predictions to animation as numbers


class AnimateFpfOptimization(ThreeDScene):

    def setup(self):

        rnn_type = 'vanilla'
        n_hidden = 24

        # initialize Flipflopper class
        flopper = Flipflopper(rnn_type=rnn_type, n_hidden=n_hidden)
        # generate trials
        stim = flopper.generate_flipflop_trials()
        # train the model
        # flopper.train(stim, 2000, save_model=True)

        # visualize a single batch after training
        # prediction = flopper.model.predict(tf.convert_to_tensor(stim['inputs'], dtype=tf.float32))
        # visualize_flipflop(stim)

        # if a trained model has been saved, it may also be loaded
        flopper.load_model()
        ############################################################
        # Initialize fpf and find fixed points
        ############################################################
        # get weights and activations of trained model
        weights = flopper.model.get_layer(flopper.hps['rnn_type']).get_weights()
        activations = flopper.get_activations(stim)
        # initialize adam fpf
        fpf = RecordingFixedpointfinder(weights, rnn_type,
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
        self.fps, self.recorded_points = fpf.find_fixed_points(states, inputs)
        self.output_weights = flopper.model.get_layer('dense').get_weights()
        self.outputs = np.vstack(stim['output'])
        # self.activations = self.outputs @ self.output_weights[0].T
        self.activations = np.vstack(activations)

    def construct(self):

        pca = skld.PCA(3)

        transformed_activations = pca.fit_transform(self.activations)

        sphere = Polygon(*transformed_activations[:500, :], width=0.05, color=BLUE)
        self.set_camera_orientation(phi=70 * DEGREES, theta=-120 * DEGREES)

        self.play(ShowCreation(sphere))
        initial_points = self.recorded_points[0]
        transformed_points = pca.transform(initial_points)
        dots = VGroup(*[Dot(transformed_point, color=RED_B, size=0.15) for transformed_point in transformed_points])

        k = 0
        #info_text = ["Iteration:", str(k * 200)]
        #info_text_mob = TextMobject(*info_text)
        #info_text_mob.to_edge()

        self.play(ShowCreation(dots))
                  # ShowCreation(info_text_mob))

        for initial_points in self.recorded_points:
            # new_info_text = ["Iteration:", str(k * 200)]
            # new_info_text_mob = TextMobject(*new_info_text)
            # new_info_text_mob.to_edge()

            transformed_points = pca.transform(initial_points)

            new_dots = VGroup(*[Dot(transformed_point, color=RED_B, size=0.15) for transformed_point in transformed_points])
            self.play(Transform(dots, new_dots))
                      # Transform(info_text_mob, new_info_text_mob))
            self.wait(0.3)
            k += 1


class AnimateActivitySingleRunGru(ThreeDScene):

    def setup(self):
        # agent_id = 1583404415  # 1583180664 lunarlandercont
        agent_id = 1583256614 # reach task
        os.chdir("../")
        chiefinvesti = Chiefinvestigator(agent_id)
        os.chdir("./rnn_dynamical_systems")
        layer_names = chiefinvesti.get_layer_names()
        print(layer_names)

        self.activations_over_episode, self.inputs_over_episode, \
        self.actions_over_episode = chiefinvesti.get_data_over_single_run("policy_recurrent_layer",
                                                                          layer_names[1])

        self.weights = chiefinvesti.weights

        self.sGru = SerializedGru(self.weights, chiefinvesti.n_hidden, 1/14) # helper class to serialize gru

        self.complex_activations_h = self.activations_over_episode @ (self.sGru.trans_factor * self.sGru.h_c_inv) # transform activations to complex system
        #self.complex_activations_h = np.vstack((np.zeros(chiefinvesti.n_hidden), self.complex_activations_h[:-1, :]))

        self.complex_activations_z = self.activations_over_episode @ (
                    self.sGru.trans_factor * self.sGru.z_c_inv)  # transform activations to complex system
        #self.complex_activations_z = np.vstack((np.zeros(chiefinvesti.n_hidden), self.complex_activations_z[:-1, :]))

        self.complex_activations_r = self.activations_over_episode @ (self.sGru.trans_factor * self.sGru.r_c_inv) # transform activations to complex system
        #self.complex_activations_r = np.vstack((np.zeros(chiefinvesti.n_hidden), self.complex_activations_r[:-1, :]))

        self.sGru.handle_inputs(self.inputs_over_episode)


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

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        pca = skld.PCA(3)

        transformed_activations = pca.fit_transform(self.complex_activations_h)

        shape = Polygon(*transformed_activations[:100, :], color=BLUE, width=0.1)
        vector = Vector(transformed_activations[0, :], color=RED_B)

        z_vector = (self.complex_activations_z[0, :] @ self.sGru.serialized["z_update"][0][0])
        z_transformed = pca.transform(z_vector.reshape(1, -1))[0]
        z_arrow = Vector(z_transformed, color=GREEN, width=0.1)

        r_vector = self.complex_activations_r[0, :] @ self.sGru.serialized["r_reset"][0][0]
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
        print(self.sGru.n_evals)
        for timestep in range(5):

            imaginary_activations_h = self.complex_activations_h[timestep, :]
            imaginary_activations_z = self.complex_activations_z[timestep, :]
            imaginary_activations_r = self.complex_activations_r[timestep, :]
            new_timestepscounter = TextMobject("Timestep:" , str(timestep)).to_edge(LEFT)
            self.play(Transform(timestepscounter, new_timestepscounter), run_time=0.7)

            z_transformed = pca.transform(imaginary_activations_z.reshape(1, -1))[0]
            r_transformed = pca.transform(imaginary_activations_r.reshape(1, -1))[0]

            new_z_arrow = Vector(z_transformed, color=GREEN)
            new_r_arrow = Vector(r_transformed, color=YELLOW)

            new_point = pca.transform(imaginary_activations_h.reshape(1, -1))[0]
            new_vector = Vector(new_point, color=RED_B)

            self.play(Transform(z_arrow, new_z_arrow),
                      Transform(r_arrow, new_r_arrow),
                      Transform(vector, new_vector),
                      run_time=0.3)

            r_iterator = 0
            z_iterator = 0
            for sub_timestep in range(self.sGru.n_evals[2]):
                if r_iterator <= self.sGru.n_evals[1]:
                    r_vector = sigmoid(imaginary_activations_r @ self.sGru.serialized["r_reset"][0][r_iterator] \
                                       + self.sGru.r_input_serial[r_iterator][timestep, :] + self.sGru.b_r)
                else:
                    r_vector = 0

                if r_iterator < (self.sGru.n_evals[1]-1):
                    r_iterator += 1

                h_vector = np.tanh((r_vector * imaginary_activations_h) @ self.sGru.serialized["h_activation"][0][sub_timestep] \
                                  + self.sGru.h_input_serial[sub_timestep][timestep, :] + self.sGru.b_h)

                if z_iterator <= self.sGru.n_evals[0]:
                    z_vector = sigmoid(imaginary_activations_z @ self.sGru.serialized["z_update"][0][z_iterator] \
                                       + self.sGru.z_input_serial[z_iterator][timestep, :] + self.sGru.b_z)
                else:
                    z_vector = 0

                if z_iterator < (self.sGru.n_evals[0]-1):
                    z_iterator += 1

                z_transformed = pca.transform(z_vector.reshape(1, -1))[0]
                r_transformed = pca.transform(r_vector.reshape(1, -1))[0]

                new_z_arrow = Vector(z_transformed, color=GREEN)
                new_r_arrow = Vector(r_transformed, color=YELLOW)

                imaginary_activations_h = self.complex_activations_h[timestep, :] + \
                                        (1 - z_vector) * (h_vector - imaginary_activations_h)
                imaginary_activations_r = imaginary_activations_r * imaginary_activations_h
                imaginary_activations_z = imaginary_activations_z * imaginary_activations_h
                #imaginary_activations = self.complex_activations[timestep, :]

                new_point = pca.transform(imaginary_activations_h.reshape(1, -1))[0]
                new_vector = Vector(new_point, color=RED_B)

                self.play(Transform(z_arrow, new_z_arrow),
                          Transform(r_arrow, new_r_arrow),
                          Transform(vector, new_vector),
                          run_time=0.3)

class AnimateAcivitySingleRunVanilla(ThreeDScene):

    def setup(self):
        agent_id = 1585500821  # cartpole-v1
        os.chdir("../")
        #agent_id = 1583404415  # 1583180664 lunarlandercont
        # agent_id = 1583256614 # reach task
        chiefinvesti = Chiefinvestigator(agent_id)
        os.chdir("./rnn_dynamical_systems")
        layer_names = chiefinvesti.get_layer_names()
        print(layer_names)

        self.activations_over_episode, self.inputs_over_episode, \
        self.actions_over_episode = chiefinvesti.get_data_over_single_run("policy_recurrent_layer",
                                                                          layer_names[1])
        self.activations_over_episode = np.vstack((np.zeros(chiefinvesti.n_hidden), self.activations_over_episode[:-1, :]))
        self.weights = chiefinvesti.weights
        self.sVanilla = SerializedVanilla(self.weights[1], chiefinvesti.n_hidden)

        self.complex_input = self.inputs_over_episode @ self.weights[0] @ (1 / 6 * self.sVanilla.evecs_c_inv)

    def construct(self):

        pca_activations = skld.PCA(3)

        complex_activations = self.sVanilla.translate_to_complex(self.activations_over_episode, 1 / 6 * self.sVanilla.evecs_c_inv)
        complex_activations = complex_activations @ self.sVanilla.all_diagonal  # move forward one timestep for contraction
        activations_transformed = pca_activations.fit_transform(complex_activations)

        vector = Vector(activations_transformed[0, :], color=RED_B)
        activation_shape = Polygon(*activations_transformed[:500, :], color=BLUE, width=0.1)

        timestep = 0
        timestepscounter = TextMobject("Timestep:", str(timestep))

        self.play(ShowCreation(TextMobject("Vanilla vector computation").to_edge(UP)),
                  ShowCreation(timestepscounter.to_edge(LEFT)),
                  ShowCreation(activation_shape),
                  ShowCreation(vector))

        for timestep in range(10):

            imaginary_activations = complex_activations[timestep, :]
            imaginary_input = self.complex_input[timestep, :]
            new_timestepscounter = TextMobject("Timestep:", str(timestep)).to_edge(LEFT)
            self.play(Transform(timestepscounter, new_timestepscounter), run_time=0.4)

            for sub_timestep in range(len(self.sVanilla.reconstructed_diagonals)):
                single_transformation = imaginary_activations @ self.sVanilla.reconstructed_diagonals[sub_timestep]
                input_update = imaginary_input @ self.sVanilla.reconstructed_diagonals[sub_timestep]
                imaginary_activations = complex_activations[timestep, :] + single_transformation + input_update

                end_point = pca_activations.transform(imaginary_activations.reshape(1, -1))[0]

                new_vector = Vector(end_point, color=RED_B)

                self.play(Transform(vector, new_vector), run_time=0.1)