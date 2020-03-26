# from analysis.chiefinvestigation import Chiefinvestigator
from manimlib.imports import *
import os
import sklearn.decomposition as skld
import sys
sys.path.append("/Users/Raphael/rnn_dynamical_systems")
from fixedpointfinder.three_bit_flip_flop import Flipflopper, RetrainableFlipflopper
from fixedpointfinder.FixedPointFinder import RecordingFixedpointfinder

# os.chdir("./animation/")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class AnimateActivitySingleRun(ThreeDScene):

    def setup(self):
        # ZoomedScene.setup(self)
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
            #dot = new_dot
            # first_bit, second_bit, third_bit = new_first_bit, new_second_bit, new_third_bit
# TODO: add input and predictions to animation


class AnimateFlipFlopLearning(ThreeDScene):

    def setup(self):

        rnn_type = 'vanilla'
        n_hidden = 24

        flopper = RetrainableFlipflopper(rnn_type=rnn_type, n_hidden=n_hidden)
        stim = flopper.generate_flipflop_trials()

        _ = flopper.initial_train(stim, 10, True)
        self.iterations = 5
        self.collected_activations = []
        for i in range(self.iterations):
            self.collected_activations.append(np.vstack(flopper.get_activations(stim)))
            _ = flopper.continued_train(stim, 50, True)

    def construct(self):

        pca = skld.PCA(3)

        pca.fit(self.collected_activations[-1])
        self.set_camera_orientation(phi=60 * DEGREES, theta=-120 * DEGREES)

        for it in range(self.iterations):
            transformed_activations = pca.transform(self.collected_activations[it])

            line = Polygon(*transformed_activations[:256, :],
                           color=BLUE, width=0.1)
            self.add(line)
            self.wait(0.5)
            self.remove(line)


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
        # vel = fpf.compute_velocities(np.hstack(activations[1:]), np.zeros((32768, 3)))
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

