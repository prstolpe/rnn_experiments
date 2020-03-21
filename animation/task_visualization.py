#from analysis.chiefinvestigation import Chiefinvestigator
from manimlib.imports import *
import os
import sklearn.decomposition as skld
import sys
sys.path.append("/Users/Raphael/rnn_dynamical_systems")
from fixedpointfinder.three_bit_flip_flop import Flipflopper, RetrainableFlipflopper

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
        # flopper.train(stim, 4000, save_model=True)

        # if a trained model has been saved, it may also be loaded
        flopper.load_model()

        self.weights = flopper.model.get_layer(flopper.hps['rnn_type']).get_weights()
        self.output_weights = flopper.model.get_layer('dense').get_weights()
        # self.activations = np.vstack(flopper.get_activations(stim))
        self.outputs = np.vstack(stim['output'])
        self.activations = self.outputs @ self.output_weights[0].T


    def construct(self):
        RUN_TIME = 0.02

        pca = skld.PCA(3)
        activations_transformed = pca.fit_transform(self.activations)
        activations_transformed = activations_transformed * 0.5
        dot = Dot(activations_transformed[0, :], size=0.2)
        self.play(ShowCreation(dot), run_time=RUN_TIME)
        for i in range(256):
            line = Line(activations_transformed[i, :], activations_transformed[i + 1, :])
            self.play(
                      ShowCreation(line), run_time=RUN_TIME)


class StimAnim(GraphScene, MappingCamera):
    CONFIG = {
        "x_min": -1,
        "x_max": 20,
        "y_min": -2,
        "y_max": 2,
        "graph_origin": LEFT_SIDE,
        "function_color": WHITE,
        "axes_color": BLUE
    }

    def construct(self):
        rnn_type = 'vanilla'
        n_hidden = 24

        flopper = Flipflopper(rnn_type=rnn_type, n_hidden=n_hidden)
        stim = flopper.generate_flipflop_trials()
        z = np.zeros(20)
        x = np.arange(0, 10, 0.5) - 7.111
        y = stim['inputs'][0, :20, 1]
        l= np.vstack((x,y, z))
        self.setup_axes(animate=True)
        for i in range(19):
            line = Line(l[:, i], l[:, i+1])
            self.play(ShowCreation(line), run_time=0.1)

class AnimateFlipFlopLearning(ThreeDScene):

    def setup(self):

        rnn_type = 'vanilla'
        n_hidden = 24

        flopper = RetrainableFlipflopper(rnn_type=rnn_type, n_hidden=n_hidden)
        stim = flopper.generate_flipflop_trials()

        _ = flopper.initial_train(stim, 10, True)
        self.iterations = 50
        self.collected_activations = []
        for i in range(self.iterations):
            self.collected_activations.append(np.vstack(flopper.get_activations(stim)))
            _ = flopper.continued_train(stim, 50, True)

    def construct(self):

        pca = skld.PCA(3)

        pca.fit(self.collected_activations[-1])
        self.set_camera_orientation(phi=60 * DEGREES, theta=-120 * DEGREES)
        transformed_activations = pca.transform(self.collected_activations[0])

        line = Polygon(*transformed_activations[:500, :],
                       color=BLUE, width=0.1)
        self.play(ShowCreation(line, run_time=0.1))

        for it in range(self.iterations):
            transformed_activations = pca.transform(self.collected_activations[it])

            new_line = Polygon(*transformed_activations[:256, :],
                        color=BLUE, width=0.1)
            self.play(Transform(line, new_line, run_time=0.1))
            self.remove(line)
            self.wait(0.5)
            line = new_line