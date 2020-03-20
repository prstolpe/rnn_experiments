#from analysis.chiefinvestigation import Chiefinvestigator
from manimlib.imports import *
import os
import sklearn.decomposition as skld
import sys
sys.path.append("/Users/Raphael/rnn_dynamical_systems")
from fixedpointfinder.three_bit_flip_flop import Flipflopper

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