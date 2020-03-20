from analysis.chiefinvestigation import Chiefinvestigator
from manimlib.imports import *
import os
import sklearn.decomposition as skld

os.chdir("./rnn_domains/")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class AnimateActivitySingleRun(ThreeDScene):

    def setup(self):
        agent_id = 1583404415  # 1583180664 lunarlandercont
        # agent_id = 1583256614 # reach task
        chiefinvesti = Chiefinvestigator(agent_id)

        layer_names = chiefinvesti.get_layer_names()
        print(layer_names)

        self.activations_over_episode, self.inputs_over_episode, \
        self.actions_over_episode = chiefinvesti.get_data_over_single_run("policy_recurrent_layer",
                                                                          layer_names[1])

    def construct(self):
        RUN_TIME = 0.02

        pca = skld.PCA(3)
        activations_transformed = pca.fit_transform(self.activations_over_episode)
        dot = Dot(activations_transformed[0, :])
        self.play(ShowCreation(dot), run_time=RUN_TIME)
        for i in range(len(self.activations_over_episode)):
            line = Line(activations_transformed[i, :], activations_transformed[i + 1, :])
            self.play(ShowCreation(line), run_time=RUN_TIME)