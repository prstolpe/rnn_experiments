from manimlib.imports import *
import os
import sklearn.decomposition as skld
import sys
sys.path.append("/Users/Raphael//rnn_dynamical_systems")
sys.path.append("/Users/Raphael/dexterous-robot-hand")
from analysis.chiefinvestigation import Chiefinvestigator
from rnn_dynamical_systems.fixedpointfinder.FixedPointFinder import Adamfixedpointfinder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class AnimateReachPhaseSpace(ThreeDScene):

    def setup(self):
        os.chdir("../")
        agent_id = 1583256614  # reach task
        self.chiefinvesti = Chiefinvestigator(agent_id)

        layer_names = self.chiefinvesti.get_layer_names()
        print(layer_names)
        # collect data from episodes
        n_episodes = 1
        self.activations_over_all_episodes, self.inputs_over_all_episodes, \
        actions_over_all_episodes = self.chiefinvesti.get_data_over_episodes(n_episodes,
                                                                        "policy_recurrent_layer",
                                                                        layer_names[1])

        # employ fixedpointfinder
        adamfpf = Adamfixedpointfinder(self.chiefinvesti.weights, self.chiefinvesti.rnn_type,
                                       q_threshold=1e-12,
                                       tol_unique=1e-03,
                                       epsilon=0.1,
                                       alr_decayr=5e-03,
                                       agnc_normclip=2,
                                       agnc_decayr=1e-03,
                                       max_iters=5000)
        n_samples, noise_level = 100, 0
        states, sampled_inputs = adamfpf.sample_inputs_and_states(self.activations_over_all_episodes,
                                                                  self.inputs_over_all_episodes,
                                                                  n_samples,
                                                                  noise_level)

        self.fps = adamfpf.find_fixed_points(states, sampled_inputs)



    def construct(self):

        timespan, stepsize = (0, 100), 20
        x, y, z, u, v, w, activations = self.chiefinvesti.compute_quiver_data(self.inputs_over_all_episodes,
                                                                              self.activations_over_all_episodes,
                                                                              timespan,
                                                                              stepsize, 0)
        start_points = np.vstack((x, y, z)).transpose()
        end_points = np.vstack((u, v, w)).transpose()

        fp_locations = np.vstack([fp['x'] for fp in self.fps])
        pca = skld.PCA(3)
        transformed_activations = pca.fit_transform(self.activations_over_all_episodes[:100, :])
        fp_transformed = pca.transform(fp_locations)

        fixedpoint_dots = VGroup(*[Dot(f, color=RED_B) for f in fp_transformed])
        activation_shape = Polygon(*transformed_activations[:100, :], color=BLUE)

        phase_space = VGroup(*[Arrow(start_points[i, :], 0.02 * end_points[i, :], width=0.1, color=GREEN_D) for i in range(len(start_points))])

        self.play(ShowCreation(activation_shape),
                  ShowCreation(fixedpoint_dots),
                  ShowCreation(phase_space))
        self.set_camera_orientation(phi=-65, theta=130)

        for k in range(30):
            timespan, stepsize = (0, 100), 10
            x, y, z, u, v, w, activations = self.chiefinvesti.compute_quiver_data(self.inputs_over_all_episodes,
                                                                                  self.activations_over_all_episodes,
                                                                                  timespan,
                                                                                  stepsize, k)

            new_end_points = np.vstack((u, v, w)).transpose()

            new_phase_space = VGroup(
                *[Arrow(start_points[i, :], 0.02 * new_end_points[i, :], color=GREEN) for i in range(len(start_points))])

            self.play(Transform(phase_space, new_phase_space))




