# actual_action = actuation_center + aoi * actuation_range
geom_quat = chiefinvesti.env.sim.model.geom_quat

ctrlrange = chiefinvesti.env.sim.model.actuator_ctrlrange
actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.
actuation_center = np.zeros(20)
for i in range(chiefinvesti.env.sim.data.ctrl.shape[0]):
    actuation_center[i] = chiefinvesti.env.sim.data.get_joint_qpos(
        chiefinvesti.env.sim.model.actuator_names[i].replace(':A_', ':'))
for joint_name in ['FF', 'MF', 'RF', 'LF']:
    act_idx = chiefinvesti.env.sim.model.actuator_name2id(
        'robot0:A_{}J1'.format(joint_name))
    actuation_center[act_idx] += chiefinvesti.env.sim.data.get_joint_qpos(
        'robot0:{}J0'.format(joint_name))
n = 2
aoi = actions_over_all_episodes[:n, :]
actual_action = np.repeat(actuation_center.reshape(1, -1), axis=0, repeats=n) + aoi * actuation_range


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def compute_a_theta(some_quat):
    a_some_quat = some_quat[1:] / np.sqrt(np.sum(np.square(some_quat[1:])))
    print(a_some_quat)
    theta_some_quat = 2 * np.arctan2(np.sqrt(np.sum(np.square(some_quat[1:]))), some_quat[0])
    print(np.degrees(theta_some_quat))

    return a_some_quat, theta_some_quat


# update quaternion
# new_quat = np.ndarray((np.cos(new_angle/2), 0, np.sin(new_angle/2)*some_quat[2], 0))

def quaternion_mult(q, r):
    return [r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3],
            r[0] * q[1] + r[1] * q[0] - r[2] * q[3] + r[3] * q[2],
            r[0] * q[2] + r[1] * q[3] + r[2] * q[0] - r[3] * q[1],
            r[0] * q[3] - r[1] * q[2] + r[2] * q[1] + r[3] * q[0]]


def point_rotation_by_quaternion(point, q):
    r = point
    q_conj = [q[0], -1 * q[1], -1 * q[2], -1 * q[3]]
    return quaternion_mult(quaternion_mult(q, r), q_conj)


# finding orientation and angle of geoms
some_quat = geom_quat[4, :]
a_some_quat, theta_some_quat = compute_a_theta(some_quat)

initial_action = -0.16514339750464327
rotation_quat = [np.cos(initial_action / 2), 0, np.sin(initial_action / 2), 0]
a = point_rotation_by_quaternion(rotation_quat, some_quat)

print('Angle between old point and new point', np.degrees(angle_between(a[1:], a_some_quat)))
print('Degrees by action', np.degrees(initial_action))
a_vector, theta_a = compute_a_theta(a)

rotation_quat = [np.cos(actual_action[0, 0] / 2), 0, np.sin(actual_action[0, 0] / 2), 0]
b = point_rotation_by_quaternion(rotation_quat, some_quat)
print('Angle between old point and new point', np.degrees(angle_between(b[1:], a_some_quat)))
print('Degrees by action', np.degrees(actual_action[0, 0]))
b_vector, theta_b = compute_a_theta(b)

rotation_quat = [np.cos(actual_action[1, 0] / 2), 0, np.sin(actual_action[1, 0] / 2), 0]
c = point_rotation_by_quaternion(rotation_quat, some_quat)
print('Angle between old point and new point', np.degrees(angle_between(c[1:], a_some_quat)))
print('Degrees by action', np.degrees(actual_action[1, 0]))
c_vector, theta_c = compute_a_theta(c)