import os
import numpy as np


def evaluate_pose(pose_seq, exercise):
    """
    Evaluate a Pose Sequence for a Particular Exercise.

    Arguments:
        pose_seq: PoseSequence object
        exercise: String name of the Exercise to be Evaluated

    Returns:
        correct: Bool whether Exercise was Performed Correctly
        feedback: Feedback String
    """
    if exercise == 'bicep_curl':
        return _bicep_curl(pose_seq)

    # More Exercises in Future

    else:
        return (False, "Exercise String Not Recognized!!!")


def _bicep_curl(pose_seq):

    # Find the Arm that is seen most Consistently
    poses = pose_seq.poses
    right_present = [1 for pose in poses 
            if pose.rshoulder.exists and pose.relbow.exists and pose.rwrist.exists]
    left_present = [1 for pose in poses
            if pose.lshoulder.exists and pose.lelbow.exists and pose.lwrist.exists]
    right_count = sum(right_present)
    left_count = sum(left_present)
    side = 'right' if right_count > left_count else 'left'

    print('Exercise Arm Detected as: {}.'.format(side))

    if side == 'right':
        joints = [(pose.rshoulder, pose.relbow, pose.rwrist, pose.rhip, pose.neck) for pose in poses]
    else:
        joints = [(pose.lshoulder, pose.lelbow, pose.lwrist, pose.lhip, pose.neck) for pose in poses]

    # Filter out Data Points where a Part does not Exist
    joints = [joint for joint in joints if all(part.exists for part in joint)]

    upper_arm_vecs = np.array([(joint[0].x - joint[1].x, joint[0].y - joint[1].y) for joint in joints])
    torso_vecs = np.array([(joint[4].x - joint[3].x, joint[4].y - joint[3].y) for joint in joints])
    forearm_vecs = np.array([(joint[2].x - joint[1].x, joint[2].y - joint[1].y) for joint in joints])

    # Normalize Vectors
    upper_arm_vecs = upper_arm_vecs / np.expand_dims(np.linalg.norm(upper_arm_vecs, axis=1), axis=1)
    torso_vecs = torso_vecs / np.expand_dims(np.linalg.norm(torso_vecs, axis=1), axis=1)
    forearm_vecs = forearm_vecs / np.expand_dims(np.linalg.norm(forearm_vecs, axis=1), axis=1)

    upper_arm_torso_angles = np.degrees(np.arccos(np.clip(np.sum(np.multiply(upper_arm_vecs, torso_vecs), axis=1), -1.0, 1.0)))
    upper_arm_forearm_angles = np.degrees(np.arccos(np.clip(np.sum(np.multiply(upper_arm_vecs, forearm_vecs), axis=1), -1.0, 1.0)))

    # Use Thresholds learned from Analysis
    upper_arm_torso_range = np.max(upper_arm_torso_angles) - np.min(upper_arm_torso_angles)
    upper_arm_forearm_min = np.min(upper_arm_forearm_angles)

    print('Upper Arm and Torso Angle Range : {}'.format(upper_arm_torso_range))
    print('Upper Arm and Forearm Minimum Angle : {}'.format(upper_arm_forearm_min))

    correct = True
    feedback = ''

    if upper_arm_torso_range > 35.0:
        correct = False
        feedback += 'Your upper arm shows Significant Rotation around the shoulder when curling. Try holding your upper arm still, parallel to your chest, ' + \
                    'and concentrate on rotating around your elbow only.\n'
    
    if upper_arm_forearm_min > 70.0:
        correct = False
        feedback += 'You are Not Curling the weight all the way to the Top, up to your shoulders. Try to curl your arm completely so that your forearm is parallel with your torso. It may help to use lighter weight.\n'

    if correct:
        return (correct, 'Exercise performed Correctly!!! Weight was lifted fully up, and upper arm did not move significantly!!!')
    else:
        return (correct, feedback)
