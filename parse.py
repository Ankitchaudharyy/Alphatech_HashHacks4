import argparse
import glob
import json
import numpy as np
import os

from pose import Pose, Part, PoseSequence
from pprint import pprint


def main():

    parser = argparse.ArgumentParser(description='AI-GYM Parser')
    parser.add_argument('--input_folder', type=str, default='poses', help='Input Folder for JSON Files')
    parser.add_argument('--output_folder', type=str, default='poses_compressed', help='Output Folder for npy Files')
    
    args = parser.parse_args()

    video_paths = glob.glob(os.path.join(args.input_folder, '*'))
    video_paths = sorted(video_paths)

    # Get all the JSON Sequences for each Video
    all_ps = []
    for video_path in video_paths:
        all_ps.append(parse_sequence(video_path, args.output_folder))
    return video_paths, all_ps


def parse_sequence(json_folder, output_folder):
    
    """
    Parse a Sequence of OpenPose JSON frames and saves a corresponding numpy file.

    Args:
        json_folder: Path to the Folder Containing OpenPose JSON for one Video.
        output_folder: Path to Save the Numpy Array Files of Keypoints.
    """
    
    json_files = glob.glob(os.path.join(json_folder, '*.json'))
    json_files = sorted(json_files)

    num_frames = len(json_files)
    all_keypoints = np.zeros((num_frames, 18, 3))
    for i in range(num_frames):
        with open(json_files[i]) as f:
            json_obj = json.load(f)
            keypoints = np.array(json_obj['people'][0]['pose_keypoints'])
            all_keypoints[i] = keypoints.reshape((18, 3))
    
    output_dir = os.path.join(output_folder, os.path.basename(json_folder))
    np.save(output_dir, all_keypoints)


def load_ps(filename):
    """
    Load a PoseSequence Object from a Given Numpy File

    Args:
        filename: file name of the numpy file containing keypoints.
    
    Returns:
        PoseSequence object with normalized joint keypoints.
    """
    all_keypoints = np.load(filename)
    return PoseSequence(all_keypoints)


if __name__ == '__main__':
    main()

