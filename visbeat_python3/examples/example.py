import os, time, matplotlib

import matplotlib.pyplot as plt
import numpy as np # Most of the math in visbeat is done with numpy
import sys
sys.path.append("..")
# We will call visbeat functions through 'vb'
import visbeat as vb

import argparse
parser = argparse.ArgumentParser()
parser.description='please enter two parameters video and audio name'
parser.add_argument("-video_name", "-v", help="this is video_name", type=str, default=None)
parser.add_argument("-audio_name", "-a", help="this is audio_name",  type=str, default=None)
parser.add_argument("-nbeat", "-n", help="the output bit",  type=int, default=0)
args = parser.parse_args()

# Set the AssetsDir
vb.SetAssetsDir('./VisBeatAssets/');

# For this tutorial we'll also use some files from the test_files dir
test_files_folder = os.path.join('.', 'test_files');

# Small convenience class to make code more readable and use basic file name if no name is provided;
class SourceMedia:
    def __init__(self, path, name=None, **kwargs):
        self.path = path;
        self._name = name;
        self.__dict__.update(**kwargs)
    @property
    def name(self):         
        if(self._name is not None):
            return self._name;
        else:
            return os.path.splitext(os.path.basename(self.path))[0];

# In this case the name of our SourceMedia will default to 'synth_ball' 
if (args.audio_name == None):
    path_to_test_video = os.path.join(test_files_folder, 'synth_ball.mp4');
else:
    path_to_test_video = os.path.join(test_files_folder, args.audio_name);
audio_file=SourceMedia(path = path_to_test_video);

# Pull the video into VisBeatAssets
vb.PullVideo(name=audio_file.name, source_location=audio_file.path);

vid = vb.LoadVideo(name=audio_file.name)

if (args.video_name == None):
    path_to_test_video2 = os.path.join(test_files_folder, 'synth_ball.mp4');
else:
    path_to_test_video2 = os.path.join(test_files_folder, args.video_name);
video_file=SourceMedia(path = path_to_test_video2);

# Pull the video into VisBeatAssets
vb.PullVideo(name=video_file.name, source_location=video_file.path);

vid2 = vb.LoadVideo(name=video_file.name)

synch_video_beat = 0;
synch_audio_beat = 0;
nbeats = args.nbeat;



output_path = './warps_result/Dance_Speech.mp4';

warped = vb.Dancify(source_video=vid2, target=vid, synch_video_beat=synch_video_beat,
                    synch_audio_beat=synch_audio_beat, force_recompute=True, warp_type = 'quad',
                    nbeats=nbeats, output_path = output_path)