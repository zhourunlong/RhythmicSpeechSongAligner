import os, time, matplotlib

import matplotlib.pyplot as plt
import numpy as np # Most of the math in visbeat is done with numpy
import sys
sys.path.append("..")
# We will call visbeat functions through 'vb'
import visbeat as vb

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
path_to_test_video = os.path.join(test_files_folder, 'synth_ball.mp4');
synth_ball=SourceMedia(path = path_to_test_video);

# Pull the video into VisBeatAssets
vb.PullVideo(name=synth_ball.name, source_location=synth_ball.path);

vid = vb.LoadVideo(name=synth_ball.name)

path_to_test_video2 = os.path.join(test_files_folder, 'synth_ball.mp4');
synth_ball=SourceMedia(path = path_to_test_video2);

# Pull the video into VisBeatAssets
vb.PullVideo(name=synth_ball.name, source_location=synth_ball.path);

vid2 = vb.LoadVideo(name=synth_ball.name)

synch_video_beat = 0;
synch_audio_beat = 0;
nbeats = 0;

output_path = './SexyTurtleScientist.mp4';

warped = vb.Dancify(source_video=vid, target=vid2, synch_video_beat=synch_video_beat,
                    synch_audio_beat=synch_audio_beat, force_recompute=True, warp_type = 'quad',
                    nbeats=nbeats, output_path = output_path)