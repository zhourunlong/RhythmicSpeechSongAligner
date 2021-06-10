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
parser.add_argument("-name", "-n", help="this is name of the file", type=str, default=None)
parser.add_argument("-type", "-t", help="the type of the file",  type=str, default="video")
args = parser.parse_args()

# Set the AssetsDir
vb.SetAssetsDir('./VisBeatAssets/');

# For this tutorial we'll also use some files from the test_files dir
test_files_folder = os.path.join('.', 'test_files');
test_files = os.path.join(test_files_folder, args.type)
test_files = os.path.join(test_files, args.name)

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

video_file=SourceMedia(path = test_files);
vb.PullVideo(name=video_file.name, source_location=video_file.path);
vid = vb.LoadVideo(name=video_file.name)
if(args.type == "video"):
    vbeats = vid.getVisualBeats();
else:
    vbeats = vid.getAudio().getBeatEvents();
vbeats_txt = []
for beat in vbeats:
    vbeats_txt.append(float(beat.start))
output_path = "./beat_result/" + args.type + "/" + args.name + ".txt"
np.savetxt(output_path,vbeats_txt,fmt='%4f')