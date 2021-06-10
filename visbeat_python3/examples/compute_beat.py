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




#read all vedio file
video_path = os.path.join(test_files_folder, 'video')
audio_path = os.path.join(test_files_folder, 'audio')


path_list = os.listdir(video_path)
print(path_list)
path_to_test_video = None
for temp_path in path_list:
    path_to_test_video = os.path.join(test_files_folder, temp_path);
    video_file=SourceMedia(path = path_to_test_video);
    vb.PullVideo(name=video_file.name, source_location=video_file.path);
    vid = vb.LoadVideo(name=video_file.name)
    vbeats = vid.getVisualBeats();
    vbeats_txt = []
    for beat in vbeats:
        vbeats_txt.append(float(beat.start))
    output_path = "./beat_result/video/" + temp_path + ".txt"
    np.savetxt(output_path,vbeats_txt,fmt='%4f')

path_list = os.listdir(audio_path)
print(path_list)
path_to_test_audio = None
for temp_path in path_list:
    path_to_test_audio = os.path.join(test_files_folder, temp_path);
    audio_file=SourceMedia(path = path_to_test_audio);
    vb.PullVideo(name=audio_file.name, source_location=audio_file.path);
    vid = vb.LoadVideo(name=audio_file.name)
    vid_audio = vid.getAudio()
    abeats = vid_audio.getBeatEvents();
    abeats_txt = []
    for beat in abeats:
        abeats_txt.append(float(beat.start))
    output_path = "./beat_result/audio/" + temp_path + ".txt"
    np.savetxt(output_path,abeats_txt,fmt='%4f')




# Pull the video into VisBeatAssets


