import srt
import argparse
import torch
import os, sys
from semantic_matching.model import SemanticMatchingModel
import visbeat as vb
import numpy as np
import pickle

from moviepy.editor import *

class SourceMedia:
    def __init__(self, path, name=None, **kwargs):
        self.path = path
        self._name = name
        self.__dict__.update(**kwargs)
    @property
    def name(self):         
        if(self._name is not None):
            return self._name
        else:
            return os.path.splitext(os.path.basename(self.path))[0]

def audio_get_beats(path_to_file, path_to_pickle):
    if os.path.exists(path_to_pickle):
        with open(path_to_pickle, 'rb') as f:
            return pickle.load(f)
    
    audio_file = SourceMedia(path_to_file)
    vb.PullVideo(name=audio_file.name, source_location=audio_file.path)
    vid = vb.LoadVideo(name=audio_file.name)
    vid_audio = vid.getAudio()
    abeats = vid_audio.getBeatEvents()
    beats = [float(beat.start) for beat in abeats]
    f = open(path_to_pickle, 'wb')
    pickle.dump(beats, f)
    return beats

def video_get_beats(path_to_file, path_to_pickle):
    if os.path.exists(path_to_pickle):
        with open(path_to_pickle, 'rb') as f:
            return pickle.load(f)

    video_file = SourceMedia(path_to_file)
    vb.PullVideo(name=video_file.name, source_location=video_file.path)
    vid = vb.LoadVideo(name=video_file.name)
    vbeats = vid.getVisualBeats()
    beats = [float(beat.start) for beat in vbeats]
    f = open(path_to_pickle, 'wb')
    pickle.dump(beats, f)
    return beats

warp = [[0, 36, 13, 1869, 1882], [36, 44, 0, 1520, 1528], [44, 52, 8, 4749, 4756], [52, 59, 10, 2726, 2732], [59, 67, 10, 2555, 2563], [67, 76, 10, 3693, 3702], [76, 83, 6, 668, 673], [83, 90, 6, 747, 753], [90, 98, 13, 3329, 3336], [98, 106, 6, 1048, 1056], [106, 116, 9, 1474, 1483], [116, 123, 2, 308, 314], [123, 131, 8, 3122, 3131], [131, 139, 8, 916, 926], [139, 147, 10, 3664, 3673], [147, 158, 13, 4204, 4214], [158, 163, 13, 3695, 3703], [163, 170, 6, 747, 753], [170, 179, 13, 3329, 3336], [179, 186, 6, 1048, 1056], [186, 194, 9, 1474, 1483], [194, 202, 6, 747, 753], [202, 211, 13, 3329, 3336], [211, 218, 6, 1048, 1056], [218, 227, 9, 1474, 1483], [227, 235, 11, 1552, 1562], [235, 244, 11, 1552, 1562], [244, 251, 13, 4498, 4504], [251, 259, 13, 4498, 4504], [259, 267, 9, 968, 977], [267, 275, 8, 3122, 3131], [275, 283, 8, 916, 926], [283, 291, 10, 3664, 3673], [291, 300, 10, 3693, 3702], [300, 306, 6, 668, 673], [306, 314, 6, 747, 753], [314, 322, 13, 3329, 3336], [322, 330, 6, 1048, 1056], [330, 338, 0, 1329, 1335], [338, 346, 6, 747, 753], [346, 354, 13, 3329, 3336], [354, 362, 6, 1048, 1056], [362, 371, 9, 1474, 1483], [371, 378, 6, 747, 753], [378, 386, 13, 3329, 3336], [386, 394, 6, 1048, 1056], [394, 398, 0, 1329, 1335]]

if __name__ == "__main__":


    all_files = os.listdir("asset")
    all_fn = []
    for fn_w_ext in all_files:
        fn = os.path.splitext(fn_w_ext)[0]
        f1 = fn + ".mp4"
        f2 = fn + ".srt"
        if (not fn in all_fn) and (f1 in all_files) and (f2 in all_files):
            all_fn.append(fn)

    if not "song" in all_fn:
        print("Please put song.mp4 and song.srt in asset/ !")
        exit(0)
    
    if len(all_fn) == 1:
        print("Please put <video_name>.mp4 and <video_name>.srt in asset/ !")
        exit(0)

    print("Song: song.{mp4, srt}")
    print("Video(s):")
    for fn in all_fn:
        if fn != "song":
            print("\t" + fn + ".{mp4, srt}")

    os.makedirs("asset/beats", exist_ok=True)

    video_captions, video_fn, vbeats = [], [], []
    for fn in all_fn:
        if fn != "song":
            vbeats.append(video_get_beats("asset/" + fn + ".mp4", "asset/beats/" + fn + ".beats"))
            video_fn.append(fn)
            with open("asset/" + fn + ".srt") as f:
                video_captions.append(list(srt.parse(f.read())))

    video_captions, video_fn, vbeats = [], [], []
    for fn in all_fn:
        if fn != "song":
            vbeats.append(video_get_beats("asset/" + fn + ".mp4", "asset/beats/" + fn + ".beats"))
            video_fn.append(fn)
            with open("asset/" + fn + ".srt") as f:
                video_captions.append(list(srt.parse(f.read())))

    abeats = audio_get_beats("asset/song.mp4", "asset/beats/song.beats")
    print(len(abeats))


    L=[]
    Warp_video = []
    for files in video_captions:
        # 拼接成完整路径
        filePath = "asset/" + fn + ".mp4"
        # 载入视频
        video = VideoFileClip(filePath)
        # 添加到数组
        L.append(video)
    audio_ = VideoFileClip("asset/song.mp4")

    synch_video_beat = 0
    synch_audio_beat = 0
    nbeats = 0

    clip_ = 0
    os.makedirs("clip_video", exist_ok=True)
    for beat in warp:
        if(clip_ >= len(warp)-1):
            continue
        video_index = beat[2]
        video_start = beat[3]
        video_end = beat[4]
        v_beat = vbeats[video_index]
        t_start = v_beat[video_start]
        t_end = v_beat[video_end]
        
        video_clip = L[video_index].subclip(t_start,t_end)

        Warp_video.append(video_clip)


        clip_+=1



    final_clip = concatenate_videoclips(Warp_video)
    final_clip.to_videofile("./clip_video/final_clip.mp4", remove_temp=False)


    Song=SourceMedia(path = './asset/song.mp4')
    vb.PullVideo(name=Song.name, source_location=Song.path)
    vid2 = vb.LoadVideo(name=Song.name)
    Video_=SourceMedia(path = './clip_video/final_clip.mp4')
    vb.PullVideo(name=Video_.name, source_location=Video_.path)
    vid = vb.LoadVideo(name=Video_.name)
    output_path = './Final_result.mp4'
    warped = vb.Dancify(source_video=vid, target=vid2, synch_video_beat=synch_video_beat,
                        synch_audio_beat=synch_audio_beat, force_recompute=True, warp_type = 'quad',
                        nbeats=nbeats, output_path = output_path)
    # 拼接视频
    
    



