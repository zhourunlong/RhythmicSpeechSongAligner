import srt
import argparse
import torch
import os, sys
from semantic_matching.model import SemanticMatchingModel
import visbeat as vb
import numpy as np
import pickle

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", default=5, type=int)
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--gap-threshold", default=5, type=int)
    return parser.parse_args()

def caption_to_text(caption):
    return [sub.content.replace("\n", " ").replace("\r", " ") for sub in caption]

class SourceMedia:
    def __init__(self, path, name=None):
        self.path = path
        self.name = os.path.splitext(os.path.basename(self.path))[0] if name is None else name

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

def lower_bound(nums, target):
    low, high = 0, len(nums) - 1
    pos = len(nums)
    while low < high:
        mid = (low + high) // 2
        if nums[mid] < target:
            low = mid + 1
        else:
            high = mid
    if nums[low] >= target:
        pos = low
    return pos

if __name__ == "__main__":
    args = get_args()

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
    
    abeats = audio_get_beats("asset/song.mp4", "asset/beats/song.beats")
    with open("asset/song.srt") as f:
        audio_caption =  list(srt.parse(f.read()))

    video_text, source = [], []
    cnt = 0
    for vcap in video_captions:
        video_text += caption_to_text(vcap)
        source += [(cnt, _) for _ in range(len(vcap))]
        cnt += 1
    audio_text = caption_to_text(audio_caption)

    print(len(audio_text), "sentences in song,", len(video_text), "sentences in video(s).")

    model = SemanticMatchingModel()
    state_dict = torch.load("semantic_matching/ft_best.pt")
    model.load_state_dict(state_dict["model"])
    model.cuda()

    with torch.no_grad():
        batched_similarities = []
        for b in range(0, len(video_text), args.batch_size):
            batched_similarities.append(model.calc_similarity(audio_text, video_text[b:b+args.batch_size]))
    similarity = torch.cat(batched_similarities, 1)
    sim_topk, idx_topk = torch.topk(similarity, args.k)

    warp = []
    for i in range(len(audio_text)):
        print(audio_text[i])
        atime_start = audio_caption[i].start.total_seconds()
        atime_end = audio_caption[i].end.total_seconds()
        apos_start = lower_bound(abeats, atime_start) 
        apos_end = lower_bound(abeats, atime_end)
        #print(atime_start, atime_end, abeats[apos_start:apos_end])
        abeats_count = apos_end - apos_start

        opt_diff = 999999999

        for k in range(args.k):
            idx = idx_topk[i, k]
            print("\t", video_text[idx], "%1.3f" % (sim_topk[i, k].item()), video_fn[source[idx][0]], source[idx][1])
            vcap = video_captions[source[idx][0]]
            vtime_start = vcap[source[idx][1]].start.total_seconds()
            vtime_end = vcap[source[idx][1]].end.total_seconds()
            beats = vbeats[source[idx][0]]
            vpos_start = lower_bound(beats, vtime_start)
            vpos_end = lower_bound(beats, vtime_end)
            
            #print(vtime_start, vtime_end, beats[vpos_start:vpos_end])
            vbeats_count = vpos_end - vpos_start

            # try to find a close number of beats
            tmp = abs(vbeats_count - abeats_count)
            if tmp < opt_diff:
                opt_diff = tmp
                video_idx = source[idx][0]
                opt_vpos_start, opt_vpos_end = vpos_start, vpos_end
        
        # warp audio beats [apos_start, apos_end) with <video_idx>th video's [opt_vpos_start, opt_vpos_end) beats
        warp.append([apos_start, apos_end, video_idx, opt_vpos_start, opt_vpos_end])
    
    # fill gaps between audio beats
    n = len(warp)
    for i in range(n - 1):
        if warp[i + 1][0] - warp[i][1] < args.gap_threshold:
            # tiny gap, continue the former video beats
            warp[i][1] = warp[i + 1][0]
        else:
            # big gap, choose another video clip, randomly
            idx = np.random.randint(len(vbeats))
            gap = np.random.randint(1, warp[i + 1][0] - warp[i][1] + 1)
            pos_start = np.random.randint(len(vbeats[idx]) - gap)
            warp.append([warp[i][1], warp[i + 1][0], idx, pos_start, pos_start + gap])
    
    if warp[0][0] != 0:
        idx = np.random.randint(len(vbeats))
        gap = np.random.randint(1, warp[0][0] + 1)
        pos_start = np.random.randint(len(vbeats[idx]) - gap)
        warp.append([0, warp[0][0], idx, pos_start, pos_start + gap])
    
    if warp[n - 1][1] != len(abeats):
        idx = np.random.randint(len(vbeats))
        gap = np.random.randint(1, len(abeats) - warp[n - 1][1] + 1)
        pos_start = np.random.randint(len(vbeats[idx]) - gap)
        warp.append([warp[n - 1][1], len(abeats), idx, pos_start, pos_start + gap])

    warp.sort(key=lambda x:x[0])
    print(warp)

    # In the following, do warping
