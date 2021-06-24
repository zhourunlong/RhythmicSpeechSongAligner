import srt
import argparse
import torch
import os, sys, time
from semantic_matching.model import SemanticMatchingModel
import visbeat as vb
import numpy as np
import pickle
from operator import truediv
from moviepy.editor import VideoFileClip, concatenate_videoclips
import pathos

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", default=5, type=int)
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--gap-threshold", default=5, type=int)
    parser.add_argument("--walk-prob", default=0.3, type=float)
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
            abeats = pickle.load(f)
    else:
        audio_file = SourceMedia(path_to_file)
        vb.PullVideo(name=audio_file.name, source_location=audio_file.path)
        vid = vb.LoadVideo(name=audio_file.name)
        vid_audio = vid.getAudio()
        abeats = vid_audio.getBeatEvents()
        f = open(path_to_pickle, 'wb')
        pickle.dump(abeats, f)
    
    beats = [float(beat.start) for beat in abeats]
    return abeats, beats

def video_get_beats(path_to_file, path_to_pickle):
    if os.path.exists(path_to_pickle):
        with open(path_to_pickle, 'rb') as f:
            vbeats = pickle.load(f)
    else:
        video_file = SourceMedia(path_to_file)
        vb.PullVideo(name=video_file.name, source_location=video_file.path)
        vid = vb.LoadVideo(name=video_file.name)
        vbeats = vid.getVisualBeats()
        f = open(path_to_pickle, 'wb')
        pickle.dump(vbeats, f)
    
    beats = [float(beat.start) for beat in vbeats]
    return vbeats, beats

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

def get_file(path_to_file):
    video_file = SourceMedia(path_to_file)
    return vb.LoadVideo(name=video_file.name)

def walk(arr, n, prob):
    k = len(arr)
    narr = []
    probs = np.random.random((n,))
    pos = 0
    for i in range(n):
        narr.append(arr[pos])
        if pos == 0:
            pos += 1
        elif pos == k - 1:
            pos -= 1
        else:
            if probs[i] < prob:
                pos -= 1
            else:
                pos += 1
    return narr

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

    video_captions, video_fn, vbeats_event, vbeats = [], [], [], []
    for fn in all_fn:
        if fn != "song":
            _vbeats_event, _vbeats = video_get_beats("asset/" + fn + ".mp4", "asset/beats/" + fn + ".beats")
            vbeats_event.append(_vbeats_event)
            vbeats.append(_vbeats)
            video_fn.append(fn)
            with open("asset/" + fn + ".srt") as f:
                video_captions.append(list(srt.parse(f.read())))
    
    abeats_event, abeats = audio_get_beats("asset/song.mp4", "asset/beats/song.beats")
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
    model = model.cuda()
    
    with torch.no_grad():
        batched_similarities = []
        for b in range(0, len(video_text), args.batch_size):
            batched_similarities.append(model.calc_similarity(audio_text, video_text[b:b+args.batch_size]))
    similarity = torch.cat(batched_similarities, 1)
    sim_topk, idx_topk = torch.topk(similarity, args.k)

    warps = []
    for i in range(len(audio_text)):
        #print(audio_text[i])
        atime_start = audio_caption[i].start.total_seconds()
        atime_end = audio_caption[i].end.total_seconds()
        apos_start = lower_bound(abeats, atime_start) 
        apos_end = lower_bound(abeats, atime_end) - 1
        #print(atime_start, atime_end, abeats[apos_start:apos_end])
        abeats_count = apos_end - apos_start
        #print(abeats_count)

        opt_diff = 999999999

        for k in range(args.k):
            idx = idx_topk[i, k]
            #print("\t", video_text[idx], "%1.3f" % (sim_topk[i, k].item()), video_fn[source[idx][0]], source[idx][1])
            vcap = video_captions[source[idx][0]]
            vtime_start = vcap[source[idx][1]].start.total_seconds()
            vtime_end = vcap[source[idx][1]].end.total_seconds()
            beats = vbeats[source[idx][0]]
            vpos_start = lower_bound(beats, vtime_start)
            vpos_end = lower_bound(beats, vtime_end) - 1
            
            #print(vtime_start, vtime_end, beats[vpos_start:vpos_end])
            vbeats_count = vpos_end - vpos_start
            #print("\t", vbeats_count)

            # try to find a close number of beats
            tmp = abs(vbeats_count - abeats_count)
            if tmp < opt_diff:
                opt_idx = idx
                opt_diff = tmp
                opt_vpos_start, opt_vpos_end = vpos_start, vpos_end
        
        print(audio_text[i], "--->", video_text[opt_idx])
        #vcap = video_captions[source[opt_idx][0]]
        #print("\t", video_fn[source[opt_idx][0]], vcap[source[opt_idx][1]].start.total_seconds(), vcap[source[opt_idx][1]].end.total_seconds())
        # warp audio beats [apos_start, apos_end) with <video_idx>th video's [opt_vpos_start, opt_vpos_end) beats
        warps.append([apos_start, apos_end, source[opt_idx][0], opt_vpos_start, opt_vpos_end])
    
    # fill gaps between audio beats
    n = len(warps)
    for i in range(n - 1):
        if warps[i + 1][0] - warps[i][1] < args.gap_threshold:
            # tiny gap, continue the former video beats
            warps[i][1] = warps[i + 1][0]
        else:
            # big gap, choose another video clip, randomly
            idx = np.random.randint(len(vbeats))
            gap = np.random.randint(1, warps[i + 1][0] - warps[i][1] + 1)
            pos_start = np.random.randint(len(vbeats[idx]) - gap)
            warps.append([warps[i][1], warps[i + 1][0], idx, pos_start, pos_start + gap])
    
    if warps[0][0] != 0:
        idx = np.random.randint(len(vbeats))
        gap = np.random.randint(int(warps[0][0] * 0.66), warps[0][0] + 1)
        pos_start = np.random.randint(len(vbeats[idx]) - gap)
        warps.append([0, warps[0][0], idx, pos_start, pos_start + gap])
    
    if warps[n - 1][1] < len(abeats) - 1:
        idx = np.random.randint(len(vbeats))
        L = len(abeats) - warps[n - 1][1] - 1
        gap = np.random.randint(int(L * 0.66), L + 1)
        pos_start = np.random.randint(len(vbeats[idx]) - gap)
        warps.append([warps[n - 1][1], len(abeats) - 1, idx, pos_start, pos_start + gap])

    warps.sort(key=lambda x:x[0])
    #print(warps)

    song = get_file("asset/song.mp4")
    
    speeches = []
    for fn in video_fn:
        speeches.append(get_file("asset/" + fn + ".mp4"))

    sampling_rate = song._getFrameRate()
    output_audio = song.getAudio()

    video_clips = []

    os.makedirs("temp", exist_ok=True)

    tidx = -1
    for warp_info in warps:
        tidx += 1
        #print(warp_info)
        apos_start, apos_end, idx, vpos_start, vpos_end = warp_info
        target_start, target_end = abeats[apos_start], abeats[apos_end]
        target_duration = target_end - target_start

        new_n_samples = int(target_duration * sampling_rate + 0.5)
        target_start_times = np.linspace(target_start, target_end, num=new_n_samples, endpoint=False)
        unwarped_target_times = []
   
        source_events = vbeats_event[idx][vpos_start:vpos_end+1]
        #tmp = [float(_.start) for _ in source_events]
        #print(tmp)
        #print(vbeats[idx][vpos_start:vpos_end+1])
        source_events = walk(source_events, apos_end + 1 - apos_start, args.walk_prob)
        target_events = abeats_event[apos_start:apos_end+1]
        #print(source_events, target_events)
        warp = vb.Warp.FromEvents(source_events, target_events)
        warp.setWarpFunc("quad")
        for st in target_start_times:
            unwarped_target_times.append(warp.warpTargetTime(st))
        #print(apos_end - apos_start, vpos_end - vpos_start)
        #print(unwarped_target_times)

        #if tidx == 1:
        #    break
        #continue

        vid = speeches[idx]
        old_frame_time = truediv(1.0, vid._getFrameRate())
        frame_index_floats = np.true_divide(np.array(unwarped_target_times), old_frame_time)
        tempfilepath = vb.get_temp_file_path(final_file_path="temp/clip" + str(tidx) + ".mp4", temp_dir_path = vb.Video.VIDEO_TEMP_DIR)
        vid.openVideoWriter(output_file_path=tempfilepath, fps=sampling_rate)
        #start_timer = time.time()
        #last_timer = start_timer
        #fcounter = 0
        
        for nf in range(len(frame_index_floats)):
            try:
                nwfr = vid.getFrame(frame_index_floats[nf])
                vid.writeFrame(nwfr)
            except ValueError:
                print("VALUE ERROR ON WRITEFRAME {}".format(frame_index_floats[nf]))
                vid.writeFrame(vid.getFrame(math.floor(frame_index_floats[nf])))

            '''
            fcounter += 1
            if (not (fcounter % 50)):
                if ((time.time() - last_timer) > 10):
                    last_timer = time.time()
                    print("{}%% done after {} seconds...".format(100.0 * truediv(fcounter, len(frame_index_floats)), last_timer - start_timer))
            '''
        vid.closeVideoWriter()

        silent_warped_vid = vb.Video(tempfilepath)

        cropped_output_audio = output_audio.AudioClip(start=target_start, end=target_end)

        #print(cropped_output_audio.getDuration(), silent_warped_vid.getDuration())

        rvid = vb.Video.CreateFromVideoAndAudioObjects(video=silent_warped_vid, audio=cropped_output_audio, output_path="temp/clip" + str(tidx) + ".mp4")
        os.remove(tempfilepath)
    
    clips = []
    for i in range(tidx):
        clips.append(VideoFileClip("temp/clip" + str(i) + ".mp4"))

    finalclip = concatenate_videoclips(clips)
    finalclip.write_videofile("output.mp4")
