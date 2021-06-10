import srt
import argparse
import torch
import os
from semantic_matching.model import SemanticMatchingModel

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--song", default=None, type=str)
    parser.add_argument("--k", default=5, type=int)
    return parser.parse_args()

def caption_to_text(caption):
    return [sub.content.replace("\n", " ").replace("\r", " ") for sub in caption]

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

    video_captions, video_fn = [], []
    for fn in all_fn:
        if fn != "song":
            video_fn.append(fn)
            with open("asset/" + fn + ".srt") as f:
                video_captions.append(list(srt.parse(f.read())))
    
    with open("asset/song.srt") as f:
        audio_caption =  list(srt.parse(f.read()))

    video_text, source = [], []
    cnt = 0
    for vcap in video_captions:
        video_text += caption_to_text(vcap)
        source += [cnt for _ in range(len(vcap))]
        cnt += 1
    audio_text = caption_to_text(audio_caption)

    model = SemanticMatchingModel()
    state_dict = torch.load("semantic_matching/ft_best.pt")
    model.load_state_dict(state_dict["model"])
    model.cuda()

    with torch.no_grad():
        similarity = model.calc_similarity(audio_text, video_text)
    sim_topk, idx_topk = torch.topk(similarity, args.k)

    for i in range(len(audio_text)):
        print(audio_text[i])
        for k in range(args.k):
            idx = idx_topk[i, k]
            print("\t", video_text[idx], "%1.3f" % (sim_topk[i, k].item()), video_fn[source[idx]])
