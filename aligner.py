import srt
import argparse
import torch
from semantic_matching.model import SemanticMatchingModel

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--song", default=None, type=str)
    parser.add_argument("--k", default=5, type=int)
    return parser.parse_args()

def caption_to_text(caption):
    return [sub.content.replace("\n", "").replace("\r", "") for sub in caption]

if __name__ == "__main__":
    args = get_args()

    with open("asset/RAW VIDEO_ Trump speaks at NC GOP convention - Part 1 - English (auto.srt") as f:
        video_caption =  list(srt.parse(f.read()))
    
    with open("asset/Rick-Astley-Never-Gonna-Give-You-Up.srt") as f:
        audio_caption =  list(srt.parse(f.read()))
    
    video_text = caption_to_text(video_caption)
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
            print("\t", video_text[idx_topk[i, k]], sim_topk[i, k].item())
