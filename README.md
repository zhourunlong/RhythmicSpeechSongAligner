# Rhythmic Speech-Song Aligner
RhythmicSpeechSongAligner is a course project for Multimedia Computing, instructed by Hang Zhao, IIIS, Tsinghua.

## Usage

1. Download model checkpoint from https://cloud.tsinghua.edu.cn/f/a9c8ab6d173347b7bd04/?dl=1, and extract to `semantic_matching/`.

If you want to produce our results by yourself, we recommend you download our pre-processed files for efficiency (computing from scratch takes hours):

2. Download audio and video files from https://cloud.tsinghua.edu.cn/f/985254fe99224962abb2/?dl=1, and extract to `asset/`.

3. Download pre-computed beats from https://cloud.tsinghua.edu.cn/f/4f8b7fc25f3d43e79679/?dl=1, and extract to `asset/beats/`.

4. Download pre-computed VisBeat assets from https://cloud.tsinghua.edu.cn/f/57600a1fab6b4e169ffb/?dl=1, and extract to `VisBeatAssets/`.

    *This configuration uses 'Never Gonna Give You Up' as the song. If you want to produce the result of 'My Love', download https://cloud.tsinghua.edu.cn/f/7d0480de4b314f2a8fe0/?dl=1 and override the files.*

5. Run `CUDA_VISIBLE_DEVICES=<gpu_id> TOKENIZERS_PARALLELISM=false python aligner.py`.

Otherwise, to produce your own results:

2. Put your `song.{mp4, srt}` and `<videos>.{mp4, srt}` under `asset/`.

3. Run `CUDA_VISIBLE_DEVICES=<gpu_id> TOKENIZERS_PARALLELISM=false python aligner.py`.

## Our results

See `results/`.
