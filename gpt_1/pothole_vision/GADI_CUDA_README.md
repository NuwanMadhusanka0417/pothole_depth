# Gadi GPU / CUDA — Quick Reference

HD-EMS training needs a **GPU node** and `--device cuda`. A CPU-only `qsub` (no `ngpus`) will not use CUDA.

## 1. Request an interactive  session

**CPU Only**
```bash
qsub -I -l walltime=12:00:00,mem=190GB,ncpus=12,jobfs=50GB   -P jq77 -l storage=gdata/jq77+scratch/jq77+scratch/mi23
```

**GPU - V100 (gpuvolta):**

```bash
qsub -I -q gpuvolta -l ncpus=12,mem=64GB,ngpus=1,walltime=12:00:00 -P mi23 -l storage=gdata/jq77+scratch/jq77+scratch/mi23
```

**GPU - A100 (gpuhopper):**

```bash
qsub -I -q gpuhopper -l ncpus=12,mem=64GB,ngpus=1,walltime=12:00:00 -P mi23 -l storage=gdata/jq77+scratch/jq77+scratch/mi23
```

## 2. Load environment

```bash
module load cuda
module load ffmpeg
module load python3/3.9.2
source /scratch/jq77/nk8155/seg/bin/activate
cd /scratch/mi23/nuwan/pothole/pothole_depth/gpt_1/pothole_vision
pip install -r requirements.txt
```

Inspect all videos in `data/input/`:

```bash
python scripts/validate_videos.py
python scripts/inspect_video.py
python scripts/run_pipeline.py --mode detection
```

If you see `error reading header` / `ftyp_in_header=False` / size very small, the MP4 on scratch is **corrupt or a partial copy** (common with OneDrive placeholders). Re-upload full files, then check:

```bash
ls -lh data/input/*.mp4
module load ffmpeg
ffprobe -v error -show_format -show_streams data/input/20260725_120804_NF.mp4
```

Re-mux if needed:

```bash
ffmpeg -i broken.mp4 -c:v libx264 -c:a aac fixed.mp4
```



## 3. Check CUDA is available

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: `True` and a GPU name (e.g. `Tesla V100`).



### _Avode "Disk quota exceed" Error white evaluating._

```bash
export MPLCONFIGDIR=/scratch/mi23/nuwan/.cache/matplotlib
export PYTORCH_KERNEL_CACHE_PATH=/scratch/mi23/nuwan/.cache/torch/kernels
export XDG_CACHE_HOME=/scratch/mi23/nuwan/.cache
mkdir -p "$MPLCONFIGDIR" "$PYTORCH_KERNEL_CACHE_PATH"
```



