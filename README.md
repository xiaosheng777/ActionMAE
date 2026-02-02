# ActionMAE
Pytorch code for our AAAI 2023 paper ["Towards Good Practices for Missing Modality Robust Action Recognition"](https://arxiv.org/abs/2211.13916).

### Action Recognition with Missing Modality
<div align="center">
  <img width="50%" alt="Missing Modality Action Recognition" src="figure/missing_modality.png">
</div>
<div align="center">
	Standard multi-modal action recognition assumes that the modalities used in the training stage are complete at inference time: (a) → (b). We address the action recognition problem in situations where such assumption is not established,
	i.e., when modalities are incomplete at inference time: (a) →
	(c). Our goal is to maintain performance in the absence of
	any input modality.
</div>

---

## Get Started
```
$ git clone https://github.com/sangminwoo/ActionMAE.git
$ cd ActionMAE
```

## Dependencies 
- Pytorch 1.11.0
- CUDA Toolkit 11.3
- NVIDIA Apex

### Environment Setup
- Install [Pytorch 1.11.0](https://pytorch.org/get-started/previous-versions/#linux-and-windows-4) with the following command.

```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

- Goto [NVIDIA Apex](https://github.com/NVIDIA/apex#linux), and follow the instruction.

- See requirements.txt for all python dependencies, and you can install them using the following command.

```
$ pip install -r requirements.txt
```

## Train & Eval

```
$ ./train_val_actionmae_multigpu.sh
```

See/modify configurations in ``ActionMAE/lib/configs.py``

### Food-101 Image Classification (torchvision)
If you want to run image classification instead of video classification, you can
use the torchvision Food-101 dataset. Each image is repeated to form a
1-frame (or multi-frame) clip so it can pass through the ActionMAE pipeline.

Example command (single-GPU):
```
torchrun --nproc_per_node=1 train_val_actionmae_multigpu.py \
  --dataset food101_image \
  --modality rgb \
  --food101_dir /path/to/food101 \
  --food101_download \
  --num_frames 1 \
  --img_size 224 \
  --model actionmae \
  --model_size small \
  --fusion sum \
  --end_epoch 50 \
  --bs 8 \
  --eval_bs 8
```

### UPMC Food-101 (RGB only)
This repo can be run on the UPMC Food-101 dataset using RGB frames only.
Prepare the dataset as extracted frames and update the dataset root to
`--upmc_food101_dir`. The loader expects one of the following layouts:

**Option A: folder splits**
```
upmc_food101/
  train/
    apple_pie/
      video_0001/
        000001.jpg
        000002.jpg
    ...
  val/
    apple_pie/
      video_0123/
        000001.jpg
  test/  # optional
```

**Option B: split files**
```
upmc_food101/
  splits/
    train.txt
    val.txt
    test.txt
    class_list.txt  # optional, list of class names (one per line)
```
Each line in `train.txt`/`val.txt`/`test.txt` should be:
```
class_name/video_dir [label]
```
If `label` is omitted, it is inferred from `class_list.txt` or the
alphabetical order of class folders.

Example command (single-GPU):
```
torchrun --nproc_per_node=1 train_val_actionmae_multigpu.py \
  --dataset upmc_food101 \
  --modality rgb \
  --upmc_food101_dir /path/to/upmc_food101 \
  --num_frames 16 \
  --img_size 224 \
  --model actionmae \
  --model_size small \
  --fusion sum \
  --end_epoch 50 \
  --bs 8 \
  --eval_bs 8
```

## Citation

    @inproceedings{woo2023towards,
      title={Towards Good Practices for Missing Modality Robust Action Recognition},
      author={Woo, Sangmin and Lee, Sumin and Park, Yeonju and Nugroho, Muhammad Adi and Kim, Changick},
      booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
      volume={37},
      number={1},
      year={2023}
    }

## Acknowledgement
We appreciate much the nicely organized codes developed by [MAE](https://github.com/facebookresearch/mae) and [pytorch-image-models](https://github.com/rwightman/pytorch-image-models). Our codebase is built on them.
