import os
import random
import torch
from PIL import Image
from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms as T


class UPMCFood101Dataset(Dataset):
    """UPMC Food-101 video dataset loader (RGB only)."""

    def __init__(self, phase='train', img_size=224, num_frames=32,
                 modality=None, temporal_augmentation=True,
                 upmc_food101_dir='/dataset/upmc_food101/',
                 upmc_food101_split=None,
                 **kwargs):
        super(UPMCFood101Dataset).__init__()
        assert phase in ['train', 'val', 'test'], f'{phase} is not available. should be one of train/val/test.'
        self.phase = phase
        self.img_size = img_size
        self.num_frames = num_frames
        self.root = upmc_food101_dir

        self.modality = modality or ['rgb']
        if self.modality != ['rgb'] and self.modality != ['rgb',]:
            raise ValueError('UPMC Food-101 supports RGB only. Set --modality rgb.')

        self.temporal_augmentation = temporal_augmentation and self.phase == 'train'
        self.data_dir = []
        self.targets = []

        class_names = self._load_class_names()
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        split_file = upmc_food101_split or os.path.join(self.root, 'splits', f'{phase}.txt')
        if os.path.isfile(split_file):
            self._load_from_split(split_file)
        else:
            self._load_from_folders()

        if phase == 'train':
            self.augmenters = {
                'rgb': T.Compose([
                    T.CenterCrop(size=self.img_size),
                    T.RandomCrop(size=0.8 * self.img_size),
                    T.Resize(size=self.img_size),
                    T.ColorJitter(brightness=.5, hue=.3),
                    T.RandomHorizontalFlip(p=0.5),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            }
        else:
            self.augmenters = {
                'rgb': T.Compose([
                    T.CenterCrop(size=0.8 * self.img_size),
                    T.Resize(size=self.img_size),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            }

    def _load_class_names(self):
        class_list_candidates = [
            os.path.join(self.root, 'class_list.txt'),
            os.path.join(self.root, 'splits', 'class_list.txt'),
        ]
        for path in class_list_candidates:
            if os.path.isfile(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return [line.strip() for line in f if line.strip()]

        train_root = os.path.join(self.root, 'train')
        if os.path.isdir(train_root):
            return sorted([d for d in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, d))])

        phase_root = os.path.join(self.root, self.phase)
        if os.path.isdir(phase_root):
            return sorted([d for d in os.listdir(phase_root) if os.path.isdir(os.path.join(phase_root, d))])

        raise FileNotFoundError('Unable to find class list or dataset directories for UPMC Food-101.')

    def _load_from_split(self, split_file):
        with open(split_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                rel_path = parts[0]
                if len(parts) > 1:
                    label = int(parts[1])
                else:
                    label = self._label_from_path(rel_path)
                abs_path = os.path.join(self.root, rel_path)
                if not os.path.isdir(abs_path):
                    raise FileNotFoundError(f'Expected frame directory at {abs_path}')
                self.data_dir.append(abs_path)
                self.targets.append(label)

    def _load_from_folders(self):
        split_root = os.path.join(self.root, self.phase)
        if not os.path.isdir(split_root):
            raise FileNotFoundError(
                f'Expected split directory at {split_root}. '
                f'Create it or provide --upmc_food101_split.'
            )
        for class_name in sorted(os.listdir(split_root)):
            class_dir = os.path.join(split_root, class_name)
            if not os.path.isdir(class_dir):
                continue
            label = self.class_to_idx[class_name]
            for video_dir in sorted(glob(os.path.join(class_dir, '*'))):
                if os.path.isdir(video_dir):
                    self.data_dir.append(video_dir)
                    self.targets.append(label)

    def _label_from_path(self, rel_path):
        parts = rel_path.replace('\\', '/').split('/')
        if not parts:
            raise ValueError(f'Invalid path in split file: {rel_path}')
        class_name = parts[0]
        if class_name not in self.class_to_idx:
            raise ValueError(f'Unknown class name {class_name} in split file.')
        return self.class_to_idx[class_name]

    def __len__(self):
        return len(self.data_dir)

    def __getitem__(self, idx):
        model_inputs = {}
        frame_paths = []
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            frame_paths.extend(sorted(glob(os.path.join(self.data_dir[idx], ext))))

        if not frame_paths:
            raise RuntimeError(f'No frames found in {self.data_dir[idx]}')

        video_len = len(frame_paths)
        idxs = list(range(video_len))
        stepsize = video_len / self.num_frames

        if self.temporal_augmentation:
            sampled_idxs = idxs if video_len < self.num_frames else sorted(random.sample(idxs, self.num_frames))
        else:
            sampled_idxs = idxs if video_len < self.num_frames else [idxs[round(stepsize * i)] for i in range(self.num_frames)]

        if 'rgb' in self.modality:
            video_rgb = []
            for i, path in enumerate(frame_paths):
                if i not in sampled_idxs:
                    continue
                try:
                    rgb = Image.open(path).convert('RGB')
                except Exception:
                    print(f'unable to open RGB from {path}.')
                    continue
                rgb = self.augmenters['rgb'](rgb)
                video_rgb.append(rgb)
            if not video_rgb:
                raise RuntimeError(f'No valid frames loaded from {self.data_dir[idx]}')
            model_inputs['rgb'] = torch.stack(video_rgb)

        return dict(model_inputs=model_inputs, target=self.targets[idx])
