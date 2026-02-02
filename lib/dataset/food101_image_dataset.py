import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms as T


class Food101ImageDataset(Dataset):
    """Food-101 image dataset wrapper for ActionMAE image classification."""

    def __init__(self, phase='train', img_size=224, num_frames=1,
                 modality=None, temporal_augmentation=True,
                 food101_dir='/dataset/food101/',
                 food101_download=False,
                 **kwargs):
        super().__init__()
        assert phase in ['train', 'val', 'test'], f'{phase} is not available. should be one of train/val/test.'
        self.phase = phase
        self.img_size = img_size
        self.num_frames = num_frames
        self.modality = modality or ['rgb']
        if self.modality != ['rgb'] and self.modality != ['rgb',]:
            raise ValueError('Food-101 image dataset supports RGB only. Set --modality rgb.')

        split = 'train' if phase == 'train' else 'test'
        self.dataset = datasets.Food101(
            root=food101_dir,
            split=split,
            download=food101_download,
            transform=self._build_transform(phase, temporal_augmentation)
        )

    def _build_transform(self, phase, temporal_augmentation):
        if phase == 'train' and temporal_augmentation:
            return T.Compose([
                T.CenterCrop(size=self.img_size),
                T.RandomCrop(size=0.8 * self.img_size),
                T.Resize(size=self.img_size),
                T.ColorJitter(brightness=.5, hue=.3),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        return T.Compose([
            T.CenterCrop(size=0.8 * self.img_size),
            T.Resize(size=self.img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.num_frames > 1:
            image = image.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)
        else:
            image = image.unsqueeze(0)
        return dict(model_inputs={'rgb': image}, target=label)
