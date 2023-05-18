from typing import Optional

from PIL import Image
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision


def load_image(image_path: str) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    xforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    # add empty batch dim before returning
    return torch.unsqueeze(xforms(img), dim=0)


def create_dataloader(
    input_tensor: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    batch_size: int = 1,
    num_workers: int = 1,
) -> DataLoader:
    # create a dataset with fake labels
    labels = labels or torch.Tensor([-1] * len(input_tensor))
    dataset = TensorDataset(input_tensor, labels)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
