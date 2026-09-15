import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class HatefulMemesDataset(Dataset):
    """Dataset class for loading multimodal meme samples."""

    def __init__(self, jsonl_path, img_dir, transform=None):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.data = pd.read_json(f, lines=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_path = os.path.join(self.img_dir, row["img"])
        if not os.path.exists(img_path):
            nested_path = os.path.join(self.img_dir, "img", row["img"])
            if os.path.exists(nested_path):
                img_path = nested_path
            else:
                img_path = os.path.join(self.img_dir, row["img"].replace("img/", ""))

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224))

        if self.transform:
            image = self.transform(image)

        text = row["text"]
        label = row["label"]
        return image, text, label


# Backward compatibility alias
hatefulmemesdataset = HatefulMemesDataset
