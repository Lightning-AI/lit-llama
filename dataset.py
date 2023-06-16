from torch.utils.data import Dataset
import json
from lit_llama.tokenizer import Tokenizer
import os
from PIL import Image
from scripts.prepare_alpaca import prepare_sample
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader
import torch


coco_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=InterpolationMode.BICUBIC, antialias=None),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
])


class LLaVAInstruct(Dataset):
    def __init__(self, config_path="data/datasets/llava-instruct/llava_instruct_150k.json", coco_root="data/datasets/coco/", max_length=256, img_transform=coco_transform):
        
        self.annotations = json.load(open(config_path))
        self.transform = img_transform
        self.max_length = max_length
        self.tokenizer = Tokenizer("checkpoints/lit-llama/tokenizer.model")
        self.coco_root = coco_root

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        annotation = self.annotations[index]
        image_name = annotation['image']
        question = annotation['conversations'][0]['value']
        answer = annotation['conversations'][1]['value']
        filename = os.path.join(self.coco_root, "train2014", f"COCO_train2014_" + image_name)
        image = Image.open(filename).convert('RGB')
        image = self.transform(image)
        example = {"instruction": question, "input": "", "output": answer}
        prepared = prepare_sample(example, self.tokenizer, max_length=self.max_length, mask_inputs=False)
        return {
            "image": image, 
            "input_ids": prepared["input_ids"].type(torch.int64), 
            "labels": prepared["labels"].type(torch.int64),
        }



def collate_fn(samples):
    images = [item["image"] for item in samples]
    input_ids = [item["input_ids"] for item in samples]
    labels = [item["labels"] for item in samples]

    max_len = max(len(s) for s in input_ids)

    def pad_right(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_right(x, pad_id=-1) for x in labels])
    img = torch.stack(images)
    return img, x, y


def get_dataloader(batch_size=1, num_workers=0, img_transform=coco_transform):
    dataset = LLaVAInstruct(img_transform=img_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
    return dataloader


if __name__ == "__main__":
    # dataset = LLaVAInstruct()
    dataloader = get_dataloader()
    for batch in iter(dataloader):
        print(batch[0].shape)
        print(batch[1].shape)
        print(batch[2].shape)
