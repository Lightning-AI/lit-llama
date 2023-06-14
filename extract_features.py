from  torchvision.datasets import CocoCaptions
import torchvision.transforms as transforms
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import clip
from timm.models.vision_transformer import Block as ViTBlock
import torch


def clip_encode_image(clip_model, x):
    # modified from CLIP
    x = clip_model.visual.conv1(x)  # shape = [*, width, grid, grid]
    # shape = [*, width, grid ** 2]
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    x = torch.cat([clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
    x = x + clip_model.visual.positional_embedding.to(x.dtype)
    x = clip_model.visual.ln_pre(x)

    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.visual.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD

    # preserve all spatial tokens
    x = clip_model.visual.ln_post(x[:, :, :])

    if clip_model.visual.proj is not None:
        x = x @ clip_model.visual.proj

    return x


clip_model, clip_transform = clip.load("ViT-L/14")

train_captions = CocoCaptions(
    root="/data/shared/datasets/coco/train2014/",
    annFile="/data/shared/datasets/coco/annotations/captions_train2014.json",
    transform=clip_transform
)

val_captions = CocoCaptions(
    root="/data/shared/datasets/coco/val2014/",
    annFile="/data/shared/datasets/coco/annotations/captions_val2014.json",
    transform=clip_transform
)

print('Number of samples: ', len(val_captions) + len(train_captions))
img, target = val_captions[3]
print(target)


print("clip_dim", clip_model.visual.proj.shape[1])


device = torch.device("cuda", 0)
clip_model.to(device)
with torch.cuda.amp.autocast():
    img = img.unsqueeze(0).to(device)
    clip_feats = clip_encode_image(clip_model, img)
print(clip_feats.shape)

torch.save({"clip_feats": clip_feats, "captions": target}, "features-0.pt")
