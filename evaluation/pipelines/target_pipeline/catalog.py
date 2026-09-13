"""Legacy strict-chain reference model; production entrypoints use explicit Features."""
from __future__ import annotations

from pathlib import Path
import torch
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from .core import Stage, Workload, Prepare, Finalize
from .operators import (
    ReadImage, to_float, GaussianBlur, Solarize, clean_caption,
    TruncateCaption, TokenizeCaption, tensorize_tokens,
)

NAMES = ("simclr", "dino", "swav", "clip", "blip")
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)


def build_workload(name, *, seed=0, epoch=0, image_root="",
                   tokenizer_path="openai/clip-vit-base-patch32"):
    if name not in NAMES:
        raise ValueError(f"Unknown workload {name!r}; choose from {NAMES}")
    stages = []

    def chain(prefix, source, target, entries, modality="image"):
        parent = ()
        for label, op, scaling, stochastic in entries:
            tag = f"{prefix}_{label}"
            stages.append(Stage(tag, source, target, op, parent,
                                modality, scaling, stochastic))
            source, parent = target, (tag,)

    if name == "simclr":
        # Two standard independent views; per-view operators/parameters are
        # exactly the existing Cedar SimCLRV2Feature (244, not 224).
        for view in range(2):
            chain(f"view{view}", "original", f"view{view}", [
                ("float", to_float, "expand_dtype", False),
                ("crop", T.RandomResizedCrop((244,244)), "resize", True),
                ("flip", T.RandomHorizontalFlip(), "none", True),
                ("jitter", T.ColorJitter(0.1,0.1,0.1,0.1), "none", True),
                ("gray", T.Grayscale(num_output_channels=1), "shrink_channels", False),
                ("blur", T.GaussianBlur(11), "none", True),
                ("normalize", T.Normalize((0.1307,), (0.3081,)), "none", False),
            ])
        views, kind = 2, "tensor"

    elif name in ("dino", "swav"):
        views, kind = (10 if name == "dino" else 8), "pil"
        for view in range(views):
            global_view = view < 2
            size = 224 if global_view else 96
            if name == "dino":
                scale = (0.4,1.0) if global_view else (0.05,0.4)
                crop = T.RandomResizedCrop(size, scale=scale,
                                          interpolation=InterpolationMode.BICUBIC)
                remaining = [
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomApply([T.ColorJitter(0.4,0.4,0.2,0.1)],p=0.8),
                    T.RandomGrayscale(p=0.2),
                    GaussianBlur(1.0 if view == 0 else 0.1 if view == 1 else 0.5),
                ]
                if view == 1:
                    remaining.append(Solarize(0.2))
                std = (0.229,0.224,0.225)
            else:
                scale = (0.14,1.0) if global_view else (0.05,0.14)
                crop = T.RandomResizedCrop(size, scale=scale)
                remaining = [
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomApply([T.ColorJitter(0.8,0.8,0.8,0.2)],p=0.8),
                    T.RandomGrayscale(p=0.2),
                    GaussianBlur(0.5, numpy_probability=True),
                ]
                # Preserve upstream SwAV's 0.228, not the usual 0.229.
                std = (0.228,0.224,0.225)
            remaining += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, std)]
            chain(f"view{view}", "original", f"view{view}", [
                ("crop", crop, "resize", True),
                ("augment", T.Compose(remaining), "expand_dtype", True),
            ])

    elif name == "clip":
        chain("image", "image", "pixels", [
            ("read", ReadImage("tensor"), "decode", False),
            ("resize", T.Resize([224], interpolation=InterpolationMode.BICUBIC),
             "resize", False),
            ("crop", T.CenterCrop(224), "shrink_spatial", False),
            ("float", T.ConvertImageDtype(torch.float32), "expand_dtype", False),
            ("normalize", T.Normalize(CLIP_MEAN, CLIP_STD), "none", False),
        ])
        chain("text", "caption", "tokens", [
            ("tokenize", TokenizeCaption(tokenizer_path), "tokenize_pad_truncate", False),
            ("tensor", tensorize_tokens, "representation", False),
        ], modality="text")

    else:
        from .references.blip.transform.randaugment import RandomAugment
        chain("image", "image", "pixels", [
            ("read", ReadImage(), "decode", False),
            ("crop", T.RandomResizedCrop(224,scale=(0.5,1.0),
                      interpolation=InterpolationMode.BICUBIC), "resize", True),
            ("flip", T.RandomHorizontalFlip(), "none", True),
            ("augment", RandomAugment(2,5,isPIL=True,augs=[
                "Identity","AutoContrast","Brightness","Sharpness","Equalize",
                "ShearX","ShearY","TranslateX","TranslateY","Rotate",
            ]), "representation", True),
            ("tensor", T.ToTensor(), "expand_dtype", False),
            ("normalize", T.Normalize(CLIP_MEAN,CLIP_STD), "none", False),
        ])
        chain("text", "caption", "caption", [
            ("clean", clean_caption, "text_normalize", False),
            ("truncate", TruncateCaption(30), "shrink_text", False),
        ], modality="text")

    if name in ("simclr","dino","swav"):
        prepare = Prepare(seed, epoch, image_root, kind)
        finalize = Finalize("views", tuple(f"view{i}" for i in range(views)))
    else:
        prepare = Prepare(seed, epoch, image_root)
        finalize = Finalize(name, ())
    return Workload(name, tuple(stages), prepare, finalize)
