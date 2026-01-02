"""
- This file building_graph.py handles creation of graph structures from dataset. 
- The 'current' and 'destination' image pairs are treated as a connection in a graph. 
- The direction label (left, right, forward) are the target for training.
- Extracts image features using a pretrained CNN backbone.
"""

import os
import torch
import torchvision.transforms as T
import torchvision.models as models
from torch_geometric.data import Data
from PIL import Image

# CNN featur extractor (options)
def get_feature_extractor(backbone_name: str, device):
    """
    Returns:
        feature_extractor (nn.Module)
        output_feature_dim (int)
    """

    if backbone_name == "resnet18":
        base = models.resnet18(pretrained=True)
        extractor = torch.nn.Sequential(*list(base.children())[:-1])
        out_dim = 512

    elif backbone_name == "resnet34":
        base = models.resnet34(pretrained=True)
        extractor = torch.nn.Sequential(*list(base.children())[:-1])
        out_dim = 512

    elif backbone_name == "mobilenet_v3":
        base = models.mobilenet_v3_small(pretrained=True)
        extractor = torch.nn.Sequential(
            base.features,
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )
        out_dim = 576

    elif backbone_name == "efficientnet_b0":
        base = models.efficientnet_b0(pretrained=True)
        extractor = torch.nn.Sequential(
            base.features,
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )
        out_dim = 1280

    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    extractor.eval().to(device)
    return extractor, out_dim


# image to feature vetcor
def ExtractImageFeatures(imagePath, model, transform, device):
    image = Image.open(imagePath).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model(image)

    features = torch.flatten(features, start_dim=1)
    return features.squeeze(0)

#graph builder
def BuildGraphFromCSV(csv_df, imageDirectory, labelMap, device):
    '''TO SWITCH CNN BACKBONES LOOK AT LINE BELOW!!!'''
    backbone_name = "efficientnet_b0"   # TO SWITCH CNN BACKBONES 
    featureExtractor, feature_dim = get_feature_extractor(backbone_name, device)

    print(f"Using CNN Feature Extractor: {backbone_name} ({feature_dim}D)")

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataList = []
    cache = {}

    for _, row in csv_df.iterrows():
        currentImage = row["current_image"].replace(".png", "_hed.png")
        destinationImage = row["destination_image"].replace(".png", "_hed.png")
        direction = row["direction"]

        currentPath = os.path.join(imageDirectory, currentImage)
        destinationPath = os.path.join(imageDirectory, destinationImage)

        # validate paths BEFORE loading
        if not os.path.exists(currentPath):
            raise FileNotFoundError(f"Missing image: {currentPath}")
        if not os.path.exists(destinationPath):
            raise FileNotFoundError(f"Missing image: {destinationPath}")

        if currentImage not in cache:
            cache[currentImage] = ExtractImageFeatures(
                currentPath, featureExtractor, transform, device
            )

        if destinationImage not in cache:
            cache[destinationImage] = ExtractImageFeatures(
                destinationPath, featureExtractor, transform, device
            )

        # two-node graph
        x = torch.stack([cache[currentImage], cache[destinationImage]])
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        y = torch.tensor([labelMap[direction]], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)
        dataList.append(data)

    return dataList