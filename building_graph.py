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

# CNN feature extractor (creates various options: ResNet, EfficientNet, MobileNet)
def get_feature_extractor(backbone_name: str, device, target_dim=256):
    """
    Returns:
        feature_extractor (nn.Module)
        projection (nn.Module)
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
        raise ValueError("Unknown feature extractor")

    #converts backbone features into fixed-size (256-dimensional) for standardisation
    projection = torch.nn.Linear(out_dim, target_dim)

    extractor.eval().to(device) #no drop-out, moves to CPU/GPU
    projection.to(device) #moves projection layer to same device as extractor

    return extractor, projection, target_dim #components necessary for ftr extraction



# converts image fileto feature vetcor
def ExtractImageFeatures(imagePath, model, projection, transform, device):
    image = Image.open(imagePath).convert("RGB")
    image = transform(image).unsqueeze(0).to(device) #preprocessing

    with torch.no_grad(): #disables gradient computation for inference speed-up
        features = model(image)
        features = torch.flatten(features, start_dim=1)
        features = projection(features)

    return features.squeeze(0)

#filename handling for image files (choosing from images/rgb, need to integrate /edges)
def resolve_image_path(imageDirectory, base_name):
    rgb_dir = os.path.join(imageDirectory, "rgb")
    search_dir = rgb_dir if os.path.exists(rgb_dir) else imageDirectory

    for f in os.listdir(search_dir):
        if f.startswith(base_name) and f.endswith(".png"):
            return os.path.join(search_dir, f)

    raise FileNotFoundError(
        f"No RGB image found for base name '{base_name}' in {search_dir}"
    )


#graph builder for PyTorch Geometric Graph
def BuildGlobalGraphFromCSV(csv_df, imageDirectory, labelMap, device):
    backbone_name = "efficientnet_b0" #TO CHANGE IMAGE EXTRACTING MODEL
    
    #initailises
    featureExtractor, projection, feature_dim = get_feature_extractor(
        backbone_name, device
    )
    print(f"Using CNN Feature Extractor: {backbone_name} with {feature_dim}D")

    #image preprocessing
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    node_features = []
    edge_index = []
    edge_labels = []

    node_id_map = {}   # image_path → node index
    cache = {}

    def get_node_id(image_path):
        if image_path not in node_id_map:
            node_id_map[image_path] = len(node_features)

            if image_path not in cache:
                cache[image_path] = ExtractImageFeatures(
                    image_path, featureExtractor, projection, transform, device
                )

            node_features.append(cache[image_path])
        return node_id_map[image_path]

    #CSV iteration
    for _, row in csv_df.iterrows():
        currentPath = resolve_image_path(imageDirectory, row["current_image"])
        destinationPath = resolve_image_path(imageDirectory, row["destination_image"])

        src = get_node_id(currentPath)
        dst = get_node_id(destinationPath)

        # directed edge
        edge_index.append([src, dst])
        edge_labels.append(labelMap[row["direction"]])

    #for tensor conversion (converts node features into a tensor...edge list/labels into PyG format)
    x = torch.stack(node_features)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_labels = torch.tensor(edge_labels, dtype=torch.long)

    #creation of graph object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_labels   # direction labels of edges
    )

    return data
