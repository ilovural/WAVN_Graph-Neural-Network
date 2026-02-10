'''
-dataset.py reads labels.csv & loads image pairs
-creates graph data samples using building_graph
'''

import torch
from torch.utils.data import Dataset
import pandas as pd
from building_graph import BuildGlobalGraphFromCSV
import os
from PIL import Image

#gives ONE global graph for GCN to learn from
class NavGraphDataset:
    def __init__(self, CSVpath, imageDirectory, device): #constructor
        self.df = pd.read_csv(CSVpath) #loads csv file to pandas d-f
        directions = sorted(self.df["direction"].unique())
        self.LabelMap = {d: i for i, d in enumerate(directions)}
        self.inverseLabelMap = {i: d for d, i in self.LabelMap.items()}

        self.graph = BuildGlobalGraphFromCSV(
            self.df, imageDirectory, self.LabelMap, device
        )

    def get_graph(self):
        return self.graph #returns global graph to feed into GCN 
        #(GCN is only training on edges not individual images)

    
#added below function for edge images
class ImageDataset(Dataset):
    #using rgb and edge segmented images
    def __init__(self, csv_file, rgb_dir, edge_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.rgb_dir = rgb_dir
        self.edge_dir = edge_dir
        self.transform = transform

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]["image"]

        #pathbuilding for corresponding imahes
        rgb_path = os.path.join(self.rgb_dir, img_name)
        edge_path = os.path.join(self.edge_dir, img_name)

        rgb = Image.open(rgb_path).convert("RGB")
        edge = Image.open(edge_path).convert("L")  # single channel

        if self.transform:
            rgb = self.transform(rgb)
            edge = self.transform(edge)

        # concatenate channels: [4, H, W], 4-channel tensor
        image = torch.cat([rgb, edge], dim=0)

        label = self.data.iloc[idx]["label"] #target value
        return image, label
