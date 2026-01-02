'''
This file trains the Graph Neural Network model on the Gazebo 'images' dataset. 
Notes: The model loads the dataset from labels.csv and then initialises the GNN model. 
After initialising, it trains with x-entropy loss to evaluate accuracy. 
'''

import torch 
from torch_geometric.loader import DataLoader
from dataset import NavGraphDataset
from GNNmodel import GNN

CSVpath="labels.csv"
imageDirectory="images/images"
device =torch.device("cuda" if torch.cuda.is_available()else "cpu")

'''
#configuring (helps figure out where to find CSV and images)
EPOCHS=20
batch_size=8
LR=1e-3
hiddenDimension=128
'''

#experiment configuration
experiment_configuration={    
    "epochs":20,
    "batchSize":8,
    "learningRate":1e-3,
    "hiddenDimension":128,
    "optimizer":"Adam",
    "scheduler":"cosine", #can also be None
    "trainSplit":0.8,}

EPOCHS=experiment_configuration["epochs"]
batch_size=experiment_configuration["batchSize"]
LR=experiment_configuration["learningRate"]
hiddenDimension=experiment_configuration["hiddenDimension"]
trainRatio=experiment_configuration["trainSplit"]

#training model (reads CSV, extracts CNN features for every image, builds graph images, produces a list of data objects)
def train():
    Dataset=NavGraphDataset(CSVpath, imageDirectory, device) #loads dataset

    #to split sample dataset
    #80% is for training, 20% is for testing 
    trainSize=int(trainRatio*len(Dataset))
    testSize=len(Dataset)-trainSize
    #batching graphs together, creates tensor mapping each node to its graph
    trainDataset, testDataset=torch.utils.data.random_split(Dataset, [trainSize, testSize])
    trainLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
    test_Loader=DataLoader(testDataset, batch_size=batch_size, shuffle=False)

    #initailising model
    inChannels=Dataset.graphs[0].x.shape[1]
    numClasses=len(Dataset.LabelMap)
    model=GNN(inChannels, hiddenDimension, numClasses).to(device)
    #optimizer
    if experiment_configuration["optimizer"]=="Adam":
        optimizer=torch.optim.Adam(model.parameters(), lr=LR)
    elif experiment_configuration["optimizer"]=="AdamW":
        optimizer=torch.optim.AdamW(
            model.paramaters(), lr=LR, weight_decay=1e-4)
   #cosine LR scheduler
    if experiment_configuration["scheduler"]=="cosine":
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS
        )
    else:
        scheduler=None
    criterion=torch.nn.NLLLoss() #bc model outputs via log softmax
    testAccuracy=[]
    #training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss=0
        for batch in trainLoader:
            #GNN outputs class log probabilities, computes loss, backpropogates, update model weights
            batch=batch.to(device)
            optimizer.zero_grad()
            out=model(batch.x, batch.edge_index, batch.batch)
            loss=criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss +=loss.item()
        avg_loss =total_loss/len(trainLoader)
        acc=evaluate (model, test_Loader)
        testAccuracy.append(acc)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}| Test Acc: {acc:.2f}%")
        if scheduler is not None:
            scheduler.step()

    meanTestAccuracy=sum(testAccuracy)/len(testAccuracy)
    print("Average Test Accuracy (over", EPOCHS, "epochs): ", meanTestAccuracy, "%")

    torch.save(model.state_dict(), "trainedGNN.pth")
    print("Training is complete. Model has been saved as trainedGNN.pth")

def evaluate(model, loader):
    #determining accuracy on the test set
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for batch in loader:
            batch=batch.to(device)
            out=model(batch.x, batch.edge_index, batch.batch)

            #picking class with highest probability, comparing with true labels, and calculating accuracy
            preds=out.argmax(dim=1)
            correct += (preds==batch.y).sum().item()
            total += batch.y.size(0)
    return 100.0 * correct /total if total >0 else 0
if __name__=="__main__":
    train()