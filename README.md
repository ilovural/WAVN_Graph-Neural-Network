Changes from 1/2026 to 3 February 2026: 
1. Changed dataset to include larger dataset (12,000 samples) with RGB and Edge Segmented images (including CSV that has ~3000 samples for four directions).
2. Changed image dimensions to be converted to 256-dimensions.
3. Increased epoch size from 20, to 100, 180, 200....to 1000. Will continue to experimengt with epoch size, current 1000 epoch size gives much higher accuracy results (comparing 65% to ~84%).
4. The model now learns from one single graph, rather than many. This drastically increased the accuracy. 

Changes from 12/2025 to 2 January 2026: 
1. Changed the dataset to Natnael's.
2. Added feature to building_graph.py to make it so that CNN feature extractor can be changed.
3. Changed train.py to have configuration functon so that changes can be made (can change learning rate, epochs, and training split). 

Goal of Code:
1. Construct a graph using stationary robots, robots represent nodes and if there is a shared, common landmark, then the common landmarks are used to make the edges of the graph.
2. To make this effective, there needs to be multiple common landmarks between the images in order to understand how the nodes relate to one another.
3. From the common landmarks (panoramic images), the robots (nodes) can extract features to use for input from the CNN encoder and then use the CNN features to train the GNN model.

Challenges:
1. Model isn't learning correctly - use this https://debuggercafe.com/training-resnet18-from-scratch-using-pytorch/
2. Maybe change the learning rate and data splitting    

Steps:
1. Establish a graph in GNN.
2. Extract features from panoramic view to input to the CNN encoder.
3. CNN will give features to GNN & the GNN will determine which direction the node is which can be used to determine how the node can get to a target. 

DATASET being used: https://drive.google.com/drive/folders/1qSaI2HnsuliABdHZOqe9DAielChFlJaF?usp=share_link


Notes:
nodes         = robots or cameras
edges         = shared landmarks b/w stationary panoramas
node featutes = CNN encodings of panoramic images
labls/tasks   = predict direction toward destination
classification= pick which neighbour to move towards next
regression    = predict a continuous bearing/angle
Image Pairs: for each sample, using id_current.png and id_destination.png (node observation & target observation)
Constructing Graph: for stationary robots, need to compute feature matches b/w every pair of panoramic images. If pairs have more than or equal to M reliable matches, an edge is added. The weight of the edge is determined by the match count. The graph = node list + edge index + edge weights
Node Features: pass each *_current.png through CNN encoder to produce a fixed length vector/node
GNN: Graph that takes node features and the graph structure and predicts the target direction/class. The GNN outputs should be the classification and the regression. 
Training: Training samples where the graph is fixed, for each sample node provide feature vector and destination. Standard loss = crossEntropy for classification & L2 for regression. 
Evaluation: Measure the accuracy (classification) or the angular error (regression)

Files: 
1. building_graph.py (constructing the graph from images via keypoint matching-CNN feature extractor can be changed)
2. dataset.py (dataset loader & feature extraction using a CNN encoder)
3. GNNmodel.py (Graph Neural Network model)
4. train.py (ties files together and trains a simple clasification/regression Graph Neural Network)
