# System Summary
<!-- Commit ID: dac8fb0c3dbd1828d96f6b6179e1ec79a3b081f4 -->
## YAMNet Features

Our system uses the YAMNet model to extract audio features from the input audio. For each 0.96 seconds of audio, YAMNet extracts 1024 features. Considering the input audio is 30 seconds long we get ~30 frames of features.

## Baseline Model

The baseline model consists of a Bidirectional LSTM architecture. It is composed of 2 layers of 150 hidden units each with a dropout of 0.25. These layers are followed by a dropout layer of 0.25 and a dense layer with 2 units that outputs the probability of the input audio being a social interaction or not.

## Our Model

Our proposed DAEL approach consists of three main components: a shared feature extractor, multiple expert models, and a novel selection mechanism.

### Shared Feature Extractor

The shared feature extractor serves as the common front-end for all expert models, responsible for extracting relevant features from the input audio data. It consists of a stack of 1D convolutional layers with max-pooling operations, aimed at capturing local temporal patterns and reducing the dimensionality of the input representation. (Input shape: (N, 1024, 30), Output shape: (N, 256, 9))

### Multiple Expert Models

These expert models are trained on subsets of the data that exhibit similar characteristics, enabling them to develop domain-specific expertise. The expert models share a similar architecture, comprising 1D convolutional layers with max-pooling operations, followed by a batch normalization layer and a fully connected layer. The output of each expert model consists of two components: a prediction for the input sample (social interaction or not) and an embedding vector (the output of the last convolutional layer). (Input shape: (256, 9), Output shape: (N, 2) and (N, 64, 3))

### Selection Mechanism

In our model, input samples are dynamically routed to specialized expert models based on domain relevance, achieved through an innovative selection mechanism. This mechanism operates in two phases: training and inference, utilizing domain-specific embeddings and a signature matrix for expert model selection.

#### Training Phase: Constructing the Signature Matrix

1. Domain-Specific Assignment: Initially, a shared feature extractor processes each input sample. The output is then directed to an expert model trained on the corresponding domain of the input. For input i, with domain d, the output of the shared feature extractor is passed to the expert model trained on domain d. Thus, the prediction for input i is obtained from the expert model trained on domain d.
2. Signature Matrix Creation: Post each training epoch, we create a signature matrix that encapsulates domain and class-specific embeddings. This involves:
   - Segregating expert model outputs by class label (either 'social interaction' or 'non-social interaction').
   - For each combination of expert model and class label, we pinpoint the data point that minimizes the distance within the same domain while maximizing the distance across different domains. This selected data point acts as the domain-class centroid.
   - These centroids are concatenated to form the signature matrix, which represents a compact and comprehensive domain-class embedding space.

#### Inference Phase: Expert Model Selection

During inference, the signature matrix guides the decision-making process to determine if the input sample is a social interaction or not. This is achieved through the following steps:

1. Distance Calculation: We compute the cosine similarity between the embeddings of the expert models and the positive and negative centroids in the signature matrix.
2. Aggregation: We aggregate the similarity scores across all expert models separately for positive and negative scores.
3. Decision: These aggregated scores are then concatenated and passed through a softmax layer to obtain the final prediction.

### Training Procedure

During the training phase, we employ the triplet loss function along with Focal loss to train the shared feature extractor and expert models. The triplet loss aims to minimize the distance between embeddings of samples from the same domain and label while maximizing the distance between embeddings of samples from different domains and labels. The Focal loss is used to handle class imbalance and focus on hard examples. We use the Adam optimizer with a learning rate of 3e-4 and train the model for 100 epochs with a batch size of 32. We also use early stopping with a patience of 10 epochs to prevent overfitting.

## Critics

### The Shared Feature Extractor and Expert Models Architecture

- The shared feature extractor and expert models are composed of 1D convolutional layers with max-pooling operations. The initial input shape of the shared feature extractor is (N, 1024, 30), and the output shape is (N, 256, 9). The expert models have an input shape of (256, 9) and an output shape of (N, 2) and (N, 64, 3). The architecture is reducing the time dimension too much, which might lead to loss of temporal information.

### Signature Matrix Creation

- The process of creating a signature matrix based on domain-class centroids might be sensitive to outliers or noise within the training data. If the selected centroids are not truly representative of their respective domains and classes, the signature matrix may not effectively guide the selection mechanism during inference.

### The Selection Mechanism

- The selection mechanism is based on the cosine similarity between the embeddings of the expert models and the positive and negative centroids in the signature matrix. This mechanism might not be robust enough to handle complex decision boundaries and might not generalize well to unseen data.
- During the inference phase, the aggregation of similarity scores across all expert models separately for positive and negative scores might lead to a loss of information and might not effectively capture the domain-specific characteristics.

## Dr. Lee Recommendations

### Logging and Visualization

- Embedding Visualization: Use t-SNE to visualize the embeddings generated by the shared feature extractor and the expert models.
- Signature Matrix Visualization: Visualize the signature matrix to examine the centroids for each domain-class combination.
- Log Distance Metrics: During inference, log the cosine similarity scores between input embeddings and the centroids in the signature matrix.

## Sam Recommendations

### Model Architecture

- Instead of doing the signature matrix creation after the expert outputs, do it before (after the shared feature extractor).
