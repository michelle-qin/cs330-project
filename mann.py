import torch
from torch import nn, Tensor
import torch.nn.functional as F

def initialize_weights(model):
    if type(model) in [nn.Linear]:
        nn.init.xavier_uniform_(model.weight)
        nn.init.zeros_(model.bias)
    elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
        print(model.weight_hh_l0.size())
        nn.init.orthogonal_(model.weight_hh_l0)
        nn.init.xavier_uniform_(model.weight_ih_l0)
        nn.init.zeros_(model.bias_hh_l0)
        nn.init.zeros_(model.bias_ih_l0)

four_layers = False
one_layer = False

class MANN(nn.Module):
    def __init__(self, num_classes, samples_per_class, hidden_dim):
        super(MANN, self).__init__()
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class

        if four_layers:
            self.layer1 = torch.nn.LSTM(num_classes + 196608, hidden_dim * 3, batch_first=True)
            self.layer2 = torch.nn.LSTM(hidden_dim * 3, hidden_dim * 2, batch_first=True)
            self.layer3 = torch.nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True)
            self.layer4 = torch.nn.LSTM(hidden_dim, num_classes, batch_first=True)
            initialize_weights(self.layer1)
            initialize_weights(self.layer2)
            initialize_weights(self.layer3)
            initialize_weights(self.layer4)
        elif one_layer:
            self.layer1 = torch.nn.LSTM(num_classes + 196608, num_classes, batch_first=True)
            initialize_weights(self.layer1)
        else:
            self.layer1 = torch.nn.LSTM(num_classes + 196608, hidden_dim, batch_first=True)
            self.layer2 = torch.nn.LSTM(hidden_dim, num_classes, batch_first=True)
            print("initializing layer 1")
            initialize_weights(self.layer1)
            print("initializing layer 2")
            initialize_weights(self.layer2)

            print("DONE INITIALIZING")

    def forward(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: [B, K+1, N, 784] flattened images
            labels: [B, K+1, N, N] ground truth labels
        Returns:
            [B, K+1, N, N] predictions
        """
        #############################
        ### START CODE HERE ###
        B = input_images.shape[0]
        N = input_images.shape[2]
        K = input_images.shape[1] - 1

        # Step 1: Concatenate the full (support & query) set of labels and images
        images_and_labels = torch.cat((input_images, input_labels), dim=3)

        # Step 2: Zero out the labels from the concatenated corresponding to the query set
        images_and_labels[:, -1, :, 196608:] = torch.zeros_like(images_and_labels[:, -1, :, 196608:])
        # [B, K+1, N, 784 + N]

        # Step 3: Pass the concatenated set sequentially to the memory-augmented network
        input = torch.reshape(images_and_labels, (B, N*(K+1), 196608+N))
        input = input.type(torch.float32)
        if four_layers:
            output, (h_1, c_1) = self.layer1(input)
            output, (h_2, c_2) = self.layer2(output)
            output, (h_3, c_3) = self.layer3(output)
            output, (h_4, c_4) = self.layer4(output)
        elif one_layer:
            output, (h_1, c_1) = self.layer1(input)
        else:
            output, (h_1, c_1) = self.layer1(input)
            output, (h_2, c_2) = self.layer2(output)

        # Step 3: Return the predictions with [B, K+1, N, N] shape
        return torch.reshape(output, (B, K+1, N, N))


        ### END CODE HERE ###

    def loss_function(self, preds, labels):
        """
        Computes MANN loss
        Args:
            preds: [B, K+1, N, N] network output
            labels: [B, K+1, N, N] labels
        Returns:
            scalar loss
        Note:
            Loss should only be calculated on the N test images
            Loss should be a scalar since mean reduction is used for cross entropy loss
            You would want to use F.cross_entropy here, specifically:
            with predicted unnormalized logits as input and ground truth class indices as target.
            Your logits would be of shape [B*N, N], and label indices would be of shape [B*N].
        """
        #############################

        loss = None

        ### START CODE HERE ###

        B = preds.shape[0]
        N = preds.shape[2]

        # Step 1: extract the predictions for the query set
        query_preds = preds[:, -1, :, :]
        query_preds = query_preds.reshape(B*N, N)

        # Step 2: extract the true labels for the query set and reverse the one hot-encoding  
        query_labels = labels[:, -1, :, :]
        query_labels = torch.argmax(query_labels, dim=2)
        query_labels = query_labels.reshape(B*N)

        # Step 3: compute the Cross Entropy Loss for the query set only!
        loss = F.cross_entropy(query_preds, query_labels)

        ### END CODE HERE ###
        return loss
