import torch
import torch.nn as nn


class FusionModel(nn.Module):
    def __init__(self, model1, model2, num_classes, embed_dim=3072):
        super(FusionModel, self).__init__()

        self.model1 = model1
        self.model2 = model2

        for param in self.model1.parameters():
            param.requires_grad = False
        for param in self.model2.parameters():
            param.requires_grad = False


        self.fusion_layer = nn.Linear(embed_dim, num_classes)

    def forward(self, inputs1):

        features1 = self.model1.forward_features(inputs1)
        features2 = self.model2(inputs1)



        combined_features = torch.cat((features1, features2), dim=1)

        output = self.fusion_layer(combined_features)
        return output