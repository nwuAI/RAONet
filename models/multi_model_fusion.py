import torch
import torch.nn as nn


class FusionModel(nn.Module):
    def __init__(self, model1_cfp, model1_oct, model2_cfp, model2_oct, num_classes, embed_dim=3072):
        super(FusionModel, self).__init__()

        self.model1_cfp = model1_cfp
        self.model1_oct = model1_oct
        self.model2_cfp = model2_cfp
        self.model2_oct = model2_oct
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        for model in [self.model1_cfp, self.model1_oct, self.model2_cfp, self.model2_oct]:
            for param in model.parameters():
                param.requires_grad = False

        self.fusion_layer = nn.Linear(embed_dim * 1, num_classes)

    def forward(self, cfp_input, oct_input):

        features1_cfp = self.model1_cfp.forward_features(cfp_input)
        features2_cfp = self.model2_cfp(cfp_input)
        features1_oct = self.model1_oct.forward_features(oct_input)
        features2_oct = self.model2_oct(oct_input)

        cfp_combined_features = torch.cat((features1_cfp, features2_cfp), dim=1)
        oct_combined_features = torch.cat((features1_oct, features2_oct), dim=1)


        fused_features = torch.cat((cfp_combined_features, oct_combined_features), dim=1)

        output = self.fusion_layer(fused_features)
        return output
