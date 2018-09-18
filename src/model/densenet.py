from src.imports import *

class DenseNet(nn.Module):
    """ 
    Modified densenet model.
    """
    def __init__(self, nclasses=2, bbox_size=4):
        super().__init__()
        self.densenet = torchvision.models.densenet121(pretrained=True)
        num_features = self.densenet.classifier.in_features
        self.densenet = nn.Sequential(*list(self.densenet.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 1),
            nn.Sigmoid()
            )
        self.regressor = nn.Linear(num_features, bbox_size)
        
    def forward(self, x):
        features = self.densenet(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        labels = self.classifier(out)
        bboxes = self.regressor(out)
        return labels,bboxes



