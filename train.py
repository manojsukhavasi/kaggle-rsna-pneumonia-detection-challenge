from src.imports import *
from src.datasets import ChestXrayDataset
from src.model.densenet import DenseNet
from src.loss import CustomLoss


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])


def train_one_epoch(data_loader, model, criterion, opt, epoch=1, interval=1000):
    
    model.train()
    train_loss = 0.0

    for _, (inputs, label_targets, bb_targets) in enumerate(data_loader):

        inputs = inputs.cuda()
        label_targets = label_targets.cuda()
        bb_targets = bb_targets.cuda()

        opt.zero_grad()
        label_preds, bb_preds = model(inputs)
        loss = criterion(label_preds, label_targets, bb_preds, bb_targets)
        loss.backward()
        opt.step()

        train_loss += loss.item()
        print(f'Loss is {loss.item()}')


def train(epochs=1):
    print('Loading training dataset .....')
    train_dataset = ChestXrayDataset('data/train/', 'data/tmp/train_labels_cleaned.csv', transform=transform ,train=True, input_size=224)
    train_loader = data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    print('Loading complete.')

    model = DenseNet(nclasses=2, bbox_size=4)

    criterion = CustomLoss()

    model.cuda()
    criterion.cuda()
    opt = optim.SGD(model.parameters(), lr = 1e-3, momentum=0.9, weight_decay=1e-4)

    for epoch in range(epochs):
        train_one_epoch(train_loader, model, criterion, opt, epoch, interval=1000)


if __name__=='__main__':
    train()
