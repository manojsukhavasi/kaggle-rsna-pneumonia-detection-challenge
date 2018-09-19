from src.imports import *
from src.datasets import ChestXrayDataset
from src.model.densenet import DenseNet
from src.loss import CustomLoss
from src.utils.torch_utils import save_checkpoint, AverageMeter, Progbar, accuracy


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])

def train_one_epoch(data_loader, model, criterion, opt):

    losses = AverageMeter()
    clf_losses = AverageMeter()
    bb_losses = AverageMeter()
    accuracies = AverageMeter()
    
    model.train()
    no_of_batches = int(data_loader.dataset.num_samples/data_loader.batch_size) + 1

    prog = Progbar(target = no_of_batches)

    for batch_idx, (inputs, label_targets, bb_targets) in enumerate(data_loader):

        inputs = inputs.cuda()
        label_targets = label_targets.cuda()
        bb_targets = bb_targets.cuda()

        opt.zero_grad()

        label_preds, bb_preds = model(inputs)
        loss, clf_loss, bb_loss = criterion(label_preds, label_targets, bb_preds, bb_targets)

        losses.update(loss.item(), inputs.size(0))
        clf_losses.update(clf_loss, inputs.size(0))
        bb_losses.update(bb_loss, inputs.size(0))
        acc = accuracy(label_preds, label_targets)
        accuracies.update(acc, inputs.size(0))

        loss.backward()
        opt.step()

        prog.update(batch_idx+1, exact=[("Loss", losses.avg), ('Accuracy', accuracies.avg), ('clf_loss', clf_losses.avg), ('bb_loss', bb_losses.avg)])


def train(epochs=100, resume=False, ckpt_path=None):
    print('Loading training dataset .....')
    train_dataset = ChestXrayDataset('data/train/', 'data/tmp/train_labels.csv', transform=transform ,train=True, input_size=224)
    train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_dataset = ChestXrayDataset('data/train/', 'data/tmp/val_labels.csv', transform=transform ,train=False, input_size=224)
    val_loader = data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    print('Loading complete.')

    model = DenseNet(nclasses=2, bbox_size=4)

    criterion = CustomLoss()

    model.cuda()
    criterion.cuda()
    opt = optim.SGD(model.parameters(), lr = 1e-4, momentum=0.9, weight_decay=1e-4)
    best_val_loss = 1000

    if resume:
        if(os.path.isfile(ckpt_path)):
            print(f'Loading from the checkpoint {ckpt_path}..')
            checkpoint = torch.load(ckpt_path)
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss']
            model.load_state_dict(checkpoint['state_dict'])
            opt.load_state_dict(checkpoint['optimizer'])
            print(f'Loaded checkpoint.')
            print(f'Continuing from the epoch: {start_epoch}')
        else:
            print(f'Error: No checkpoint is found at the path : {ckpt_path}')


    for epoch in range(epochs):
        train_one_epoch(train_loader, model, criterion, opt)
        val_loss, val_acc, val_clf_loss, val_bb_loss = validate(val_loader, model, criterion)
        print(f'Epoch:{epoch+1}/{epochs} | val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | val_clf_loss: {val_clf_loss:.4f} | val_bb_loss: {val_bb_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint({
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'best_val_loss': best_val_loss,
                'optimizer' : opt.state_dict()
            }, is_best=True)


def validate(data_loader, model, criterion):
    model.eval()

    losses = AverageMeter()
    clf_losses = AverageMeter()
    bb_losses = AverageMeter()
    accuracies = AverageMeter()

    #no_of_batches = int(data_loader.dataset.num_samples/data_loader.batch_size) + 1

    with torch.no_grad():
        for _, (inputs, label_targets, bb_targets) in enumerate(data_loader):

            inputs = inputs.cuda()
            label_targets = label_targets.cuda()
            bb_targets = bb_targets.cuda()

            label_preds, bb_preds = model(inputs)
            loss, clf_loss, bb_loss = criterion(label_preds, label_targets, bb_preds, bb_targets)

            losses.update(loss.item(), inputs.size(0))
            clf_losses.update(clf_loss, inputs.size(0))
            bb_losses.update(bb_loss, inputs.size(0))
            acc = accuracy(label_preds, label_targets)
            accuracies.update(acc, inputs.size(0))

    return losses.avg, accuracies.avg, clf_losses.avg, bb_losses.avg

if __name__=='__main__':
    train()
