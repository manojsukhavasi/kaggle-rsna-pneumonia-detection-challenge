from src.imports import *
from src.datasets import ChestXrayDataset
from src.model.densenet import DenseNet
from src.loss import CustomLoss
from src.utils.torch_utils import save_checkpoint, AverageMeter, Progbar, accuracy


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])


def train_one_epoch(data_loader, model, criterion, opt, epoch=1, interval=100):
    #batch_time = AverageMeter()
    #data_time = AverageMeter()
    losses = AverageMeter()
    clf_losses = AverageMeter()
    bb_losses = AverageMeter()
    accuracies = AverageMeter()
    
    model.train()
    #train_loss = 0.0
    no_of_batches = int(data_loader.dataset.num_samples/data_loader.batch_size) + 1

    #end = time.time()
    prog = Progbar(target = no_of_batches)

    for batch_idx, (inputs, label_targets, bb_targets) in enumerate(data_loader):

        #data_time.update(time.time()-end)

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

        #batch_time.update(time.time()-end)
        #end=time.time()

        #train_loss += loss.item()
        prog.update(batch_idx+1, exact=[("Loss", losses.avg), ('Accuracy', accuracies.avg), ('clf_loss', clf_losses.avg), ('bb_loss', bb_losses.avg)])

        # if(batch_idx%interval == 0):
        #     print(f'Train -> Batch : [{batch_idx}/{no_of_batches}]| Batch avg time :{batch_time.avg} \
        #     | Data_avg_time: {data_time.avg} | avg_loss: {train_loss/(batch_idx+1)}')

        # if(batch_idx%(500)==0):
        #     save_checkpoint({
        #         'epoch': epoch,
        #         'state_dict': model.state_dict(),
        #         'best_val_loss': train_loss/(batch_idx+1),
        #         'optimizer' : opt.state_dict()
        #     }, is_best=True, fname=f'checkpoint_{epoch}_{batch_idx}.pth.tar')


def train(epochs=5, resume=False, interval=100, ckpt_path=None):
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
    opt = optim.SGD(model.parameters(), lr = 1e-3, momentum=0.9, weight_decay=1e-4)
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
        train_one_epoch(train_loader, model, criterion, opt, epoch, interval=100)
        val_loss, val_acc, val_clf_loss, val_bb_loss = validate(val_loader, model, criterion, interval)
        print(f'Epoch:{epoch+1}/{epochs} | val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | val_clf_loss: {val_clf_loss:.4f} | val_bb_loss: {val_bb_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint({
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'best_val_loss': best_val_loss,
                'optimizer' : opt.state_dict()
            }, is_best=True)


def validate(data_loader, model, criterion, interval=1000):
    model.eval()

    #batch_time = AverageMeter()
    #data_time = AverageMeter()
    losses = AverageMeter()
    clf_losses = AverageMeter()
    bb_losses = AverageMeter()
    accuracies = AverageMeter()

    no_of_batches = int(data_loader.dataset.num_samples/data_loader.batch_size) + 1

    #end = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, label_targets, bb_targets) in enumerate(data_loader):
            #data_time.update(time.time()-end)

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


            #batch_time.update(time.time()-end)
            #end=time.time()

            # if(batch_idx%interval == 0):
            #     print(f'Train -> Batch : [{batch_idx}/{no_of_batches}]| Batch avg time :{batch_time.avg} \
            #     | Data_avg_time: {data_time.avg} | avg_loss: {val_loss/(batch_idx+1)}')
    
    # val_loss = losses.avg
    # print("_________________________________________________________________________________")
    # print(f'Val -> Final Loss:{val_loss} \t')
    # print("_________________________________________________________________________________")
    return losses.avg, accuracies.avg, clf_losses.avg, bb_losses.avg

if __name__=='__main__':
    train()
