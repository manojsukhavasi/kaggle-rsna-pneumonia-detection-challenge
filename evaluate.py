from src.imports import *

def evaluate(data_loader, model):

    model.eval()
    preds = torch.empty([0,0]).cuda()

    with torch.no_grad():
        for _, inputs in enumerate(data_loader):

            inputs = inputs.cuda()
            label_preds, bb_preds = model(inputs)
            preds = torch.cat([preds, torch.cat([label_preds, bb_preds], dim =1)], dim=0)

    return preds
    