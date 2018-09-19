from src.imports import *
from src.datasets import TestDataset
from src.model.densenet import DenseNet
from src.transforms import resize_to_original
from src.utils.utils import clean_bb_boxes
from evaluate import evaluate

def submit(model):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])
    test_dataset = TestDataset('data/test/', list_file='data/stage_1_sample_submission.csv', transform=transform)
    test_loader = data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    preds = evaluate(test_loader, model)
    preds = resize_to_original(preds.cpu())
    #preds = clean_bb_boxes(preds)
    preds_string = [' '.join(str(j) for j in i) if i[0]>0.5 else '' for i in preds.tolist()]
    test_df = pd.read_csv('data/stage_1_sample_submission.csv')
    test_df['PredictionString'] = preds_string
    test_df.to_csv('data/tmp/submissions/submissions.csv', index=False)

if __name__=='__main__':
    model = DenseNet(nclasses=2, bbox_size=4)
    model.cuda()
    checkpoint = torch.load('best_val_loss.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    submit(model)
