from src.imports import *
from src.transforms import resize
from src.utils.utils import read_dicom

class ChestXrayDataset(data.Dataset):

    def __init__(self, root, list_file, transform=None, train=False, input_size=1024):
        """
        Arguments:
            root: (str) images path
            list_file: (str) path to index file
            train: (boolean) train or test
            transform: ([transforms]) image transforms to be applied
            input_size: (int) model input image size 
        """

        self.root = root
        self.train = train
        self.transform = transform
        self.input_size = input_size

        self.fnames = []
        self.boxes = []
        self.labels = []

        with open(list_file) as f:
            lines = f.readlines()
            lines = lines[1:] #Removing the header
            self.num_samples = len(lines)

            for line in lines:
                splitted = line.split('\"')
                fn = splitted[0].split(',')[1] + '.dcm'
                bboxes = ast.literal_eval(splitted[1])
                self.fnames.append(fn)
                #num_boxes = len(bboxes)
                # For now considering one one box
                bb = bboxes[0]
                label, x, y, w, h = bb
                self.labels.append(torch.FloatTensor([int(label)]))
                self.boxes.append(torch.FloatTensor([float(x), float(y), float(w), float(h)]))

    def __getitem__(self, idx):

        fname = self.fnames[idx]
        img = read_dicom(self.root+ fname)
        boxes = self.boxes[idx].clone()
        label = self.labels[idx]
        img, boxes = resize(img, boxes, self.input_size)
        if self.transform is not None:
            img = self.transform(img)

        return img ,label, boxes


    def __len__(self):
        return self.num_samples


    # def collate_fn(self, batch):
    #     ''' Pad images and encode targets 
        
    #     Args:
    #         batch, (list) of images, cls_targets, loc_targets
    #     Returns:
    #         padded_images, stacked loc_targets, stacked_cls_targets
    #     '''
    #     imgs   = [x[0] for x in batch]
    #     labels = [x[1] for x in batch]
    #     boxes  = [x[2] for x in batch] 

    #     h = w = self.input_size
    #     num_imgs = len(imgs)
    #     inputs = torch.zeros(num_imgs, 3, h, w)
    #     targets = torch.zeros(len(labels))

    #     loc_targets = []
    #     for i in range(num_imgs):
    #         inputs[i] = imgs[i]
    #         targets[i] = labels[i]
    #         loc_targets.append(boxes[i])
    #     return inputs, labels, torch.stack(loc_targets)

    

