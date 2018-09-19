from src.imports import *

def resize(img, boxes, size):
    w,h = img.size
    tw,th = size,size
    sw = tw/w
    sh = th/h
    return img.resize((tw,th), Image.BILINEAR), boxes*torch.Tensor([sw,sh,sw,sh])

def resize_to_original(preds, org_size=1024, curr_size=224):
    sw = org_size/curr_size
    sh = org_size/curr_size
    return preds*torch.Tensor([1,sw,sh,sw,sh])
