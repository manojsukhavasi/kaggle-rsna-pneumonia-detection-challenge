from src.imports import *

def resize(img, boxes, size):
    w,h = img.size
    tw,th = size,size
    sw = tw/w
    sh = th/h
    return img.resize((tw,th), Image.BILINEAR), boxes*torch.Tensor([sw,sh,sw,sh])
