from src.imports import *

 
def read_dicom(path):
    """
    Reads dicom and gives out RGB PIL Image
    """
    pd = pydicom.read_file(path)
    img_arr = pd.pixel_array
    img_arr = img_arr/img_arr.max()
    img_arr = (255*img_arr).clip(0,255).astype(np.uint8)
    img = Image.fromarray(img_arr).convert('RGB')
    return img

def clean_bb_boxes(preds):
    mask = preds[:,0]>0.5
    preds[~mask] = torch.Tensor([])
    return preds