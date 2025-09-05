import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
from datetime import datetime


# valori di normalizzazione per ImageNet
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])



def load_img(image_path, target_width):
    
    # carichiamo l'immagine.
    image = Image.open(image_path)
    
    # riscala l'immagine (di default non lo facciamo)
    if target_width is not None:
        w, h = image.size
        aspect_ratio = h / w
        new_height = int(target_width * aspect_ratio)
        image = image.resize((target_width, new_height))
        
    # convertiamo image (oggetto PIL) in tensore (PyTorch)
    # in particolare, forma tensore: forma [C, H, W] normalizzato in [0, 1] float
    tensor = transforms.ToTensor()(image)

    # aggiungiamo la dimensione del batch (input modello)
    # risultato: tensore forma [1, C, H, W]
    tensor = tensor.unsqueeze(0) 

    return tensor



def resize_img(tensor, size):
    return F.interpolate(tensor, size=size, mode='bilinear')



def save_img(tensor):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_name = os.path.splitext(os.path.basename(args.input_image))[0]
    output_path = f"{input_name}_dream_{timestamp}.jpg"



    # convertiamo da tensore PyTorch a immagine PIL  
    to_pil = transforms.ToPILImage()
    image = to_pil(tensor.cpu())
    image.save(output_path)

def save_img(tensor, output_path=None):
    # rimuoviamo la dimensione del batch se presente
    if tensor.dim()==4 and tensor.size(0)==1:
        tensor = tensor.squeeze(0)
    # convertiamo da tensore PyTorch a immagine PIL  
    image = transforms.ToPILImage()(tensor)
    if output_path is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"dream_{ts}.jpg"
    image.save(output_path)
    return output_path



def preprocess_img(img):

    # ci assicuriamo di fare il broadcasting,
    # e nel device corretto.
    mean = IMAGENET_MEAN.view(1, 3, 1, 1).to(img.device)
    std  = IMAGENET_STD.view(1, 3, 1, 1).to(img.device)
    img = (img - mean) / std

    return img




def postprocess_img(img):
    
    mean = IMAGENET_MEAN.view(1, 3, 1, 1).to(tensor.device)
    std  = IMAGENET_STD.view(1, 3, 1, 1).to(tensor.device)
    

    out = (img * std) + mean
    
    
    return out