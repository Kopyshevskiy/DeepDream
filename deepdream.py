import torch
import torch.nn.functional as F


from image_utils import preprocess_img, postprocess_img, resize_img 



def get_activations(model, input_tensor, layer_names):
    

    activations = {}
    hooks = []

    def hook_fn(module, input, output):
        # Il nome del layer è l'attributo a cui il modulo è assegnato nel modello.
        # Questo è un modo per ottenere il nome del layer dinamicamente.
        for name, layer in model.named_modules():
            if layer is module:
                activations[name] = output
                return

    # Registra un hook per ogni layer di interesse
    for name, layer in model.named_modules():
        if name in layer_names:
            hooks.append(layer.register_forward_hook(hook_fn))
    
    # Esegui il forward pass per triggerare gli hooks
    model(input_tensor)

    # Rimuovi gli hooks
    for hook in hooks:
        hook.remove()
        
    return [activations[name] for name in layer_names]





def gradient_ascent(model, img, args):

    iterations = args.iterations
    layers = args.layers
    lr = args.learning_rate

    # non teniamo traccia della storia delle operazioni sul tensore img.
    img = img.detach()             

    # questa è l'istruzione più importante.
    # i gradienti vengono calcolati rispetto all'immagine.
    img = img.requires_grad_(True)  
    

    for i in range(iterations):

        # di default, PyTorch accumula i gradienti.
        # di conseguenza, dobbiamo azzerare il gradiente accumulato,
        # in maniera da considerare (solo) il gradiente sull'immagine attuale
        # che cambia ad ogni iterazione.
        img.grad = None
        

        # facciamo forward model, e otteniamo le feature maps che vogliamo massimizzare.
        activations = get_activations(model, img, layers)
        # calcoliamo la loss finale come somma delle norme L2 delle feature maps.
        loss = torch.stack([a.norm() for a in activations]).mean()
        # calcoliamo i gradienti della loss rispetto all'immagine
        loss.backward()

        
        # otteniamo i gradienti e normalizziamo (stabilità).
        grad = img.grad
        grad = (grad - torch.mean(grad)) / (torch.std(grad) + 1e-8)

        # disabilitiamo il calcolo dei gradienti per le operazioni che seguono.
        with torch.no_grad():

            # passo di gradient ascent.
            # nota: img += lr * grad avrebbe creato un nuovo tensore 
            # con requires_grad=False (e bisognerebbe riabilitarlo, il che sarebbe ridondante).
            img.add_(lr * grad)

            # clampiamo i valori dei pixel.
            img.clamp_(-3.0, 3.0)
    

    return img






def deep_dream(model, img, args):

    pyramid_levels = args.pyramid_levels
    pyramid_ratio  = args.pyramid_ratio

    # pre-processing: lo facciamo subito.
    # necessario siccome usiamo modelli pre-trainati su ImageNet.
    # nota: volendo, si potrebbe il pre-processing per ogni step del gradient ascent,
    # in quanto non abbiamo garanzia che dopo uno step del gradient ascent mean sia zero e std 1.
    # tuttavia, in pratica, non sembra essere un problema.
    img = preprocess_img(img)
    

    # prendiamo la terza e quarta dimensione del tensore
    # rispettivamente: height e width
    h, w = img.shape[2:]
    

    for lv in range(pyramid_levels):
       
        # calcola le nuove dimensioni per il livello corrente della piramide.
        # si parte da una versione ridotta dell'immagine,
        # e si arriva alla dimensione originale
        new_h = int(h * (pyramid_ratio ** (lv - pyramid_levels + 1)))
        new_w = int(w * (pyramid_ratio ** (lv - pyramid_levels + 1)))        
        img = resize_img(img, (new_h, new_w))

        # jitter?
        # todo: applicare translation shift
        # todo: applicare circolar shift
        

        # applica il gradient ascent
        img = gradient_ascent(model, img, args)
        

        # riportiamo immagine dimensioni iniziali
        # img = resize_img(img, (h, w))

        # ri-jitter?
        # todo: ri-applicare translation shift
        # todo: ri-applicare circolar shift


    
    ## Assicurati che i valori dei pixel siano nel range [0, 1]
    ## non va bene: il clamp lo si fa a fine di ogni iterazione di gradient ascent
    # out = torch.clamp(img, 0, 1)

    # post-processing
    out = postprocess_img(img)


    return out