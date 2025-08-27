import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import copy
import time
import json
import numpy as np
import torch
import datasets
import models2_mutliclip
import argparse
from tqdm import tqdm
import math
from scipy import stats
from losses_multiclip_pos7s_cub import compute_batch_loss
import datetime
from instrumentation import train_logger
import warnings
import torchvision.transforms as transforms
warnings.filterwarnings("ignore")
from sig_t_module_multiclip3_7s import sig_t
from transformers import CLIPVisionModelWithProjection
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms as transforms
from transformers import AutoProcessor, AutoModel

# Initialize CLIP model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14-336", output_attentions=True).to(device)
import open_clip
model, preprocess = open_clip.create_model_from_pretrained('hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K')
tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K')
from transformers import SiglipModel, SiglipProcessor

siglip_model = SiglipModel.from_pretrained("google/siglip-so400m-patch14-384").to(device)
siglip_processor = SiglipProcessor.from_pretrained("google/siglip-so400m-patch14-384")
#siglip_model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to(device)
#siglip_processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
modelH, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
#modelH = modelH.to(Z['device'])
#model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizerH = open_clip.get_tokenizer('ViT-H-14')
modelG, _, preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s34b_b88k')
#modelG = modelG.to(Z['device'])
#model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizerG = open_clip.get_tokenizer('ViT-g-14')
# Placeholder for results
features_list = []
indices_list = []


def run_train_phase(model, P, Z, logger, epoch, phase):
    model.train()
    #siglip_model.train()
    model2,_,preprocess = open_clip.create_model_and_transforms('convnext_large_d_320', pretrained='laion2b_s29b_b131k_ft_soup')
    model2 = model2.to(Z['device'])
    tokenizer = open_clip.get_tokenizer('convnext_large_d_320')
    modelH, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
    #modelH = modelH.to(Z['device'])
    #model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    tokenizerH = open_clip.get_tokenizer('ViT-H-14')
    modelG, _, preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s34b_b88k')
    #modelG = modelG.to(Z['device'])
    #model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    tokenizerG = open_clip.get_tokenizer('ViT-g-14')
    modelBG, _, preprocess = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k')
    modelBG = modelBG.to(Z['device'])
    tokenizerBG = open_clip.get_tokenizer('ViT-bigG-14')
    modelC, _, preprocess = open_clip.create_model_and_transforms('convnext_xxlarge', pretrained='laion2b_s34b_b82k_augreg_soup')
    #model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    tokenizerC = open_clip.get_tokenizer('convnext_xxlarge')
    modelC = modelC.to(Z['device'])
    modelH = modelH.to(Z['device'])
    modelG = modelG.to(Z['device'])
        

    #tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K')
    desc = '[{}/{}]{}'.format(epoch, P['num_epochs'], phase.rjust(8, ' '))
    voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                 'sheep', 'sofa', 'train', 'tvmonitor']
    for batch in tqdm(Z['dataloaders'][phase], desc=desc, mininterval=300):
        # Move data to device
        # move data to GPU:
        batch['image'] = batch['image'].to(Z['device'], non_blocking=True)
        
        desired_size = (336, 336)
        resize_transform = transforms.Resize(desired_size)
        resized_batch=[]
        resized_batch = torch.stack([resize_transform(img) for img in batch['image']]).to(Z['device'], non_blocking=True)
        resized_batch = resized_batch.float()
        if resized_batch.dim() == 3:  # If single image
            resized_batch = resized_batch.unsqueeze(0)
        #print(resized_batch.shape)
        #resized_batch.to(Z['device'], non_blocking=True)
        batch['labels_np'] = batch['label_vec_obs'].clone().numpy()  # copy of labels for use in metrics
        batch['label_vec_obs'] = batch['label_vec_obs'].to(Z['device'], non_blocking=True)
        clip_features = clip_model(resized_batch).image_embeds
        image_embedding = F.normalize(clip_features, dim=-1)
        y_clip=P['txt_features']
        similarity_clip = image_embedding @ y_clip.T
        similarity_clip=F.softmax(similarity_clip/ 0.01, dim=-1)
        
        # OPENCLIP
        batch['image'] = batch['image'].to(Z['device'], non_blocking=True)
        desired_size = (224, 224)
        resize_transform = transforms.Resize(desired_size)
        resized_batch=[]
        resized_batch = torch.stack([resize_transform(img.to(Z['device'])) for img in batch['image']])
        #print(resized_batch.shape)
        #resized_batch.to(Z['device'], non_blocking=True)
        resized_batch = resized_batch.half()  # Converts to float16
        #print(resized_batch.shape)
        #batch['labels_np'] = batch['label_vec_obs'].clone().numpy()  # copy of labels for use in metrics
        #batch['label_vec_obs'] = batch['label_vec_obs'].to(Z['device'], non_blocking=True)

        text = tokenizer(voc_classes)    
        text = text.to(Z['device'])
        # Use SigLIP processor for resizing and tokenization

        

        # Extract features using SigLIP
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model2.encode_image(resized_batch)
            text_features = model2.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = image_features @ text_features.T
            text_probs =F.softmax(text_probs / 0.01, dim=-1)
       
        batch['image'] = batch['image'].to(Z['device'], non_blocking=True)
        image_list = [transforms.ToPILImage()(img) for img in batch['image']]
        desired_size = (336, 336)
        resize_transform = transforms.Resize(desired_size)
        resized_batch=[]
        resized_batch = torch.stack([resize_transform(img) for img in batch['image']]).to(Z['device'], non_blocking=True)
        resized_batch = resized_batch.float()
        if resized_batch.dim() == 3:  # If single image
            resized_batch = resized_batch.unsqueeze(0)
        #batch['labels_np'] = batch['label_vec_obs'].clone().numpy()  # copy of labels for use in metrics
        #batch['label_vec_obs'] = batch['label_vec_obs'].to(Z['device'], non_blocking=True)
        # Ensure text_features is a list of strings
        if isinstance(P['txt_features'], np.ndarray):
            text_features = [f"Feature {i}" for i in range(len(P['txt_features']))]
        elif isinstance(P['txt_features'], list):
            if all(isinstance(item, list) for item in P['txt_features']):
                # If list of lists, convert each list to a string
                text_features = [" ".join(map(str, row)) for row in P['txt_features']]
            else:
                # If list of strings or other types, ensure they are strings
                text_features = [str(item) for item in P['txt_features']]
        else:
            # For any other type, convert to a single-item list of strings
            text_features = [str(P['txt_features'])]
        # Use SigLIP processor for resizing and tokenization
        inputs = siglip_processor(
            text=voc_classes,
            images=image_list,
            padding="max_length",
            return_tensors="pt"
        ).to(Z['device'])

        # Extract features using SigLIP
        with torch.no_grad():
            outputs = siglip_model(**inputs)
            image_embeds = outputs.image_embeds  # Extract image embeddings
            text_embeds = outputs.text_embeds
            similarity_siglip = image_embeds @ text_embeds.T
            similarity_siglip=F.softmax(similarity_siglip/ 0.01, dim=-1)  
        batch['image'] = batch['image'].to(Z['device'], non_blocking=True)
        desired_size = (224, 224)
        resize_transform = transforms.Resize(desired_size)
        resized_batch=[]
        resized_batch = torch.stack([resize_transform(img.to(Z['device'])) for img in batch['image']])
        #print(resized_batch.shape)
        #resized_batch.to(Z['device'], non_blocking=True)
        resized_batch = resized_batch.half()  # Converts to float16
        #print(resized_batch.shape)

        text = tokenizerH(voc_classes)    
        text = text.to(Z['device'])
        # Use SigLIP processor for resizing and tokenization
        
        # Extract features using SigLIP
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = modelH.encode_image(resized_batch)
            text_features = modelH.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = image_features @ text_features.T
            text_probs_VITH =F.softmax(text_probs / 0.01, dim=-1)
            
        text = tokenizerG(voc_classes)    
        text = text.to(Z['device'])
        # Use SigLIP processor for resizing and tokenization
        # Extract features using SigLIP
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = modelG.encode_image(resized_batch)
            text_features = modelG.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = image_features @ text_features.T
            text_probs_VITG =F.softmax(text_probs / 0.01, dim=-1)
        text = tokenizerBG(voc_classes)    
        text = text.to(Z['device'])
        # Use SigLIP processor for resizing and tokenization
        # Extract features using SigLIP
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = modelBG.encode_image(resized_batch)
            text_features = modelBG.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = image_features @ text_features.T
            text_probs_VITBG =F.softmax(text_probs / 0.01, dim=-1)    
        text = tokenizerBG(voc_classes)    
        text = text.to(Z['device'])
        # Use SigLIP processor for resizing and tokenization
        # Extract features using SigLIP
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = modelBG.encode_image(resized_batch)
            text_features = modelBG.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = image_features @ text_features.T
            text_probs_VITBG =F.softmax(text_probs / 0.01, dim=-1)  
        text = tokenizerC(voc_classes)    
        text = text.to(Z['device'])
        # Use SigLIP processor for resizing and tokenization
        # Extract features using SigLIP
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = modelC.encode_image(resized_batch)
            text_features = modelC.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = image_features @ text_features.T
            text_probs_VITC =F.softmax(text_probs / 0.01, dim=-1)          
        concatenated_similarity = torch.cat([similarity_clip, text_probs,similarity_siglip,text_probs_VITH,text_probs_VITG,text_probs_VITBG,text_probs_VITC], dim=0)
        Z['optimizer'].zero_grad()
        with torch.set_grad_enabled(True):
            #print('Text_FeaturesP[txt_features]',P['txt_features'])
            # Get logits and predictions
            batch['logits'], batch['logits_pl'], batch['similarity'] = model.f(batch['image'], concatenated_similarity, text)
            batch['preds2'] = torch.sigmoid(batch['logits'])
            
            # Apply transformation with matrix A from sig_t (trans)
            #trans = sig_t(Z['device'], num_classes=20, init=5.75)
            A = Z['trans']()  # Get the transformation matrix
            batch['preds'] = torch.matmul(A, batch['preds2'].T).T
            batch['preds_np'] = batch['preds'].clone().detach().cpu().numpy()
            

            batch = compute_batch_loss(batch, P, Z)  # Assuming this uses `batch['preds']`
            batch['loss_tensor'].backward()
            Z['optimizer'].step()
            #print('A',A)
            
            # Log the A matrix for analysis (optional)
                
        # Save current batch data:
        logger.update_phase_data(batch)

def run_eval_phase(model, P, Z, logger, epoch, phase):

    '''
    Run one evaluation phase.

    Parameters
    model: Model to train.
    P: Dictionary of parameters, which completely specify the training procedure.
    Z: Dictionary of temporary objects used during training.
    logger: Object used to track various metrics during training.
    epoch: Integer index of the current epoch.
    phase: String giving the phase name
    '''
    
    assert phase in ['val', 'test']
    model.eval()
    #siglip_model.eval()
    #siglip_model.train()
    model2,_, preprocess = open_clip.create_model_and_transforms('convnext_large_d_320', pretrained='laion2b_s29b_b131k_ft_soup')
    model2 = model2.to(Z['device'])
    tokenizer = open_clip.get_tokenizer('convnext_large_d_320')
    modelH, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
    #modelH = modelH.to(Z['device'])
    #model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    tokenizerH = open_clip.get_tokenizer('ViT-H-14')
    modelG, _, preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s34b_b88k')
    #modelG = modelG.to(Z['device'])
    #model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    tokenizerG = open_clip.get_tokenizer('ViT-g-14')
    modelBG, _, preprocess = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k')
    modelBG = modelBG.to(Z['device'])
    tokenizerBG = open_clip.get_tokenizer('ViT-bigG-14')
    modelC, _, preprocess = open_clip.create_model_and_transforms('convnext_xxlarge', pretrained='laion2b_s34b_b82k_augreg_soup')
    #model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    tokenizerC = open_clip.get_tokenizer('convnext_xxlarge')
    modelC = modelC.to(Z['device'])
    modelH = modelH.to(Z['device'])
    modelG = modelG.to(Z['device'])
        
    #modelH.eval()
    #modelG.eval()
    desc = '[{}/{}]{}'.format(epoch, P['num_epochs'], phase.rjust(8, ' '))
    voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
             'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
             'sheep', 'sofa', 'train', 'tvmonitor']
    for batch in tqdm(Z['dataloaders'][phase], desc=desc, mininterval=300):
        # Move data to device
        # move data to GPU:
        batch['image'] = batch['image'].to(Z['device'], non_blocking=True)
        
        desired_size = (336, 336)
        resize_transform = transforms.Resize(desired_size)
        resized_batch=[]
        resized_batch = torch.stack([resize_transform(img) for img in batch['image']]).to(Z['device'], non_blocking=True)
        resized_batch = resized_batch.float()
        if resized_batch.dim() == 3:  # If single image
            resized_batch = resized_batch.unsqueeze(0)
        #print(resized_batch.shape)
        #resized_batch.to(Z['device'], non_blocking=True)
        batch['labels_np'] = batch['label_vec_obs'].clone().numpy()  # copy of labels for use in metrics
        batch['label_vec_obs'] = batch['label_vec_obs'].to(Z['device'], non_blocking=True)
        clip_features = clip_model(resized_batch).image_embeds
        image_embedding = F.normalize(clip_features, dim=-1)
        y_clip=P['txt_features']
        similarity_clip = image_embedding @ y_clip.T
        similarity_clip=F.softmax(similarity_clip/ 0.01, dim=-1)
        
        # OPENCLIP
        batch['image'] = batch['image'].to(Z['device'], non_blocking=True)
        desired_size = (224, 224)
        resize_transform = transforms.Resize(desired_size)
        resized_batch=[]
        resized_batch = torch.stack([resize_transform(img.to(Z['device'])) for img in batch['image']])
        #print(resized_batch.shape)
        #resized_batch.to(Z['device'], non_blocking=True)
        resized_batch = resized_batch.half()  # Converts to float16
        #print(resized_batch.shape)
        #batch['labels_np'] = batch['label_vec_obs'].clone().numpy()  # copy of labels for use in metrics
        #batch['label_vec_obs'] = batch['label_vec_obs'].to(Z['device'], non_blocking=True)

        text = tokenizer(voc_classes)    
        text = text.to(Z['device'])
        # Use SigLIP processor for resizing and tokenization

        

        # Extract features using SigLIP
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model2.encode_image(resized_batch)
            text_features = model2.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = image_features @ text_features.T
            text_probs =F.softmax(text_probs / 0.01, dim=-1)
       
        batch['image'] = batch['image'].to(Z['device'], non_blocking=True)
        image_list = [transforms.ToPILImage()(img) for img in batch['image']]
        desired_size = (336, 336)
        resize_transform = transforms.Resize(desired_size)
        resized_batch=[]
        resized_batch = torch.stack([resize_transform(img) for img in batch['image']]).to(Z['device'], non_blocking=True)
        resized_batch = resized_batch.float()
        if resized_batch.dim() == 3:  # If single image
            resized_batch = resized_batch.unsqueeze(0)
        #batch['labels_np'] = batch['label_vec_obs'].clone().numpy()  # copy of labels for use in metrics
        #batch['label_vec_obs'] = batch['label_vec_obs'].to(Z['device'], non_blocking=True)
        # Ensure text_features is a list of strings
        if isinstance(P['txt_features'], np.ndarray):
            text_features = [f"Feature {i}" for i in range(len(P['txt_features']))]
        elif isinstance(P['txt_features'], list):
            if all(isinstance(item, list) for item in P['txt_features']):
                # If list of lists, convert each list to a string
                text_features = [" ".join(map(str, row)) for row in P['txt_features']]
            else:
                # If list of strings or other types, ensure they are strings
                text_features = [str(item) for item in P['txt_features']]
        else:
            # For any other type, convert to a single-item list of strings
            text_features = [str(P['txt_features'])]
        # Use SigLIP processor for resizing and tokenization
        inputs = siglip_processor(
            text=voc_classes,
            images=image_list,
            padding="max_length",
            return_tensors="pt"
        ).to(Z['device'])

        # Extract features using SigLIP
        with torch.no_grad():
            outputs = siglip_model(**inputs)
            image_embeds = outputs.image_embeds  # Extract image embeddings
            text_embeds = outputs.text_embeds
            similarity_siglip = image_embeds @ text_embeds.T
            similarity_siglip=F.softmax(similarity_siglip/ 0.01, dim=-1)  
        batch['image'] = batch['image'].to(Z['device'], non_blocking=True)
        desired_size = (224, 224)
        resize_transform = transforms.Resize(desired_size)
        resized_batch=[]
        resized_batch = torch.stack([resize_transform(img.to(Z['device'])) for img in batch['image']])
        #print(resized_batch.shape)
        #resized_batch.to(Z['device'], non_blocking=True)
        resized_batch = resized_batch.half()  # Converts to float16
        #print(resized_batch.shape)

        text = tokenizerH(voc_classes)    
        text = text.to(Z['device'])
        # Use SigLIP processor for resizing and tokenization
        
        # Extract features using SigLIP
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = modelH.encode_image(resized_batch)
            text_features = modelH.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = image_features @ text_features.T
            text_probs_VITH =F.softmax(text_probs / 0.01, dim=-1)
            
        text = tokenizerG(voc_classes)    
        text = text.to(Z['device'])
        # Use SigLIP processor for resizing and tokenization
        # Extract features using SigLIP
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = modelG.encode_image(resized_batch)
            text_features = modelG.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = image_features @ text_features.T
            text_probs_VITG =F.softmax(text_probs / 0.01, dim=-1)
        text = tokenizerC(voc_classes)    
        text = text.to(Z['device'])
        # Use SigLIP processor for resizing and tokenization
        # Extract features using SigLIP
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = modelC.encode_image(resized_batch)
            text_features = modelC.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = image_features @ text_features.T
            text_probs_VITC =F.softmax(text_probs / 0.01, dim=-1)      
        text = tokenizerBG(voc_classes)    
        text = text.to(Z['device'])
        # Use SigLIP processor for resizing and tokenization
        # Extract features using SigLIP
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = modelBG.encode_image(resized_batch)
            text_features = modelBG.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = image_features @ text_features.T
            text_probs_VITBG =F.softmax(text_probs / 0.01, dim=-1)         
        concatenated_similarity = torch.cat([similarity_clip, text_probs,similarity_siglip,text_probs_VITH,text_probs_VITG,text_probs_VITBG,text_probs_VITC], dim=0)
        # forward pass:
        with torch.set_grad_enabled(False):
            
            batch['logits'], batch['logits_pl'], batch['similarity'] = model.f(batch['image'], concatenated_similarity, P['txt_features'])
            batch['preds'] = torch.sigmoid(batch['logits'])
            if batch['preds'].dim() == 1:
                batch['preds'] = torch.unsqueeze(batch['preds'], 0)
            batch['preds_np'] = batch['preds'].clone().detach().cpu().numpy()  # copy of preds for use in metrics
            batch['loss_np'] = -1
            batch['reg_loss_np'] = -1
        # save current batch data:
        logger.update_phase_data(batch)


def train(model, P, Z):
    '''
    Train the model.

    Parameters
    P: Dictionary of parameters, which completely specify the training procedure.
    Z: Dictionary of temporary objects used during training.
    '''

    best_weights_f = copy.deepcopy(model.f.state_dict())
    logger = train_logger(P) # initialize logger
    if_early_stop = False

    for epoch_idx in range(0, P['num_epochs']):
        print('start epoch [{}/{}] ...'.format(epoch_idx + 1, P['num_epochs']))
        P['epoch'] = epoch_idx + 1
        for phase in ['train', 'val', 'test']:
            # reset phase metrics:
            logger.reset_phase_data()

            # run one phase:
            t_init = time.time()
            if phase == 'train':
                run_train_phase(model, P, Z, logger, P['epoch'], phase)
                #if P['epoch'] >= P['warmup_epoch'] and P['loss'] == 'EM_APL':
                    #aysmmetric_pseudo_labeling(model, P, Z, logger, P['epoch'], phase)
            else:
                run_eval_phase(model, P, Z, logger, P['epoch'], phase)

            # save end-of-phase metrics:
            logger.compute_phase_metrics(phase, P['epoch'])

            # print epoch status:
            logger.report(t_init, time.time(), phase, P['epoch'])

            # update best epoch, if applicable:
            new_best = logger.update_best_results(phase, P['epoch'], P['val_set_variant'])
            if new_best:
                print('*** new best weights ***')
                best_weights_f = copy.deepcopy(model.f.state_dict())
                #print('\nSaving best weights for f to {}/best_model_state.pt'.format(P['save_path']))
                #torch.save(best_weights_f, os.path.join(P['save_path'], '_best_model_state.pt'))
                
            '''
            elif (not new_best) and (phase == 'val'):
                print('*** early stop ***')
                if_early_stop = True
                break
            '''
        if if_early_stop:
            break

    print('')
    print('*** TRAINING COMPLETE ***')
    print('Best epoch: {}'.format(logger.best_epoch))
    print('Best epoch validation score: {:.2f}'.format(logger.get_stop_metric('val', logger.best_epoch, P['val_set_variant'])))
    print('Best epoch test score:       {:.2f}'.format(logger.get_stop_metric('test', logger.best_epoch, 'clean')))

    return P, model, logger, best_weights_f


def initialize_training_run(P, feature_extractor, linear_classifier):

    '''
    Set up for model training.
    Parameters
    P: Dictionary of parameters, which completely specify the training procedure.
    feature_extractor: Feature extractor model to start from.
    linear_classifier: Linear classifier model to start from.
    estimated_labels: NumPy array containing estimated training set labels to start from (for ROLE).
    '''
    
    np.random.seed(P['seed'])

    Z = {}

    # accelerator:
    #GPU=1
    #device = torch.device('cuda:'+str(GPU) if torch.cuda.is_available() else 'cpu')
    #text_features = np.load('VOC20text_feature.npy')
    
    Z['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    P['txt_features'] = torch.from_numpy(P['txt_features']).to(Z['device'])

    # data:
    Z['datasets'] = datasets.get_data(P)

    # observed label matrix:
    label_matrix = Z['datasets']['train'].label_matrix
    num_examples = int(np.shape(label_matrix)[0])
    mtx = np.array(label_matrix).astype(np.int8)
    total_pos = np.sum(mtx == 1)
    total_neg = np.sum(mtx == 0)
    print('training samples: {} total'.format(num_examples))
    print('true positives: {} total, {:.2f} per example on average.'.format(total_pos, total_pos / num_examples))
    print('true negatives: {} total, {:.2f} per example on average.'.format(total_neg, total_neg / num_examples))
    observed_label_matrix = Z['datasets']['train'].label_matrix_obs
    num_examples = int(np.shape(observed_label_matrix)[0])
    obs_mtx = np.array(observed_label_matrix).astype(np.int8)
    obs_total_pos = np.sum(obs_mtx == 1)
    obs_total_neg = np.sum(obs_mtx == -1)
    print('observed positives: {} total, {:.2f} per example on average.'.format(obs_total_pos, obs_total_pos / num_examples))
    print('observed negatives: {} total, {:.2f} per example on average.'.format(obs_total_neg, obs_total_neg / num_examples))

    # save dataset-specific parameters:
    P['num_classes'] = Z['datasets']['train'].num_classes
    
    
    # dataloaders:
    Z['dataloaders'] = {}
    for phase in ['train', 'val', 'test']:
        Z['dataloaders'][phase] = torch.utils.data.DataLoader(
            Z['datasets'][phase],
            batch_size = P['bsize'],
            shuffle = phase == 'train',
            sampler = None,
            num_workers = P['num_workers'],
            drop_last = False  # FIXME
        )

    # pseudo-labeling data:
    P['unlabel_num'] = []
    for i in range(observed_label_matrix.shape[1]):
        P['unlabel_num'].append(np.sum(observed_label_matrix[:, i] == 0))

    # model:
    model = models2_mutliclip.MultilabelModel(P, Z, feature_extractor, linear_classifier)
    #model = models.MultilabelModel_baseline(P, Z, feature_extractor, linear_classifier)

    f_params = [param for param in list(model.f.parameters()) if param.requires_grad]
    #trans = sig_t(Z['device'], num_classes=20, init=5.75).to(Z['device'])
    Z['trans'] = sig_t(Z['device'], num_classes=20, init=8).to(Z['device'])
    Z['optimizer'] = torch.optim.Adam([
       {'params': f_params, 'lr': P['lr']},
        {'params': Z['trans'].parameters(), 'lr':P['lr']*300}])
    #Z['trans'] = sig_t(Z['device'], num_classes=20, init=5.75).to(Z['device'])

# Set up the optimizer with different learning rates for `model` and `Z['trans']`
    #Z['optimizer'] = torch.optim.Adam(
    #   f_params,
    #   lr=P['lr']
    #)

    return P, Z, model
def execute_training_run(P, feature_extractor, linear_classifier):

    '''
    Initialize, run the training process, and save the results.

    Parameters
    P: Dictionary of parameters, which completely specify the training procedure.
    feature_extractor: Feature extractor model to start from.
    linear_classifier: Linear classifier model to start from.
    estimated_labels: NumPy array containing estimated training set labels to start from (for ROLE).
    
    '''
    
    P, Z, model = initialize_training_run(P, feature_extractor, linear_classifier)
    model.to(Z['device'])

    P, model, logger, best_weights_f = train(model, P, Z)

    final_logs = logger.get_logs()
    model.f.load_state_dict(best_weights_f)

    return model.f.feature_extractor, model.f.linear_classifier, final_logs

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SPML_CLIP')
    parser.add_argument('-g', '--gpu', default='0', choices=['0', '1', '2', '3'], type=str)
    parser.add_argument('-d', '--dataset', default='pascal', choices=['pascal', 'coco', 'nuswide', 'cub'], type=str)
    parser.add_argument('-l', '--loss', default='EM_PL', choices=['bce', 'iun', 'an', 'EM', 'EM_APL', 'EM_PL', 'EM_PL_ASL'], type=str)
    parser.add_argument('-m', '--model', default='resnet50', choices=['clip_vision','resnet50', 'convnext_xlarge_22k', 'convnext_xlarge_1k'], type=str)
    parser.add_argument('-t', '--temp', default=0.01, type=float)
    parser.add_argument('-th', '--threshold', default=0.3, type=float)
    parser.add_argument('-p', '--partial', default=0.0, type=float)
    parser.add_argument('-s', '--pytorch_seed', default=0, type=int)  # try 0, 1, 8
    
    args = parser.parse_args()

    P = {}

    # Top-level parameters:
    P['GPU'] = args.gpu
    P['dataset'] = args.dataset
    P['loss'] = args.loss
    P['val_set_variant'] = 'clean'  # clean, observed
    P['test_set_variant'] = 'clean' # clean, observed
    # System parameters:
    os.environ["CUDA_VISIBLE_DEVICES"] = P['GPU']
    P['pytorch_seed'] = args.pytorch_seed
    torch.manual_seed(P['pytorch_seed'])
    torch.cuda.manual_seed(P['pytorch_seed'])
    
    # Optimization parameters:
    # Optimization parameters:
    if P['dataset'] == 'pascal':
        P['bsize'] = 8 #8 for resnet50, 6 for ViT-L
        P['lr'] = 1e-5 
        P['warmup_epoch'] = 0
        P['alpha'] = 0.4
        P['beta_pos'] = 0.7  #0.7
        P['beta_neg'] = 0 
        P['unknown']  = 4.0  #P['alpha'] = 0.2
        P['positive'] = 2.0  #P['beta_neg'] = 0.0 
        P['negative'] = 4.0  #P['beta_pos'] = 0.2 
        P['txt_features'] = np.load('VOC20text_feature_labelonly.npy')
        P['partial'] = 0
        
        P['temp'] = args.temp #[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
        P['threshold'] = args.threshold #[0.1, 0.15, 0.2, 0.25, 0.3]
        
    elif P['dataset'] == 'cub':
        P['bsize'] = 8 #8 for resnet50, 6 for ViT-L
        P['lr'] = 1e-4 #1e-4 resnet50
        P['warmup_epoch'] = 0
        P['unknown']  = 4.0  #P['alpha'] = 0.2
        P['positive'] = 2.0  #P['beta_neg'] = 0.0 
        P['negative'] = 4.0
        P['alpha'] = 0.01
        P['beta_pos'] = 0.7
        P['beta_neg'] = 0.0 #0.2
        P['txt_features'] = np.load('CUB312text_feature.npy') 
        P['partial'] = 0.0
        
        P['temp'] = args.temp #[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
        P['threshold'] = args.threshold #[0.1, 0.15, 0.2, 0.25, 0.3]
        
    elif P['dataset'] == 'nuswide':
        P['bsize'] = 8 #8 for resnet50, 6 for Vit-L
        P['lr'] = 1e-5
        P['warmup_epoch'] = 0
        P['unknown']  = 4  #P['alpha'] = 0.2
        P['positive'] = 2  #P['beta_neg'] = 0.0 
        P['negative'] = 4
        P['alpha'] = 0.1
        P['beta_pos'] = 0.7
        P['beta_neg'] = 0.0 
        P['partial'] = 0.0
        P['txt_features'] = np.load('NUS81text_feature.npy') 
        
        P['temp'] = args.temp #[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
        P['threshold'] = args.threshold #[0.1, 0.15, 0.2, 0.25, 0.3]
        
    elif P['dataset'] == 'coco':
        P['bsize'] = 8 #8 for resnet50, 6 for ViT-L
        P['lr'] = 1e-5 
        P['warmup_epoch'] = 0
        P['unknown']  = 4  #P['alpha'] = 0.2
        P['positive'] = 2  #P['beta_neg'] = 0.0 
        P['negative'] = 4  #P['beta_pos'] = 0.2 
        P['alpha'] = 0.1
        P['beta_pos'] = 0.7
        P['beta_neg'] = 0.0 
        P['partial'] = 0.0
        P['txt_features'] = np.load('CoCo80text_feature_labelonly.npy')
        
        P['temp'] = args.temp #[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
        P['threshold'] = args.threshold ##[0.1, 0.15, 0.2, 0.25, 0.3]
        
       

    # Additional parameters:
    P['seed'] = 1200  # overall numpy seed
    P['use_pretrained'] = True  # True, False
    P['num_workers'] = 8
    P['stop_metric'] = 'map'  # metric used to select the best epoch

    # Dataset parameters:
    P['split_seed'] = 1200  # seed for train/val splitting
    P['val_frac'] = 0.2  # fraction of train set to split off for val
    P['ss_seed'] = 999  # seed for subsampling
    P['ss_frac_train'] = 1.0  # fraction of training set to subsample
    P['ss_frac_val'] = 1.0  # fraction of val set to subsample

    # Dependent parameters:
    if P['loss'] == 'bce':
        P['train_set_variant'] = 'clean'
    else:
        P['train_set_variant'] = 'observed'

    # training parameters:
    P['num_epochs'] = 10
    P['freeze_feature_extractor'] = False
    P['use_feats'] = False
    P['arch'] = args.model #{'clip_vision','resnet50', 'convnext_xlarge_22k', 'convnext_xlarge_1k','clip_vision_1k+12k'}
    #P['feature_extractor_arch'] = 'resnet50'
    #P['feat_dim'] = 2048 
    P['save_path'] = './results/' + P['dataset'] + P['arch']
    # run training process:
    print('[{} + {}] start exp ...'.format(P['dataset'], P['loss']))
    print("P is: ", P)
   
    (feature_extractor, linear_classifier, logs) = execute_training_run(P, feature_extractor=None, linear_classifier=None)
