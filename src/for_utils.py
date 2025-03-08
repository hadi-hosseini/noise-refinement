import json
import numpy as np
from scipy.ndimage import label, find_objects
from PIL import Image

import torch
import torch.nn.functional as F
import torch.linalg as linalg
import matplotlib.pyplot as plt
import cv2
import difflib
import spacy

from src.ptp_utils import AttentionStore, aggregate_attention 
from src.guassian_smoothing import GaussianSmoothing
from src.for_losses import compute_iou, remove_first_token_from_attention_map, sigmoid_distance_loss
from src.for_indices import get_token_indices
from src.dascore_metric import compute_dascores
from src.delete_adjective import find_adjectives_and_delete_all


def compute_max_attention_per_index(prompt,
                                    attention_maps,
                                    entity_indices,
                                    tokenizer,
                                    smooth_attentions: bool = False,
                                    sigma: float = 0.5,
                                    kernel_size: int = 3,
                                    normalize_eot: bool = False):
    last_idx = -1
    if normalize_eot:
        if isinstance(prompt, list):
            prompt = prompt[0]
        last_idx = len(tokenizer(prompt)['input_ids']) - 1
    attention_for_text = attention_maps[:, :, 1:last_idx]
    attention_for_text *= 100
    attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

    entity_indices = [index - 1 for index in entity_indices]

    max_indices_list = []
    for i in entity_indices:
        image = attention_for_text[:, :, i]
        if smooth_attentions:
            smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).cuda()
            input = F.pad(image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
            image = smoothing(input).squeeze(0).squeeze(0)
        max_indices_list.append(image.max())
    return max_indices_list


def find_faulty_subject_token(prompt, attention_store: AttentionStore,
                                                entity_indices,
                                                tokenizer,
                                                adj_entity_indices,
                                                s_rel_indices,
                                                attention_res: int = 16,
                                                smooth_attentions: bool = False,
                                                sigma: float = 0.5,
                                                kernel_size: int = 3,
                                                normalize_eot: bool = False):

    attention_maps = aggregate_attention(
        attention_store=attention_store,
        res=attention_res,
        from_where=("up", "down", "mid"),
        is_cross=True,
        select=0)
        
    max_attention_per_index = compute_max_attention_per_index(
            prompt=prompt,
            attention_maps=attention_maps,
            entity_indices=entity_indices,
            tokenizer = tokenizer,
            smooth_attentions=smooth_attentions,
            sigma=sigma,
            kernel_size=kernel_size,
            normalize_eot=normalize_eot)
    min_token, min_index = None, None

    if max_attention_per_index:
        min_index = max_attention_per_index.index(min(max_attention_per_index))
        min_token = entity_indices[min_index]
        

    faulty_adj_entity_index = [None, None]
    for adj_entity_index in adj_entity_indices:
        if min_token in adj_entity_index:
            faulty_adj_entity_index = adj_entity_index
            break
    
    faulty_s_rel_index = [None, None, None]
    
    for s_rel_index in s_rel_indices:
        if min_token in s_rel_index:
            faulty_s_rel_index = s_rel_index
            break

    return [(min_token, faulty_adj_entity_index, faulty_s_rel_index)]

  
def improve_faulty_attention_map_continious(faulty_subject_token_idx, attention_store, subject_indices, adjecitve_entity_index_for_refinement, s_rel_index_for_refinement, multi_loss, last_attention_maps): 
    attention_maps = aggregate_attention(attention_store=attention_store, res=16, from_where=("up", "down", "mid"), is_cross=True, select=0)   
    attention_maps = remove_first_token_from_attention_map(attention_maps=attention_maps, last_index=None)
    loss = 0 

    if faulty_subject_token_idx != None:
        objective = 0
        for j in range(len(subject_indices)):
            if subject_indices[j] == faulty_subject_token_idx:
                continue
            att_map1 = attention_maps[:, :, faulty_subject_token_idx-1]
            att_map2 = attention_maps[:, :, subject_indices[j]-1]

            att_map1 = 100*(att_map1 / att_map1.sum())
            att_map2 = 100*(att_map2 / att_map2.sum())
            
            objective = objective + max(torch.max(att_map1 - att_map2), 0)
        
        if len(subject_indices) > 1:
            loss = loss + max(0, 1.0 - (objective/(len(subject_indices)-1)))
        
        for j in range(len(subject_indices)):
            if subject_indices[j] == faulty_subject_token_idx:
                continue
            att_map1 = attention_maps[:, :, faulty_subject_token_idx-1]
            att_map2 = attention_maps[:, :, subject_indices[j]-1]
            loss = loss + compute_iou(att_map1, att_map2)   

            # loss = loss + (1 - compute_iou(attention_maps[:, :, subject_indices[j] - 1], last_attention_maps[:, :, subject_indices[j] - 1]))
        
    if multi_loss:
        if adjecitve_entity_index_for_refinement[0] != None and adjecitve_entity_index_for_refinement[1] != None:
            att_map1 = attention_maps[:, :, adjecitve_entity_index_for_refinement[0]-1]
            att_map2 = attention_maps[:, :, adjecitve_entity_index_for_refinement[1]-1]
            entity_adjective_refinement_loss = 1 - compute_iou(att_map1, att_map2)
            loss = loss + entity_adjective_refinement_loss
        
        
        if s_rel_index_for_refinement[0] != None and s_rel_index_for_refinement[2] != None:
            print('spatial relationship refinement loss')
            att_map1 = attention_maps[:, :, s_rel_index_for_refinement[0]-1]
            att_map2 = attention_maps[:, :, s_rel_index_for_refinement[2]-1]


            if s_rel_index_for_refinement[1] == "l": # left
                loss = loss + sigmoid_distance_loss(att_map1, att_map2, dim=0)

            elif s_rel_index_for_refinement[1] == "r": # right
                loss = loss + sigmoid_distance_loss(att_map2, att_map1, dim=0)
            
            elif s_rel_index_for_refinement[1] == "t": # top 
                loss = loss + sigmoid_distance_loss(att_map1, att_map2, dim=1)

            elif s_rel_index_for_refinement[1] == "b": # bottom
                loss = loss + sigmoid_distance_loss(att_map2, att_map1, dim=1)
         
    return loss


def compute_mean_attention_per_index(attention_maps: torch.Tensor,
                                         indices_to_alter,
                                         smooth_attentions: bool = False,
                                         sigma: float = 0.5,
                                         kernel_size: int = 3,):
        last_idx = -1
        attention_for_text = attention_maps[:, :, 1:last_idx]
        attention_for_text *= 100
        attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        indices_to_alter = [index - 1 for index in indices_to_alter]

        max_indices_list = []
        for i in indices_to_alter:
            image = attention_for_text[:, :, i]
            if smooth_attentions:
                smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).cuda()
                input = F.pad(image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
                image = smoothing(input).squeeze(0).squeeze(0)
            
            max_indices_list.append(image.max())
        return max_indices_list


def fo_get_faulty_word(faulty_subject_token, entities):
    closest_match = difflib.get_close_matches(faulty_subject_token, entities, n=1, cutoff=0.7)
    
    if len(closest_match):
        return closest_match[0]
    else:
        best_match = None
        best_ratio = 0.0
        for entity in entities:
            ratio = difflib.SequenceMatcher(None, faulty_subject_token, entity).ratio()
                
            if ratio > best_ratio:
                best_match = entity
                best_ratio = ratio
    return best_match



def get_entity_index(parsed_input, tokenizer, entity, subject_indices):
    prompt = parsed_input[0]['prompt']
    entities = parsed_input[0]['parsed_input']['entities']
    token_idx_to_word = {idx: tokenizer.decode(t)
                         for idx, t in enumerate(tokenizer(prompt)['input_ids'])
                         if 0 < idx < len(tokenizer(prompt)['input_ids']) - 1}
    entity = fo_get_faulty_word(faulty_subject_token=entity, entities=token_idx_to_word.values())
    for en_inx in subject_indices:
        if entity == token_idx_to_word[en_inx]:
            return en_inx
    return None


def get_adjective_entity_index(parsed_input, tokenizer, entity, adjective, adj_entity_indices):
    prompt = parsed_input[0]['prompt']
        
    token_idx_to_word = {idx: tokenizer.decode(t)
                         for idx, t in enumerate(tokenizer(prompt)['input_ids'])
                         if 0 < idx < len(tokenizer(prompt)['input_ids']) - 1}
    entity = entity.split(' ')
    adjective = adjective.split(' ')
    
    for adj_ent_index in adj_entity_indices:     

        if len(entity) == 1 and len(adjective) == 1:
            if isinstance(adj_ent_index[1], int) and isinstance(adj_ent_index[0], int):
                if entity[0] == token_idx_to_word[adj_ent_index[1]] and adjective[0] == token_idx_to_word[adj_ent_index[0]]:
                    return adj_ent_index
        elif len(entity) > 1 and len(adjective) == 1:
            if not isinstance(adj_ent_index[1], int) and isinstance(adj_ent_index[0], int):
                if entity[-1] == token_idx_to_word[adj_ent_index[1][-1]] and adjective[0] == token_idx_to_word[adj_ent_index[0]]:
                    return adj_ent_index[0], adj_ent_index[1][-1]
        elif len(entity) == 1 and len(adjective) > 1:
            if not isinstance(adj_ent_index[0], int) and isinstance(adj_ent_index[1], int):
                if entity[0] == token_idx_to_word[adj_ent_index[1]] and adjective[-1] == token_idx_to_word[adj_ent_index[0][-1]]:
                    return adj_ent_index[0][-1], adj_ent_index[1]
        elif len(entity) > 1 and len(adjective) > 1:
            if not isinstance(adj_ent_index[1], int) and not isinstance(adj_ent_index[0], int):
                if entity[-1] == token_idx_to_word[adj_ent_index[1][-1]] and adjective[-1] == token_idx_to_word[adj_ent_index[0][-1]]:
                    return adj_ent_index[0][-1], adj_ent_index[1][-1]

    return None, None



def get_spatial_rel_index(parsed_input, tokenizer, object1, rel, object2, s_rel_indices):
    prompt = parsed_input[0]['prompt']
        
    token_idx_to_word = {idx: tokenizer.decode(t)
                         for idx, t in enumerate(tokenizer(prompt)['input_ids'])
                         if 0 < idx < len(tokenizer(prompt)['input_ids']) - 1}

    object1 = object1.split(' ')
    object2 = object2.split(' ')
    for s_rel_index in s_rel_indices:
        if len(object1) == 1 and len(object2) == 1:
            if isinstance(s_rel_index[0], int) and isinstance(s_rel_index[2], int):
                if object1[0] == token_idx_to_word[s_rel_index[0]] and object2[0] == token_idx_to_word[s_rel_index[2]]:
                    return s_rel_index
        elif len(object1) > 1 and len(object2) == 1:
            if not isinstance(s_rel_index[0], int) and isinstance(s_rel_index[2], int):
                if object1[-1] == token_idx_to_word[s_rel_index[0][-1]] and object2[0] == token_idx_to_word[s_rel_index[2]]:
                    return s_rel_index[0][-1], s_rel_index[1], s_rel_index[2]
        elif len(object1) == 1 and len(object2) > 1:
            if not isinstance(s_rel_index[2], int) and isinstance(s_rel_index[0], int):
                if object1[0] == token_idx_to_word[s_rel_index[0]] and object2[-1] == token_idx_to_word[s_rel_index[2][-1]]:
                    return s_rel_index[0], s_rel_index[1], s_rel_index[2][-1]
        elif len(object1) > 1 and len(object2) > 1:
            if not isinstance(s_rel_index[0], int) and not isinstance(s_rel_index[2], int):
                if object1[-1] == token_idx_to_word[s_rel_index[0][-1]] and object2[-1] == token_idx_to_word[s_rel_index[2][-1]]:
                    return s_rel_index[0][-1], s_rel_index[1], s_rel_index[2][-1]
    
    return None, None, None



def crop_object_with_attention(image, attention_map, output_path='object.jpg', expand_ratio=0.1): # 0.1
    image = np.array(image)
    attention_map = np.array(attention_map, dtype=np.float32)
    
    attention_map_upsampled = cv2.resize(attention_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    attention_map_blurred = cv2.GaussianBlur(attention_map_upsampled, (5, 5), 0)    
    attention_map_normalized = (attention_map_upsampled - attention_map_upsampled.min()) / (attention_map_upsampled.max() - attention_map_upsampled.min())

    threshold_value = np.percentile(attention_map_normalized, 90) # 90
    _, binary_mask = cv2.threshold(attention_map_normalized, threshold_value, 1, cv2.THRESH_BINARY)
    
    binary_mask = (binary_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        
        x_expand = int(w * expand_ratio)
        y_expand = int(h * expand_ratio)
        x = max(0, x - x_expand)
        y = max(0, y - y_expand)
        w = min(image.shape[1] - x, w + 2 * x_expand)
        h = min(image.shape[0] - y, h + 2 * y_expand)
        
        cropped_image = image[y:y+h, x:x+w]
        
        cropped_image_pil = Image.fromarray(cropped_image)
        # cropped_image_pil.save(output_path)
        return cropped_image_pil
    
    return None


def detect_object_with_attention_and_draw_rectangle(selected_color, image, attention_map, output_path='object.jpg', expand_ratio=0.1): # 0.1
    image = np.array(image)
    attention_map = np.array(attention_map, dtype=np.float32)
    
    attention_map_upsampled = cv2.resize(attention_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    attention_map_blurred = cv2.GaussianBlur(attention_map_upsampled, (5, 5), 0)
    
    attention_map_normalized = (attention_map_upsampled - attention_map_upsampled.min()) / (attention_map_upsampled.max() - attention_map_upsampled.min())

    threshold_value = np.percentile(attention_map_normalized, 90) # 90
    _, binary_mask = cv2.threshold(attention_map_normalized, threshold_value, 1, cv2.THRESH_BINARY)
    
    binary_mask = (binary_mask * 255).astype(np.uint8)
    
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        
        x_expand = int(w * expand_ratio)
        y_expand = int(h * expand_ratio)
        x = max(0, x - x_expand)
        y = max(0, y - y_expand)
        w = min(image.shape[1] - x, w + 2 * x_expand)
        h = min(image.shape[0] - y, h + 2 * y_expand)

        cv2.rectangle(image, (x, y), (x + w, y + h), selected_color, 2)
        
        cropped_image = image[y:y+h, x:x+w]
        
        cropped_image_pil = Image.fromarray(cropped_image)
        cropped_image_pil.save(output_path)
        image = Image.fromarray(image)
        return image
        
    return image

def get_score_for_object_question(image, attention_map, question, vqa_model):
    box = crop_object_with_attention(image, attention_map)
    fine_grained_score = compute_dascores([box], [question], vqa_model)[0]
    coarse_grained_score = compute_dascores([image], [question], vqa_model)[0]
    score = (fine_grained_score + coarse_grained_score)/2
    return score

def get_adjecitve_entity_index_from_question(question, parsed_input, constraints, tokenizer, adj_entity_indices):
    subsets = constraints['subset']
    adj_index, entity_index = None, None
    for subset in subsets:
        if 'adjective' in subset.keys() and 'entity' in subset.keys():
            adj = subset['adjective']
            entity = subset['entity']
            if adj in question and entity in question:
                adj_index, entity_index = get_adjective_entity_index(parsed_input=parsed_input, tokenizer=tokenizer,
                                                                     entity=entity, adjective=adj, adj_entity_indices=adj_entity_indices)
                break 
    return [adj_index, entity_index]

def get_spatial_rel_index_from_question(question, parsed_input, constraints, tokenizer, s_rel_indices):
    s_relations = constraints['spatial']
    ob1_index, rel, ob2_index = None, None, None
    for s_rel in s_relations:
        object1 = s_rel["subject"]
        relation = s_rel["spatial"]
        object2 = s_rel["entity"]
        if object1 in question and relation in question and object2 in question:
            ob1_index, rel, ob2_index = get_spatial_rel_index(parsed_input=parsed_input, tokenizer=tokenizer, object1=object1, rel=relation,
                                                              object2=object2, s_rel_indices=s_rel_indices)
            break

    return [ob1_index, rel, ob2_index]


def find_faulty_subject_token_with_attention_maps(image,
                                                  assertion_mapping,
                                                  parsed_input,
                                                  constraints,
                                                  controller,
                                                  tokenizer,
                                                  subject_indices,
                                                  attention_res,
                                                  vqa_model,
                                                  adj_entity_indices,
                                                  s_rel_indices):
    attention_maps = aggregate_attention(
        attention_store=controller,
        res=attention_res,
        from_where=("up", "down", "mid"),
        is_cross=True,
        select=0).detach().cpu()
    object_question_scores = []
    adjecitve_entity_index_list = []
    s_rel_index_list = []

    for key, value in assertion_mapping.items():
        entity = key
        entity_index = get_entity_index(parsed_input, tokenizer, entity, subject_indices)
        # if entity_index!=None:
        entity_index -= 1 
        score = get_score_for_object_question(image=image[0], attention_map=attention_maps[:, :, entity_index], question=value, vqa_model=vqa_model)
        adjecitve_entity_index = get_adjecitve_entity_index_from_question(question=value, parsed_input=parsed_input, constraints=constraints, tokenizer=tokenizer, adj_entity_indices=adj_entity_indices)
        adjecitve_entity_index_list.append(adjecitve_entity_index)
        s_rel_index = get_spatial_rel_index_from_question(question=value, parsed_input=parsed_input, constraints=constraints, tokenizer=tokenizer, s_rel_indices=s_rel_indices)
        s_rel_index_list.append(s_rel_index)
        object_question_scores.append(score)

    print(f'object_question_scores: {object_question_scores}')
    proper_attention_maps = {}
    for score_idx, score in enumerate(object_question_scores):
        if score >= 0.8 and score > min(object_question_scores):
            token_index = subject_indices[score_idx]
            proper_attention_maps[score_idx] = attention_maps[:, :, token_index-1]

    min_index = object_question_scores.index(min(object_question_scores))
    min_token_objective = subject_indices[min_index]

    return [(min_token_objective, adjecitve_entity_index_list[min_index], s_rel_index_list[min_index])]


def map_entities_to_subphrases(entities, sub_phrases):
    entity_dict = {}
    index_dict = {}
    for i, entity in enumerate(entities):
        entity = entity.split()[-1]
        for j, sub_phrase in enumerate(sub_phrases):
            if entity in sub_phrase:
                entity_dict[entity] = sub_phrase
                index_dict[i] = j
                break
    return entity_dict, index_dict


def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def write_json(file_path, data):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def compute_for_loss(current_attention_map, good_subject_indices, adjecitve_entity_index_list, s_rel_index_list, main_attention_maps, last_attention_maps, multi_loss):
    loss = 0
    
    ### entity missing loss to refine the current attention map
    objective = 0
    for j in range(len(good_subject_indices)):
        att_map1 = current_attention_map
        att_map2 = main_attention_maps[:, :, good_subject_indices[j]-1]

        att_map1 = 100*(att_map1 / att_map1.sum())
        att_map2 = 100*(att_map2 / att_map2.sum())

        objective = objective + max(torch.max(att_map1 - att_map2), 0)
        
    if len(good_subject_indices) > 1:
        loss = loss + max(0, 1.0 - (objective/(len(good_subject_indices)-1)))

    for j in range(len(good_subject_indices)):
        att_map1 = current_attention_map
        att_map2 = main_attention_maps[:, :, good_subject_indices[j]-1] 
        loss = loss + compute_iou(att_map1, att_map2)


    if len(good_subject_indices) == 0:
        loss += 1 - current_attention_map.max()

    if multi_loss:
        if adjecitve_entity_index_list[0] != None and adjecitve_entity_index_list[1] != None:
            att_map1 = main_attention_maps[:, :, adjecitve_entity_index_list[0]-1]
            att_map2 = main_attention_maps[:, :, adjecitve_entity_index_list[1]-1]
            entity_adjective_refinement_loss = 1 - compute_iou(att_map1, att_map2)
            loss = loss + entity_adjective_refinement_loss
            print(f'entity-adjective refinement loss: {entity_adjective_refinement_loss}')
    
        if s_rel_index_list[0] != None and s_rel_index_list[2] != None:
            print('spatial relationship refinement loss')
            att_map1 = main_attention_maps[:, :, s_rel_index_list[0]-1]
            att_map2 = main_attention_maps[:, :, s_rel_index_list[2]-1]
            
            if s_rel_index_list[1] == "l": # left
                loss = loss + sigmoid_distance_loss(att_map1, att_map2, dim=0)

            elif s_rel_index_list[1] == "r": # right
                loss = loss + sigmoid_distance_loss(att_map2, att_map1, dim=0)
        
            elif s_rel_index_list[1] == "t": # top 
                loss = loss +  sigmoid_distance_loss(att_map1, att_map2, dim=1)

            elif s_rel_index_list[1] == "b": # bottom
                loss = loss +  sigmoid_distance_loss(att_map2, att_map1, dim=1)
    

    #### to keep others in their good conditions
    # for subject_indx in good_subject_indices:
    #     loss = loss + (1 - compute_iou(main_attention_maps[:, :, subject_indx-1], last_attention_maps[:, :, subject_indx-1]))

    return loss
    
    
def improve_multi_object(faulty_information, main_attention_maps, subject_indices, pipe, latents, iterative, scale_factor, scale_range, prompt_embeds, i, t, cross_attention_kwargs, attention_store, multi_loss, last_attention_maps):
    main_attention_maps = main_attention_maps[:, :, 1:-1]
    main_attention_maps = torch.nn.functional.softmax(main_attention_maps, dim=-1)
    
    last_attention_maps = last_attention_maps[:, :, 1:-1]
    last_attention_maps = torch.nn.functional.softmax(last_attention_maps, dim=-1)
    
    faulty_subject_tokens_idx = [faulty_info[0] for faulty_info in faulty_information]
    good_subject_indices = [subject_index for subject_index in subject_indices if subject_index not in faulty_subject_tokens_idx]
    for faulty_info in faulty_information:

        faulty_token_idx = faulty_info[0]

        ##### multi-loss
        adjecitve_entity_index_list = faulty_info[1] 
        s_rel_index_list = faulty_info[2]


        # fix the faulty attention map
        current_attention_map = main_attention_maps[:, :, faulty_token_idx-1]

        # compute the loss
        loss = compute_for_loss(current_attention_map, good_subject_indices, adjecitve_entity_index_list, s_rel_index_list, main_attention_maps, last_attention_maps, multi_loss)
        print(f"subject indx: {faulty_token_idx}, loss: {loss}")

        if loss != 0:
            latents = pipe._update_latent(latents=latents, loss=loss, step_size=scale_factor * np.sqrt(scale_range[i]))
            print(f'Improvement Loss: {loss:0.4f}')
            print("--------------------------------------------------")


        noise_pred_text = pipe.unet(latents, t, encoder_hidden_states=prompt_embeds[1].unsqueeze(0), cross_attention_kwargs=cross_attention_kwargs).sample
        pipe.unet.zero_grad()

        # update the information for the next bad attention map
        main_attention_maps = aggregate_attention(attention_store=attention_store, res=16, from_where=("up", "down", "mid"), is_cross=True, select=0)
        main_attention_maps = main_attention_maps[:, :, 1:-1]
        main_attention_maps = torch.nn.functional.softmax(main_attention_maps, dim=-1) 
        
        last_attention_maps[:,:,faulty_token_idx-1] = main_attention_maps[:, :, faulty_token_idx-1].clone()
        
        good_subject_indices.append(faulty_token_idx)

    torch.cuda.empty_cache()
    return 0, latents


def find_faulty_multi_objects_tokens_with_attention_maps(image,
                                                  assertion_mapping,
                                                  parsed_input,
                                                  constraints,
                                                  controller,
                                                  tokenizer,
                                                  subject_indices,
                                                  adj_entity_indices,
                                                  s_rel_indices,
                                                  attention_res,
                                                  vqa_model,
                                                  threshold=0.5):
    attention_maps = aggregate_attention(
        attention_store=controller,
        res=attention_res,
        from_where=("up", "down", "mid"),
        is_cross=True,
        select=0).detach().cpu()
    
    
    object_fine_question_scores = []
    object_coarse_question_scores = []
    adjecitve_entity_index_list = []
    s_rel_index_list = []
    
    nlp = spacy.load("en_core_web_sm")
    
    for key, value in assertion_mapping.items():
        entity = key
        entity_index = get_entity_index(parsed_input, tokenizer, entity, subject_indices)
        print(entity, entity_index)
        if entity_index!=None:
            entity_index -= 1
            
            fine_question = value
            coarse_question = find_adjectives_and_delete_all(fine_question, nlp)
            
            
            fine_score = get_score_for_object_question(image=image[0], attention_map=attention_maps[:, :, entity_index], question=value, vqa_model=vqa_model)
            coarse_score = get_score_for_object_question(image=image[0], attention_map=attention_maps[:, :, entity_index], question=coarse_question, vqa_model=vqa_model)
            adjecitve_entity_index = get_adjecitve_entity_index_from_question(question=value, parsed_input=parsed_input, constraints=constraints, tokenizer=tokenizer, adj_entity_indices=adj_entity_indices)
            adjecitve_entity_index_list.append(adjecitve_entity_index)
            s_rel_index = get_spatial_rel_index_from_question(question=value, parsed_input=parsed_input, constraints=constraints, tokenizer=tokenizer, s_rel_indices=s_rel_indices)
            s_rel_index_list.append(s_rel_index)

            object_fine_question_scores.append(fine_score[0])
            object_coarse_question_scores.append(coarse_score[0])
            
            
    
    print(f'Object Fine Questions Scores: {object_fine_question_scores}')
    print(f'Object Coarse Questions Scores: {object_coarse_question_scores}')
    
    
    faulty_information = []
    for i, subject_token in enumerate(subject_indices[:len(object_fine_question_scores)]):
        if object_fine_question_scores[i] < threshold and object_coarse_question_scores[i] < threshold:
            faulty_information.append((subject_token, adjecitve_entity_index_list[i], s_rel_index_list[i])) # both (entity and adjective)
        elif not (object_fine_question_scores[i] < threshold) and object_coarse_question_scores[i] < threshold:
            faulty_information.append((subject_token, [None, None], s_rel_index_list[i])) # just entity
    
    
    if len(faulty_information) == 0:
        min_index = object_fine_question_scores.index(min(object_fine_question_scores))
        faulty_information = [(subject_indices[min_index], adjecitve_entity_index_list[min_index], s_rel_index_list[min_index])]

    return faulty_information
