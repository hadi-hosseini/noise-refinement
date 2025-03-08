import os, shutil
from typing import List, Union, Tuple, Dict
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from src.ptp_utils import AttentionStore, aggregate_attention, text_under_image


def remove_first_token_from_attention_map(attention_maps: torch.Tensor, last_index: int):
        last_idx = -1
        if last_index != None:
            last_idx = last_index
        attention_for_text = attention_maps[:, :, 1:last_idx].clone()
        # attention_for_text = attention_for_text * 100
        attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)
        return attention_for_text

def compute_iou(attention_map1: torch.Tensor, attention_map2: torch.Tensor, mode=1) -> float:  
    # mode 1: multiplication 
    # mode 2: minimum
    
    attention_map1 = (attention_map1 - attention_map1.min()) / (attention_map1.max() - attention_map1.min())
    attention_map2 = (attention_map2 - attention_map2.min()) / (attention_map2.max() - attention_map2.min())
    
    if mode == 1:
        intersection = attention_map1 * attention_map2
    elif mode == 2:
        intersection = torch.min(attention_map1, attention_map2)
        
    union = attention_map1 + attention_map2
    
    intersection = torch.sum(intersection)
    union = torch.sum(union)
    iou = intersection / union

    return iou

def compute_segregation_loss(prompt: Union[str, List[str]], tokenizer, entity_indices: List[int], attention_store: AttentionStore, attention_res: int = 16):
    attention_maps = aggregate_attention(
        attention_store=attention_store,
        res=attention_res,
        from_where=("up", "down", "mid"),
        is_cross=True,
        select=0)

    last_index = len(tokenizer(prompt)['input_ids']) - 1
    attention_maps = remove_first_token_from_attention_map(attention_maps=attention_maps, last_index=None)

    loss = 0
    
    for i in range(len(entity_indices)):
        for j in range(i + 1, len(entity_indices)):
            if isinstance(entity_indices[i], int):
                att_map1 = attention_maps[:, :, entity_indices[i]-1]
            else:
                att_map1 = (attention_maps[:, :, entity_indices[i][0]-1] + attention_maps[:, :, entity_indices[i][1]-1]) / 2
            
            if isinstance(entity_indices[j], int):
                att_map2 = attention_maps[:, :, entity_indices[j]-1]
            else:
                att_map2 = (attention_maps[:, :, entity_indices[j][0]-1] + attention_maps[:, :, entity_indices[j][1]-1]) / 2
            loss = loss + compute_iou(att_map1, att_map2)
    return loss

def compute_adj_loss(prompt: Union[str, List[str]], tokenizer, adj_entity_indices, attention_store: AttentionStore, attention_res: int = 16):
    attention_maps = aggregate_attention(
        attention_store=attention_store,
        res=attention_res,
        from_where=("up", "down", "mid"),
        is_cross=True,
        select=0)
    
    last_index = len(tokenizer(prompt)['input_ids']) - 1
    attention_maps = remove_first_token_from_attention_map(attention_maps=attention_maps, last_index=None)
    
    loss = 0
    for t in adj_entity_indices:
        
        ### for adjective
        if isinstance(t[0], int):
            adj_att_map = attention_maps[:, :, t[0]-1]
        else:
            adj_att_map = (attention_maps[:, :, t[0][0]-1] + attention_maps[:, :, t[0][1]-1]) / 2
        
        ### for noun
        if isinstance(t[1], int):
            entity_att_map = attention_maps[:, :, t[1]-1]
        else:
            entity_att_map = (attention_maps[:, :, t[1][0]-1] + attention_maps[:, :, t[1][1]-1]) / 2
        
        loss = loss +  (1 - compute_iou(adj_att_map, entity_att_map))
    return loss


def compute_entity_disentangle_loss(prompt: Union[str, List[str]], tokenizer, entity_indices: List[int], attention_store: AttentionStore, attention_res: int = 16):
    attention_maps = aggregate_attention(
        attention_store=attention_store,
        res=attention_res,
        from_where=("up", "down", "mid"),
        is_cross=True,
        select=0)
    
    last_index = len(tokenizer(prompt)['input_ids']) - 1
    attention_maps = remove_first_token_from_attention_map(attention_maps=attention_maps, last_index=None)
    disentangle_loss = 0
    
    for i in range(len(entity_indices)):
        objective = 0
        for j in range(len(entity_indices)):

            if isinstance(entity_indices[i], int):
                att_map1 = attention_maps[:, :, entity_indices[i]-1]
            else:
                att_map1 = (attention_maps[:, :, entity_indices[i][0]-1] + attention_maps[:, :, entity_indices[i][1]-1]) / 2
            
            if isinstance(entity_indices[j], int):
                att_map2 = attention_maps[:, :, entity_indices[j]-1]
            else:
                att_map2 = (attention_maps[:, :, entity_indices[j][0]-1] + attention_maps[:, :, entity_indices[j][1]-1]) / 2

            att_map1 = 1000 * (att_map1 / att_map1.sum())
            att_map2 = 1000 * (att_map2 / att_map2.sum())
            
            objective = objective + max(torch.max(att_map1 - att_map2), 0)
            
        if len(entity_indices) > 1:
            disentangle_loss = disentangle_loss + max(0, 1.0 - (objective/(len(entity_indices)-1)))
        else:
            disentangle_loss = disentangle_loss + max(0, 1.0 - objective)

    return disentangle_loss

def sigmoid_distance_loss(attention_map1, attention_map2, dim):
    loss = 0.0

    distribution_first = torch.sum(attention_map1, dim=dim)
    distribution_second =  torch.sum(attention_map2, dim=dim)

    normalize_attention_map1 = torch.sum(attention_map1, dim=dim) / torch.sum(distribution_first)
    normalize_attention_map2 = torch.sum(attention_map2, dim=dim) / torch.sum(distribution_second)

    domain_values = torch.arange(0, 16).cuda()
    expected_value_first = torch.sum(normalize_attention_map1 * domain_values)
    expected_value_second = torch.sum(normalize_attention_map2 * domain_values)

    distance = expected_value_first - expected_value_second
    loss = F.sigmoid(distance)

    return loss

def compute_new_spatial_loss(prompt: Union[str, List[str]], tokenizer, s_rel_indices: List[int], attention_store: AttentionStore, attention_res: int = 16):
    attention_maps = aggregate_attention(
        attention_store=attention_store,
        res=attention_res,
        from_where=("up", "down", "mid"),
        is_cross=True,
        select=0)

    attention_for_text = attention_maps[:, :, 1:-1].clone()
    attention_for_text = attention_for_text * 100
    attention_maps = torch.nn.functional.softmax(attention_for_text, dim=-1)
    
    loss = torch.tensor(0.0, requires_grad=True)
    for t in s_rel_indices:
        if isinstance(t[0], int):
            attention_map1 = attention_maps[:, :, t[0]-1]
        else:
            attention_map1 = (attention_maps[:, :, t[0][0]-1] + attention_maps[:, :, t[0][1]-1]) / 2
            
        if isinstance(t[2], int):
            attention_map2 = attention_maps[:, :, t[2]-1]
        else:
            attention_map2 = (attention_maps[:, :, t[2][0]-1] + attention_maps[:, :, t[2][1]-1]) / 2
        

        if t[1] == "l": # left
            loss = loss + sigmoid_distance_loss(attention_map1, attention_map2, dim=0)

        elif t[1] == "r": # right
            loss = loss + sigmoid_distance_loss(attention_map2, attention_map1, dim=0)
        
        elif t[1] == "t": # top 
            loss = loss +  sigmoid_distance_loss(attention_map1, attention_map2, dim=1)

        elif t[1] == "b": # bottom
            loss = loss +  sigmoid_distance_loss(attention_map2, attention_map1, dim=1)

    return loss

def compute_comtie_loss(prompt: Union[str, List[str]], tokenizer, entity_indices: List[int], adj_entity_indices, s_rel_indices: List[int], attention_store: AttentionStore, mode_loss: int = 1):
    if mode_loss == 1:
        seg_loss = compute_segregation_loss(prompt=prompt, tokenizer=tokenizer, entity_indices=entity_indices, attention_store=attention_store, attention_res=16)
        adj_iou_loss = compute_adj_loss(prompt=prompt, tokenizer=tokenizer, adj_entity_indices=adj_entity_indices, attention_store=attention_store, attention_res=16)
        s_rel_loss = compute_new_spatial_loss(prompt=prompt, tokenizer=tokenizer, s_rel_indices=s_rel_indices, attention_store=attention_store, attention_res=16)
        entity_disentangle_loss = compute_entity_disentangle_loss(prompt=prompt, tokenizer=tokenizer, entity_indices=entity_indices, attention_store=attention_store, attention_res=16)
        if s_rel_loss:
            loss = s_rel_loss + entity_disentangle_loss + adj_iou_loss
        else:
            loss = seg_loss + s_rel_loss + entity_disentangle_loss + adj_iou_loss
    return loss