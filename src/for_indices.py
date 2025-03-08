import difflib

def get_key_from_value(d, value):
    for key, val in d.items():
        if val == value:
            return key
    return None

def get_key_from_value_start_from_idx(d, value, start_idx=0):
    if start_idx > 0:
        start_idx -= 1
    for key, val in list(d.items())[start_idx:]:
        if val == value:
            return key
    return None


def get_entity_indices(parsed_input, tokenizer):
    ### TODO handle more complicated cases.
    
    prompt = parsed_input[0]['prompt']
    entities = parsed_input[0]['parsed_input']['entities']
    
    entity_indices = []
    
    token_idx_to_word = {idx: tokenizer.decode(t)
                         for idx, t in enumerate(tokenizer(prompt)['input_ids'])
                         if 0 < idx < len(tokenizer(prompt)['input_ids']) - 1}
    
    last_idx = 0
    for entity in entities:
        splitted = entity.split(' ')
        if len(splitted) > 1:
            idx = (get_key_from_value_start_from_idx(token_idx_to_word, splitted[0], last_idx), get_key_from_value_start_from_idx(token_idx_to_word, splitted[1], last_idx))
            if idx[0] != None and idx[1] != None:
                last_idx = idx[1] + 1
            else:
                idx = None
        else:
            idx = get_key_from_value_start_from_idx(token_idx_to_word, entity, last_idx)
            if idx != None:
                last_idx = idx + 1
                
        if idx != None:
            entity_indices.append(idx)
        

    return entity_indices


def get_subject_indices_for_attend_and_excite(parsed_input, tokenizer):
    ### TODO handle more complicated cases.
    
    prompt = parsed_input[0]['prompt']
    entities = parsed_input[0]['parsed_input']['entities']
    
    subject_indices = []
    
    token_idx_to_word = {idx: tokenizer.decode(t)
                         for idx, t in enumerate(tokenizer(prompt)['input_ids'])
                         if 0 < idx < len(tokenizer(prompt)['input_ids']) - 1}
    
    last_idx = 0
    for entity in entities:
        splitted = entity.split(' ')
        if len(splitted) > 1:
            idx = get_key_from_value_start_from_idx(token_idx_to_word, splitted[-1], last_idx)
            if idx != None:
                last_idx = idx + 1
        else:
            idx = get_key_from_value_start_from_idx(token_idx_to_word, entity, last_idx)
            if idx != None:
                last_idx = idx + 1
                
        if idx != None:
            subject_indices.append(idx)

    return subject_indices


def get_adj_entity_indices(constraints, tokenizer):
    ### TODO handle more complicated cases.
    
    prompt = constraints['prompt']
    subsets = constraints['subset']
    adj_entity_indices = []
    
    tokenized_prompt = tokenizer(prompt)['input_ids']
    token_idx_to_word = {idx: tokenizer.decode([t])
                         for idx, t in enumerate(tokenized_prompt)
                         if 0 < idx < len(tokenized_prompt) - 1}
    
    last_adj_idx = 0
    last_entity_idx = 0
    
    for subset in subsets:
        if 'adjective' in subset.keys() and 'entity' in subset.keys():
            adj = subset['adjective']
            entity = subset['entity']

            ### for adjective
            splitted_adj = adj.split(' ')
            if len(splitted_adj) > 1:
                idx_adj = (get_key_from_value_start_from_idx(token_idx_to_word, splitted_adj[0], start_idx=last_adj_idx),
                              get_key_from_value_start_from_idx(token_idx_to_word, splitted_adj[1], start_idx=last_adj_idx))
                if idx_adj[0] != None and idx_adj[1] != None:
                    last_adj_idx = idx_adj[1] + 1
                else:
                    idx_adj = None
            else:
                idx_adj = get_key_from_value_start_from_idx(token_idx_to_word, adj, start_idx=last_adj_idx)
                if idx_adj != None:
                    last_adj_idx = idx_adj + 1  
            
            ### for noun
            splitted_entity = entity.split(' ')
            if len(splitted_entity) > 1:
                idx_entity = (get_key_from_value_start_from_idx(token_idx_to_word, splitted_entity[0], start_idx=last_entity_idx),
                              get_key_from_value_start_from_idx(token_idx_to_word, splitted_entity[1], start_idx=last_entity_idx))
                if idx_entity[0] != None and idx_entity[1] != None:
                    last_entity_idx = idx_entity[1] + 1
                else:
                    idx_entity = None
            else:
                idx_entity = get_key_from_value_start_from_idx(token_idx_to_word, entity, start_idx=last_entity_idx)
                if idx_entity != None:
                    last_entity_idx = idx_entity + 1  
            
            if idx_adj != None and idx_entity != None:
                adj_entity_indices.append((idx_adj, idx_entity))
    
    return adj_entity_indices


def get_rel_indices(constraints, tokenizer):
    ### TODO handle more complicated cases.
    
    prompt = constraints['prompt']
    s_relations = constraints['spatial']
    s_rel_indices = []
    
    tokenized_prompt = tokenizer(prompt)['input_ids']
    token_idx_to_word = {idx: tokenizer.decode([t])
                         for idx, t in enumerate(tokenized_prompt)
                         if 0 < idx < len(tokenized_prompt) - 1}
    
    last_ob1_idx = 0
    last_rel_idx = 0
    last_ob2_idx = 0
    
    for s_rel in s_relations:
        object1 = s_rel["subject"]
        relation = s_rel["spatial"]
        object2 = s_rel["entity"]
        
        #### for object1
        splitted_ob1 = object1.split(' ')
        if len(splitted_ob1) > 1:
            ob1_idx = (get_key_from_value_start_from_idx(token_idx_to_word, splitted_ob1[0], last_ob1_idx), 
                       get_key_from_value_start_from_idx(token_idx_to_word, splitted_ob1[1], last_ob1_idx))
            if ob1_idx[0] != None and ob1_idx[1] != None:
                last_ob1_idx = ob1_idx[1] + 1
            else:
                ob1_idx = None
        else:
            ob1_idx = get_key_from_value_start_from_idx(token_idx_to_word, object1, start_idx=last_ob1_idx)
            if ob1_idx != None:
                last_ob1_idx = ob1_idx + 1
                
        #### for object2
        splitted_ob2 = object2.split(' ')
        if len(splitted_ob2) > 1:
            ob2_idx = (get_key_from_value_start_from_idx(token_idx_to_word, splitted_ob2[0], start_idx=last_ob2_idx), 
                       get_key_from_value_start_from_idx(token_idx_to_word, splitted_ob2[1], start_idx=last_ob2_idx))
            if ob2_idx[0] != None and ob2_idx[1] != None:
                last_ob2_idx = ob2_idx[1] + 1
            else:
                ob2_idx = None
        else:
            ob2_idx = get_key_from_value_start_from_idx(token_idx_to_word, object2, start_idx=last_ob2_idx)
            if ob2_idx != None:
                last_ob2_idx = ob2_idx + 1 
        
        if ob1_idx != None and ob2_idx != None:
            if "right" in relation:
                s_rel_indices.append((ob1_idx, "r", ob2_idx))
            elif "left" in relation:
                s_rel_indices.append((ob1_idx, "l", ob2_idx))
            elif ("top" in relation or "above" in relation or "over" in relation):
                s_rel_indices.append((ob1_idx, "t", ob2_idx))
            elif ("bottom" in relation or "below" in relation or "under" in relation):
                s_rel_indices.append((ob1_idx, "b", ob2_idx))
            elif ("side" in relation or "next" in relation or "near" in relation): # for on we should seperately
                s_rel_indices.append((ob1_idx, "l", ob2_idx)) # left is sensible.
    return s_rel_indices


def get_intransitive_verb_indices(constraints, tokenizer):
    ### TODO handle more complicated cases.
    
    prompt = constraints['prompt']
    subsets = constraints['subset']
    subject_intransverb_indices = []
    
    tokenized_prompt = tokenizer(prompt)['input_ids']
    token_idx_to_word = {idx: tokenizer.decode([t])
                         for idx, t in enumerate(tokenized_prompt)
                         if 0 < idx < len(tokenized_prompt) - 1}
    
    print(f"token_idx_to_word: {token_idx_to_word}")
    last_subject_idx = 0
    last_verb_idx = 0
    for subset in subsets:
        if "verb_type" in subset.keys():
            if subset['verb_type'] == "intransitive":
                subject = subset['subject']
                verb = subset['verb']
                
                if verb in ["was", "were", "is", "are"]:
                    continue 
                
                splitted = subject.split(' ')
                if len(splitted) > 1:
                    subject_idx = get_key_from_value_start_from_idx(token_idx_to_word, splitted[-1], start_idx=last_subject_idx)
                    if subject_idx != None:
                        last_subject_idx = subject_idx + 1
                else:
                    subject_idx = get_key_from_value_start_from_idx(token_idx_to_word, subject, start_idx=last_subject_idx)
                    if subject_idx != None:
                        last_subject_idx = subject_idx + 1  
                
                verb_idx = get_key_from_value_start_from_idx(token_idx_to_word, verb, start_idx=last_verb_idx)
                if verb_idx !=  None:
                    last_verb_idx = verb_idx + 1
                
                if subject_idx != None and verb_idx != None:
                    subject_intransverb_indices.append((subject_idx, verb_idx))
                    
    # print(f"subject_intransverb_indices: {subject_intransverb_indices}")
    return subject_intransverb_indices


def get_transitive_verb_indices(constraints, tokenizer):
    ### TODO handle more complicated cases.
    
    prompt = constraints['prompt']
    subsets = constraints['overlap']
    subject_transverb_indices = []
    
    tokenized_prompt = tokenizer(prompt)['input_ids']
    token_idx_to_word = {idx: tokenizer.decode([t])
                         for idx, t in enumerate(tokenized_prompt)
                         if 0 < idx < len(tokenized_prompt) - 1}
    
    print(f"token_idx_to_word: {token_idx_to_word}")
    last_subject_idx = 0
    last_verb_idx = 0
    for subset in subsets:
        if "verb_type" in subset.keys():
            if subset['verb_type'] == "transitive":
                subject = subset['subject']
                verb = subset['verb']
                
                splitted = subject.split(' ')
                if len(splitted) > 1:
                    subject_idx = get_key_from_value_start_from_idx(token_idx_to_word, splitted[-1], start_idx=last_subject_idx)
                    if subject_idx != None:
                        last_subject_idx = subject_idx + 1
                else:
                    subject_idx = get_key_from_value_start_from_idx(token_idx_to_word, subject, start_idx=last_subject_idx)
                    if subject_idx != None:
                        last_subject_idx = subject_idx + 1  
                
                verb_idx = get_key_from_value_start_from_idx(token_idx_to_word, verb, start_idx=last_verb_idx)
                if verb_idx !=  None:
                    last_verb_idx = verb_idx + 1
                
                if subject_idx != None and verb_idx != None:
                    subject_transverb_indices.append((subject_idx, verb_idx))
                    
    # print(f"subject_intransverb_indices: {subject_intransverb_indices}")
    return subject_transverb_indices


def find_closest_matches(entities, input_dict):
    output_dict = {}
    combined_indices = []
    
    for entity in entities:
        closest_match = difflib.get_close_matches(entity, input_dict.keys(), n=1, cutoff=0.7)
                       
        if closest_match:
            output_dict[entity] = input_dict[closest_match[0]]
            combined_indices.extend(input_dict[closest_match[0]])
        else:
            best_match = None
            best_ratio = 0.0

            for key in input_dict.keys():
                ratio = difflib.SequenceMatcher(None, entity, key).ratio()
                
                if ratio > best_ratio:
                    best_match = key
                    best_ratio = ratio

            output_dict[entity] = input_dict[best_match]
            combined_indices.extend(input_dict[best_match])
    
    return output_dict, combined_indices


def get_indices(prompt, tokenizer):
        """Utility function to list the indices of the tokens you wish to alter"""
        ids = tokenizer(prompt).input_ids
        indices = {i: tok for tok, i in zip(tokenizer.convert_ids_to_tokens(ids), range(len(ids)))}
        return indices

def get_token_dict(prompt, tokenizer):
        input_dict = get_indices(prompt, tokenizer)
        output_dict = {}

        for key, value in input_dict.items():
            if key == 0 or key == max(input_dict.keys()):
                continue
            clean_value = value.replace('</w>', '')

            if clean_value in output_dict:
                output_dict[clean_value].append(key)
            else:
                output_dict[clean_value] = [key]
        return output_dict


def get_token_indices(prompt, entities, tokenizer):
    entities_keyword = [x.split()[-1] for x in entities]
    token_dict = get_token_dict(prompt, tokenizer)
    _, token_indices = find_closest_matches(entities_keyword,token_dict)
    return token_indices