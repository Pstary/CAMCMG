import torch
import numpy as np
def tokenize_line(line):
    return line.strip().split()

def get_change_tokens(source_str):
    lines = source_str.strip().split("<nl>")
    added_tokens = []
    deleted_tokens = []
    modified_tokens = []  
    context_tokens = []
    source_tokens = []

    i = 0
    in_context_block = False  

    while i < len(lines):
        line = lines[i].strip()

        if not line:
            i += 1
            continue

        if line.startswith('-') or line.startswith('+'):
            if in_context_block:
                source_tokens.append("<KEEP_END>")
                in_context_block = False

        if line.startswith('-'):
            del_lines = []
            while i < len(lines) and lines[i].strip().startswith('-'):
                del_lines.append(lines[i].strip())
                i += 1

            add_lines = []
            while i < len(lines) and lines[i].strip().startswith('+'):
                add_lines.append(lines[i].strip())
                i += 1

            if add_lines:
                del_tokens = [tok for l in del_lines for tok in tokenize_line(l)]
                add_tokens = [tok for l in add_lines for tok in tokenize_line(l)]
                modified_tokens.append((del_tokens, add_tokens))

                source_tokens.extend(["<REPLACE_OLD>"] + del_tokens)
                source_tokens.extend(["<REPLACE_NEW>"] + add_tokens + ["<REPLACE_END>"])
            else:
                for line in del_lines:
                    toks = tokenize_line(line)
                    deleted_tokens.extend(toks)
                    source_tokens.extend(["<DELETE>"] + toks + ["<DELETE_END>"])

        elif line.startswith('+'):
            toks = tokenize_line(line.strip())
            added_tokens.extend(toks)
            source_tokens.extend(["<INSERT>"] + toks + ["<INSERT_END>"])
            i += 1

        else:
            toks = tokenize_line(line)
            context_tokens.extend(toks)
            if not in_context_block:
                source_tokens.append("<KEEP>")
                in_context_block = True
            source_tokens.extend(toks)
            i += 1

    if in_context_block:
        source_tokens.append("<KEEP_END>")

    return source_tokens

def generate_masks(source_tokens, tokenizer, window_size=64):
    seq_length = len(source_tokens) 
    
    change_mask = []
    in_added = False
    in_deleted = False
    in_modified = False
    
    for token in source_tokens:
        if token == "<INSERT>":
            in_added = True
            change_mask.append(False)
        elif token == "<INSERT_END>":
            in_added = False
            change_mask.append(False)
        elif token == "<DELETE>":
            in_deleted = True
            change_mask.append(False)
        elif token == "<DELETE_END>":
            in_deleted = False
            change_mask.append(False)
        elif token == "<REPLACE_OLD>":
            in_modified = True
            change_mask.append(False)
        elif token == "<REPLACE_NEW>":
            in_modified = True
            change_mask.append(False)
        elif token == "<REPLACE_END>":
            in_modified = False
            change_mask.append(False)
        elif token == "<KEEP>":
            in_added = in_deleted = in_modified = False
            change_mask.append(False)
        elif token == "<KEEP_END>":
            change_mask.append(False)
        elif token == tokenizer.pad_token:
            change_mask.append(False) 
        else:
            change_mask.append(in_added or in_deleted or in_modified)
    
    local_mask = generate_local_mask(seq_length, window_size=window_size)
    special_tokens = ["<INSERT>", "<INSERT_OLD>", "<DELETE>", "<DELETE_END>", "<KEEP>", "<KEEP_END>","<REPLACE_OLD>", "<REPLACE_OLD>", "<REPLACE_END>",
                     tokenizer.cls_token, tokenizer.sep_token]
    
    for i in range(seq_length):
        token = source_tokens[i]
        if token in special_tokens:
            local_mask[i, :] = False
            local_mask[:, i] = False
    return local_mask, change_mask

def generate_local_mask(seq_length, window_size):
    local_mask = torch.zeros(seq_length, seq_length, dtype=torch.bool)
    for i in range(seq_length):
        start = max(0, i - window_size)
        end = min(seq_length, i + window_size + 1)
        local_mask[i, start:end] = True
    return local_mask

def expand_local_mask(temp_local_mask, max_length):
    n = temp_local_mask.shape[0]
    if n > max_length:
        temp_local_mask = temp_local_mask[:max_length, :max_length]
        n = max_length
        return temp_local_mask
    
    local_mask = np.zeros((max_length, max_length), dtype=bool)
    local_mask[:n, :n] = temp_local_mask
    return local_mask


