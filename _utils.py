import json
from parse_diff import generate_masks, expand_local_mask

def add_lang_by_task(target_str, task, sub_task):
    if task == 'summarize':
        target_str = '<en> ' + target_str
    elif task == 'refine':
        target_str = '<java> ' + target_str
    elif task == 'translate':
        if sub_task == 'java-cs':
            target_str = '<c_sharp> ' + target_str
        else:
            target_str = '<java> ' + target_str
    elif task == 'concode':
        target_str = '<java> ' + target_str
    elif task == 'defect':
        target_str = target_str
    return target_str

def extract_changed_code(code_tokens):
    in_added = False
    in_deleted = False
    in_modified = False
    in_context = False

    changed_tokens = []

    for tok in code_tokens:
        if tok == "<INSERT>":
            in_added = True
            continue
        elif tok == "<INSERT_END>":
            in_added = False
            continue
        elif tok == "<DELETE>":
            in_deleted = True
            continue
        elif tok == "<DELETE_END>":
            in_deleted = False
            continue
        elif tok == "<REPLACE_OLD>" or tok == "<REPLACE_NEW>":
            in_modified = True
            continue
        elif tok == "<REPLACE_END>":
            in_modified = False
            continue
        elif tok == "<KEEP>":
            in_context = True
            continue
        elif tok == "<KEEP_END>":
            in_context = False
            continue

        if tok.startswith("<") and tok.endswith(">"):
            continue

        if not in_context and (in_added or in_deleted or in_modified):
            changed_tokens.append(tok.lower())

    return changed_tokens


def create_ca_mask(target_str, tokenizer, code_tokens, max_target_length):

    changed_code_tokens = extract_changed_code(code_tokens)

    target_str = target_str.replace('</s>', '<unk>')
    word_list = target_str.strip().split()
    encoding = tokenizer(
        text=[word_list],
        is_split_into_words=True,
        padding='max_length',
        truncation=True,
        max_length=max_target_length,
        return_attention_mask=True
    )

    word_ids = encoding.word_ids(batch_index=0)
    target_ids = encoding['input_ids'][0] 

    ca_mask = [0] * len(target_ids)
    for i, wid in enumerate(word_ids):
        if wid is None:
            continue
        word = word_list[wid].lower()
        if word in changed_code_tokens:
            ca_mask[i] = 1

    return ca_mask, target_ids

def convert_examples_to_features(item):
    example, example_index, tokenizer, args, stage = item

    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        if args.sub_task != 'none':
            source_str = "{} {}: {}".format(args.task, args.sub_task, example.source)
        else:
            source_str = "{}: {}".format(args.task, example.source)
    else:
        source_str = example.source
    
    source_str = source_str.replace('</s>', '<unk>')
    source_ids = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    assert source_ids.count(tokenizer.eos_token_id) == 1
    if stage == 'test':
        target_ids = []
        ca_mask = []
    else:
        target_str = example.target
        if args.add_lang_ids:
            target_str = add_lang_by_task(example.target, args.task, args.sub_task)
        if args.task in ['defect', 'clone']:
            if target_str == 0:
                target_str = 'false'
            elif target_str == 1:
                target_str = 'true'
            else:
                raise NameError
        target_str = target_str.replace('</s>', '<unk>')
        ca_mask, target_ids = create_ca_mask(
            target_str=target_str,
            tokenizer=tokenizer,
            code_tokens=source_str.split(),
            max_target_length=args.max_target_length
        )
        assert target_ids.count(tokenizer.eos_token_id) == 1
    
   
    return InputFeatures(
        example_index,
        source_ids,
        target_ids,
        url=example.url,
        ca_mask=ca_mask
    )

def post_process_feature(args_tuple):
    feature, example, tokenizer, args = args_tuple
    source_str = example.source.replace('</s>', '<unk>')
    source_tokens = source_str.split()
    
    if len(source_tokens) > args.max_source_length - 2:
        source_tokens = source_tokens[:args.max_source_length - 2]
    
    source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]

    temp_local_mask, change_mask = generate_masks(source_tokens, tokenizer, window_size=args.window_size)

    local_mask = expand_local_mask(temp_local_mask, args.max_source_length)
    change_mask += [False] * (args.max_source_length - len(change_mask))

    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    source_ids += [tokenizer.pad_token_id] * (args.max_source_length - len(source_ids))

    return InputFeatures(
        example.idx,
        source_ids,
        feature.target_ids,
        local_mask=local_mask,
        change_mask=change_mask,
        url=example.url,
        ca_mask=feature.ca_mask
    )


def convert_clone_examples_to_features(item):
    example, example_index, tokenizer, args = item
    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        source_str = "{}: {}".format(args.task, example.source)
        target_str = "{}: {}".format(args.task, example.target)
    else:
        source_str = example.source
        target_str = example.target
    code1 = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    code2 = tokenizer.encode(target_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    source_ids = code1 + code2
    return CloneInputFeatures(example_index, source_ids, example.label, example.url1, example.url2)


def convert_defect_examples_to_features(item):
    example, example_index, tokenizer, args = item
    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        source_str = "{}: {}".format(args.task, example.source)
    else:
        source_str = example.source
    code = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    return DefectInputFeatures(example_index, code, example.target)


class CloneInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 label,
                 url1,
                 url2
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label
        self.url1 = url1
        self.url2 = url2


class DefectInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 label
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 local_mask=None,
                 change_mask=None,
                 url=None,
                 ca_mask=None
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.local_mask = local_mask
        self.change_mask = change_mask
        self.url = url
        self.ca_mask = ca_mask


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 url=None,
                 task='',
                 sub_task=''
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.url = url
        self.task = task
        self.sub_task = sub_task


class CloneExample(object):
    """A single training/test example."""

    def __init__(self,
                 code1,
                 code2,
                 label,
                 url1,
                 url2
                 ):
        self.source = code1
        self.target = code2
        self.label = label
        self.url1 = url1
        self.url2 = url2


def read_translate_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0
    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            src = line1.strip()
            trg = line2.strip()
            examples.append(
                Example(
                    idx=idx,
                    source=src,
                    target=trg,
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_refine_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0

    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            examples.append(
                Example(
                    idx=idx,
                    source=line1.strip(),
                    target=line2.strip(),
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_concode_examples(filename, data_num):
    """Read examples from filename."""
    examples = []

    with open(filename) as f:
        for idx, line in enumerate(f):
            x = json.loads(line)
            examples.append(
                Example(
                    idx=idx,
                    source=x["nl"].strip(),
                    target=x["code"].strip()
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_summarize_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            code = ' '.join(js['code_tokens']).replace('\n', ' ')
            code = ' '.join(code.strip().split())
            nl = ' '.join(js['docstring_tokens']).replace('\n', '')
            nl = ' '.join(nl.strip().split())
            examples.append(
                Example(
                    idx=idx,
                    source=code,
                    target=nl,
                )
            )
            if idx + 1 == data_num:
                break
    return examples


def read_defect_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)

            code = ' '.join(js['func'].split())
            examples.append(
                Example(
                    idx=js['idx'],
                    source=code,
                    target=js['target']
                )
            )
            if idx + 1 == data_num:
                break
    return examples


def read_clone_examples(filename, data_num):
    """Read examples from filename."""
    index_filename = filename
    url_to_code = {}
    with open('/'.join(index_filename.split('/')[:-1]) + '/data.jsonl') as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            code = ' '.join(js['func'].split())
            url_to_code[js['idx']] = code

    data = []
    with open(index_filename) as f:
        idx = 0
        for line in f:
            line = line.strip()
            url1, url2, label = line.split('\t')
            if url1 not in url_to_code or url2 not in url_to_code:
                continue
            if label == '0':
                label = 0
            else:
                label = 1
            data.append(CloneExample(url_to_code[url1], url_to_code[url2], label, url1, url2))
            idx += 1
            if idx == data_num:
                break
    return data
