import argparse
import json
import logging
import random
import time
from pathlib import Path
import logddd
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelWithLMHead, AutoTokenizer

import autoprompt.utils as utils

logger = logging.getLogger(__name__)


class GradientStorage:
    """
    此对象存储给定PyTorch模块的输出的中间梯度，否则可能不会保留。
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained.
    """
    def __init__(self, module):
        self._stored_gradient = None
        module.register_backward_hook(self.hook)

    def hook(self, module, grad_in, grad_out):
        self._stored_gradient = grad_out[0]

    def get(self):
        return self._stored_gradient


class PredictWrapper:
    """
    PyTorch transformers model wrapper. Handles necc. preprocessing of inputs for triggers
    experiments.
    """

    def __init__(self, model, config=None):
        self._model = model
        self.config = config

    def __call__(self, model_inputs, trigger_ids):
        # Copy dict so pop operations don't have unwanted side-effects
        model_inputs = model_inputs.copy()
        # 在dict中删除这两个属性
        trigger_mask = model_inputs.pop('trigger_mask')
        predict_mask = model_inputs.pop('predict_mask')
        # for item in predict_mask:
        #     print(item)
        # logddd.log(predict_mask)
        # exit(0)
        # 使用trigger_id来替换trigger位
        model_inputs = replace_trigger_tokens(model_inputs, trigger_ids, trigger_mask)
        # 模型推理
        logits, *_ = self._model(**model_inputs)
        # import logddd
        # logddd.log(logits)
        # https://blog.csdn.net/qq_43049542/article/details/125821983
        # masked_select 找出mask位置的值
        predict_logits = logits.masked_select(predict_mask.unsqueeze(-1)).view(logits.size(0), -1)
        # logddd.log(predict_logits.shape)
        # 如果传入的配置，那就按照配置的方式，截取标签位置的结果
        if self.config is not None:
            # 截取词性标签部分的预测输出
            predict_logits = predict_logits[:,1:self.config.class_num + 1]
        # logddd.log(predict_logits.shape)
        # exit(0)
        # 返回推理结果
        return predict_logits


class AccuracyFn:
    """
    Computing the accuracy when a label is mapped to multiple tokens is difficult in the current
    framework, since the data generator only gives us the token ids. To get around this we
    compare the target logp to the logp of all labels. If target logp is greater than all (but)
    one of the label logps we know we are accurate.
    """
    def __init__(self, tokenizer, label_map, device, tokenize_labels=False):
        self._all_label_ids = []
        self._pred_to_label = []
        logger.info(label_map)
        for label, label_tokens in label_map.items():
            self._all_label_ids.append(utils.encode_label(tokenizer, label_tokens, tokenize_labels).to(device))
            self._pred_to_label.append(label)
        logger.info(self._all_label_ids)

    def __call__(self, predict_logits, gold_label_ids):
        # Get total log-probability for the true label
        gold_logp = get_loss(predict_logits, gold_label_ids)

        # Get total log-probability for all labels
        bsz = predict_logits.size(0)
        all_label_logp = []
        for label_ids in self._all_label_ids:
            label_logp = get_loss(predict_logits, label_ids.repeat(bsz, 1))
            all_label_logp.append(label_logp)
        #     官方解释：沿着一个新维度对输入张量序列进行连接。 序列中所有的张量都应该为相同形状。
        # 浅显说法：把多个2维的张量凑成一个3维的张量；多个3维的凑成一个4维的张量…以此类推，也就是在增加新的维度进行堆叠。
        all_label_logp = torch.stack(all_label_logp, dim=-1)
        _, predictions = all_label_logp.max(dim=-1)
        predictions = [self._pred_to_label[x] for x in predictions.tolist()]

        # Add up the number of entries where loss is greater than or equal to gold_logp.
        ge_count = all_label_logp.le(gold_logp.unsqueeze(-1)).sum(-1)
        correct = ge_count.le(1)  # less than in case of num. prec. issues

        return correct.float()

    # TODO: @rloganiv - This is hacky. Replace with something sensible.
    def predict(self, predict_logits):
        bsz = predict_logits.size(0)
        all_label_logp = []
        for label_ids in self._all_label_ids:
            label_logp = get_loss(predict_logits, label_ids.repeat(bsz, 1))
            all_label_logp.append(label_logp)
        all_label_logp = torch.stack(all_label_logp, dim=-1)
        _, predictions = all_label_logp.max(dim=-1)
        predictions = [self._pred_to_label[x] for x in predictions.tolist()]
        return predictions


# 加载预训练模型
def load_pretrained(model_name):
    """
    Loads pretrained HuggingFace config/model/tokenizer, as well as performs required
    initialization steps to facilitate working with triggers.
    """
    model_name = "/home/dlf/autoprompt/model/bert_large_chinese"
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelWithLMHead.from_pretrained(model_name)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    utils.add_task_specific_tokens(tokenizer)
    return config, model, tokenizer


def set_seed(seed: int):
    """Sets the relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_embeddings(model, config):
    """Returns the wordpiece embedding module."""
    base_model = getattr(model, config.model_type)
    embeddings = base_model.embeddings.word_embeddings
    return embeddings


def hotflip_attack(averaged_grad,
                   embedding_matrix,
                   increase_loss=False,
                   num_candidates=1,
                   filter=None):
    """Returns the top candidate replacements."""
    with torch.no_grad():
        gradient_dot_embedding_matrix = torch.matmul(
            embedding_matrix,
            averaged_grad
        )
        if filter is not None:
            gradient_dot_embedding_matrix -= filter
        if not increase_loss:
            gradient_dot_embedding_matrix *= -1
        _, top_k_ids = gradient_dot_embedding_matrix.topk(num_candidates)

    return top_k_ids


def replace_trigger_tokens(model_inputs, trigger_ids, trigger_mask):
    """Replaces the trigger tokens in input_ids."""
    out = model_inputs.copy()
    input_ids = model_inputs['input_ids']
    trigger_ids = trigger_ids.repeat(trigger_mask.size(0), 1)
    # trigger_ids 因为没有给定初始的trigger，所以，初始的trigger全都是用mask来代替的
    # import logddd
    # logddd.log(trigger_mask)
    # for item in trigger_mask:
    #     print(item)
    # exit(0)
    try:
        # https://blog.csdn.net/qq_43049542/article/details/125821983
        # masked_scatter 的作用是按照trigger_mask（是一个判定当前位置是否mask的矩阵）
        # 将input_ids中mask位置替换为trigger_id
        filled = input_ids.masked_scatter(trigger_mask, trigger_ids)
    except RuntimeError:
        filled = input_ids
    out['input_ids'] = filled
    return out


def get_loss(predict_logits, label_ids):
    # label_ids 是对应的label
    logddd.log(predict_logits.shape)
    logddd.log(label_ids.shape)

    predict_logp = F.log_softmax(predict_logits, dim=-1)
    target_logp = predict_logp.gather(-1, label_ids)
    target_logp = target_logp - 1e32 * label_ids.eq(0)  # Apply mask
    target_logp = torch.logsumexp(target_logp, dim=-1)
    return -target_logp


def isupper(idx, tokenizer):
    """
    Determines whether a token (e.g., word piece) begins with a capital letter.
    """
    _isupper = False
    # We only want to check tokens that begin words. Since byte-pair encoding
    # captures a prefix space, we need to check that the decoded token begins
    # with a space, and has a capitalized second character.
    if isinstance(tokenizer, transformers.GPT2Tokenizer):
        decoded = tokenizer.decode([idx])
        if decoded[0] == ' ' and decoded[1].isupper():
            _isupper = True
    # For all other tokenization schemes, we can just check the first character
    # is capitalized.
    elif tokenizer.decode([idx])[0].isupper():
            _isupper = True
    return _isupper


def run_model(args):

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info('Loading model, tokenizer, etc.')
    # 加载预训练模型
    config, model, tokenizer = load_pretrained(args.model_name)
    model.to(device)
    # 
    embeddings = get_embeddings(model, config)
    # 获取中间梯度
    embedding_gradient = GradientStorage(embeddings)

    predictor = PredictWrapper(model,args)

    if args.label_map is not None:
        label_map = json.loads(args.label_map)
        logger.info(f"Label map: {label_map}")
    else:
        label_map = None
        logger.info('No label map')

    templatizer = utils.TriggerTemplatizer(
        args.template,
        config,
        tokenizer,
        label_map=label_map,
        label_field=args.label_field,
        tokenize_labels=args.tokenize_labels,
        add_special_tokens=False,
        use_ctx=args.use_ctx
    )

    # Obtain the initial trigger tokens and label mapping
    if args.initial_trigger:
        # 如果有自定义的trigger，那就用自定义的trigger
        trigger_ids = tokenizer.convert_tokens_to_ids(args.initial_trigger)
        logger.debug(f'Initial trigger: {args.initial_trigger}')
        logger.debug(f'Trigger ids: {trigger_ids}')
        assert len(trigger_ids) == templatizer.num_trigger_tokens
    else:
        # 如果没有最开始定义的trigger，那就用mask位来充当初始trigger
        trigger_ids = [tokenizer.mask_token_id] * templatizer.num_trigger_tokens
    trigger_ids = torch.tensor(trigger_ids, device=device).unsqueeze(0)
    best_trigger_ids = trigger_ids.clone()

    # NOTE: Accuracy can only be computed if a fixed pool of labels is given, which currently
    # requires the label map to be specified. Since producing a label map may be cumbersome (e.g.,
    # for link prediction tasks), we just use (negative) loss as the evaluation metric in these cases.
    # 只有给定固定的标签池，才能计算准确性，目前需要指定标签映射，由于生成标签映射很麻烦，
    if label_map:
        evaluation_fn = AccuracyFn(tokenizer, label_map, device)
    else:
        evaluation_fn = lambda x, y: -get_loss(x, y)

    logger.info('Loading datasets')
    collator = utils.Collator(pad_token_id=tokenizer.pad_token_id)

    if args.perturbed:
        train_dataset = utils.load_augmented_trigger_dataset(args.train, templatizer, limit=args.limit)
    else:
        train_dataset = utils.load_trigger_dataset(args.train, templatizer, use_ctx=args.use_ctx, limit=args.limit)
    train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)

    if args.perturbed:
        dev_dataset = utils.load_augmented_trigger_dataset(args.dev, templatizer)
    else:
        dev_dataset = utils.load_trigger_dataset(args.dev, templatizer, use_ctx=args.use_ctx)
    dev_loader = DataLoader(dev_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)

    # To "filter" unwanted trigger tokens, we subtract a huge number from their logits.
    filter = torch.zeros(tokenizer.vocab_size, dtype=torch.float32, device=device)
    if args.filter:
        logger.info('Filtering label tokens.')
        if label_map:
            for label_tokens in label_map.values():
                label_ids = utils.encode_label(tokenizer, label_tokens).unsqueeze(0)
                filter[label_ids] = -1e32
        else:
            for _, label_ids in train_dataset:
                filter[label_ids] = -1e32
        logger.info('Filtering special tokens and capitalized words.')
        for word, idx in tokenizer.get_vocab().items():
            if len(word) == 1 or idx >= tokenizer.vocab_size:
                continue
            # Filter special tokens.
            if idx in tokenizer.all_special_ids:
                logger.debug('Filtered: %s', word)
                filter[idx] = -1e32
            # Filter capitalized words (lazy way to remove proper nouns).
            if isupper(idx, tokenizer):
                logger.debug('Filtered: %s', word)
                filter[idx] = -1e32

    logger.info('Evaluating')
    numerator = 0
    denominator = 0
    for model_inputs, labels in tqdm(dev_loader):
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = labels.to(device)
        with torch.no_grad():
            logddd.log(model_inputs)
            logddd.log(trigger_ids)
            logddd.log(labels)
            # exit(0)
            predict_logits = predictor(model_inputs, trigger_ids)
            logddd.log(predict_logits.shape)
            exit(0)
        numerator += evaluation_fn(predict_logits, labels).sum().item()
        denominator += labels.size(0)
    dev_metric = numerator / (denominator + 1e-13)
    logger.info(f'Dev metric: {dev_metric}')

    best_dev_metric = -float('inf')
    # Measure elapsed time of trigger search
    start = time.time()

    for i in range(args.iters):

        logger.info(f'Iteration: {i}')

        logger.info('Accumulating Gradient')
        model.zero_grad()

        pbar = tqdm(range(args.accumulation_steps))
        train_iter = iter(train_loader)
        averaged_grad = None

        # Accumulate
        for step in pbar:

            # Shuttle inputs to GPU
            try:
                model_inputs, labels = next(train_iter)
            except:
                logger.warning(
                    'Insufficient data for number of accumulation steps. '
                    'Effective batch size will be smaller than specified.'
                )
                break
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            labels = labels.to(device)
            # here 在这里进行的训练
            predict_logits = predictor(model_inputs, trigger_ids)
            # 计算loss
            loss = get_loss(predict_logits, labels).mean()
            loss.backward()

            grad = embedding_gradient.get()
            bsz, _, emb_dim = grad.size()
            selection_mask = model_inputs['trigger_mask'].unsqueeze(-1)
            grad = torch.masked_select(grad, selection_mask)
            grad = grad.view(bsz, templatizer.num_trigger_tokens, emb_dim)

            if averaged_grad is None:
                averaged_grad = grad.sum(dim=0) / args.accumulation_steps
            else:
                averaged_grad += grad.sum(dim=0) / args.accumulation_steps

        logger.info('Evaluating Candidates')
        pbar = tqdm(range(args.accumulation_steps))
        train_iter = iter(train_loader)

        token_to_flip = random.randrange(templatizer.num_trigger_tokens)
        candidates = hotflip_attack(averaged_grad[token_to_flip],
                                    embeddings.weight,
                                    increase_loss=False,
                                    num_candidates=args.num_cand,
                                    filter=filter)

        current_score = 0
        candidate_scores = torch.zeros(args.num_cand, device=device)
        denom = 0
        for step in pbar:

            try:
                model_inputs, labels = next(train_iter)
            except:
                logger.warning(
                    'Insufficient data for number of accumulation steps. '
                    'Effective batch size will be smaller than specified.'
                )
                break
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            labels = labels.to(device)
            with torch.no_grad():
                predict_logits = predictor(model_inputs, trigger_ids)
                eval_metric = evaluation_fn(predict_logits, labels)

            # Update current score
            current_score += eval_metric.sum()
            denom += labels.size(0)

            # NOTE: Instead of iterating over tokens to flip we randomly change just one each
            # time so the gradients don't get stale.
            for i, candidate in enumerate(candidates):

                # if candidate.item() in filter_candidates:
                #     candidate_scores[i] = -1e32
                #     continue

                temp_trigger = trigger_ids.clone()
                temp_trigger[:, token_to_flip] = candidate
                with torch.no_grad():
                    predict_logits = predictor(model_inputs, temp_trigger)
                    eval_metric = evaluation_fn(predict_logits, labels)

                candidate_scores[i] += eval_metric.sum()

        # TODO: Something cleaner. LAMA templates can't have mask tokens, so if
        # there are still mask tokens in the trigger then set the current score
        # to -inf.
        if args.print_lama:
            if trigger_ids.eq(tokenizer.mask_token_id).any():
                current_score = float('-inf')

        if (candidate_scores > current_score).any():
            logger.info('Better trigger detected.')
            best_candidate_score = candidate_scores.max()
            best_candidate_idx = candidate_scores.argmax()
            trigger_ids[:, token_to_flip] = candidates[best_candidate_idx]
            logger.info(f'Train metric: {best_candidate_score / (denom + 1e-13): 0.4f}')
        else:
            logger.info('No improvement detected. Skipping evaluation.')
            continue

        logger.info('Evaluating')
        numerator = 0
        denominator = 0
        for model_inputs, labels in tqdm(dev_loader):
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            labels = labels.to(device)
            with torch.no_grad():
                predict_logits = predictor(model_inputs, trigger_ids)
            numerator += evaluation_fn(predict_logits, labels).sum().item()
            denominator += labels.size(0)
        dev_metric = numerator / (denominator + 1e-13)

        logger.info(f'Trigger tokens: {tokenizer.convert_ids_to_tokens(trigger_ids.squeeze(0))}')
        logger.info(f'Dev metric: {dev_metric}')

        # TODO: Something cleaner. LAMA templates can't have mask tokens, so if
        # there are still mask tokens in the trigger then set the current score
        # to -inf.
        if args.print_lama:
            if best_trigger_ids.eq(tokenizer.mask_token_id).any():
                best_dev_metric = float('-inf')

        if dev_metric > best_dev_metric:
            logger.info('Best performance so far')
            best_trigger_ids = trigger_ids.clone()
            best_dev_metric = dev_metric

    best_trigger_tokens = tokenizer.convert_ids_to_tokens(best_trigger_ids.squeeze(0))
    logger.info(f'Best tokens: {best_trigger_tokens}')
    logger.info(f'Best dev metric: {best_dev_metric}')
    if args.print_lama:
        # Templatize with [X] and [Y]
        if args.use_ctx:
            model_inputs, label_ids = templatizer({
                'sub_label': '[X]',
                'obj_label': tokenizer.lama_y,
                'context': ''
            })
        else:
            model_inputs, label_ids = templatizer({
                'sub_label': '[X]',
                'obj_label': tokenizer.lama_y,
            })
        lama_template = model_inputs['input_ids']
        # Instantiate trigger tokens
        lama_template.masked_scatter_(
            mask=model_inputs['trigger_mask'],
            source=best_trigger_ids.cpu())
        # Instantiate label token
        lama_template.masked_scatter_(
            mask=model_inputs['predict_mask'],
            source=label_ids)
        # Print LAMA JSON template
        relation = args.train.parent.stem

        # The following block of code is a bit hacky but whatever, it gets the job done
        if args.use_ctx:
            template = tokenizer.decode(lama_template.squeeze(0)[1:-1]).replace('[SEP] ', '').replace('</s> ', '').replace('[ X ]', '[X]')
        else:
            template = tokenizer.decode(lama_template.squeeze(0)[1:-1]).replace('[ X ]', '[X]')

        out = {
            'relation': args.train.parent.stem,
            'template': template
        }
        print(json.dumps(out))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, required=True, help='Train data path')
    parser.add_argument('--dev', type=Path, required=True, help='Dev data path')
    parser.add_argument('--template', type=str, help='Template string')
    parser.add_argument('--label-map', type=str, default=None, help='JSON object defining label map')

    # LAMA-specific
    parser.add_argument('--tokenize-labels', action='store_true',
                        help='If specified labels are split into word pieces.'
                             'Needed for LAMA probe experiments.')
    parser.add_argument('--filter', action='store_true',
                        help='If specified, filter out special tokens and gold objects.'
                             'Furthermore, tokens starting with capital '
                             'letters will not appear in triggers. Lazy '
                             'approach for removing proper nouns.')
    parser.add_argument('--print-lama', action='store_true',
                        help='Prints best trigger in LAMA format.')

    parser.add_argument('--initial-trigger', nargs='+', type=str, default=None, help='Manual prompt')
    parser.add_argument('--label-field', type=str, default='label',
                        help='Name of the label field')

    parser.add_argument('--bsz', type=int, default=32, help='Batch size')
    parser.add_argument('--eval-size', type=int, default=256, help='Eval size')
    parser.add_argument('--iters', type=int, default=100,
                        help='Number of iterations to run trigger search algorithm')
    parser.add_argument('--accumulation-steps', type=int, default=10)
    parser.add_argument('--model-name', type=str, default='bert-base-cased',
                        help='Model name passed to HuggingFace AutoX classes.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--use-ctx', action='store_true',
                        help='Use context sentences for relation extraction only')
    parser.add_argument('--perturbed', action='store_true',
                        help='Perturbed sentence evaluation of relation extraction: replace each object in dataset with a random other object')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--num-cand', type=int, default=10)
    parser.add_argument('--sentence-size', type=int, default=50)
    parser.add_argument('--class_num', type=int, default=15)

    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)

    run_model(args)
