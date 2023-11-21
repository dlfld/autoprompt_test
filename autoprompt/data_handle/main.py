from typing import List

import joblib
from tqdm.auto import tqdm
from transformers import BertConfig, AutoTokenizer, AutoModelForMaskedLM

from data.ud.Config import Config


def generate_prompt(sentence: str, word: str, pre_part_of_speech: str, pre_word: str, part_of_speech: str) -> str:
    """
        生成一个prompt句子，目前提示模板的构造使用的是这个方法
    :param sentence: 句子
    :param word: 当前的词语
    :param pre_part_of_speech: 前一个词语的词性
    :param pre_word: 前一个词语
    :param part_of_speech: 当前词语的词性
    """
    template = "在句子“{sentence}”中，词语“{word}”的前文如果是由“{pre_part_of_speech}”词性的词语“{pre_word}”来修饰，那么词语“{word}”的词性是“[MASK]”→ {part_of_speech}"
    template2 = "句子“{sentence}”中，“{word}”是由“{pre_part_of_speech}”词性的词语“{pre_word}”来修饰，“{word}”的词性是“[MASK]”→ {part_of_speech}"
    template3 = f"在句子“{sentence}”中，[T] [T] [T] “{pre_word}”[T] [T] [T] [T]“{word}” [T] [T] [T] “[MASK]”→ {part_of_speech}"
    # return template2.format(sentence=sentence, word=word, pre_part_of_speech=pre_part_of_speech, pre_word=pre_word,part_of_speech=part_of_speech)
    return template3.format(sentence=sentence, pre_word=pre_word, word=word, part_of_speech=part_of_speech)


def build_a_list_of_prompts_not_split(datas: List[List[str]], is_train_data: bool) -> List[List[str]]:
    # 数据集
    """
        生成不按照具体划分的数据集
        :param datas: 输入是标准的数据集
        :param is_train_data: 是否是加载train数据，train 和test的数据加载有一定的偏差
        具体的偏差是：如果是训练数据的话，前文词性用[PLB]来做占位符号
        :return  [
                    [data,label]
                ] 输出
    """
    dataset = []

    # 遍历整个数据集
    for item in datas:
        # 进行条数据生成
        sentence = item[0].split("/")
        label = item[1].split("/")

        for index, word in enumerate(zip(sentence, label)):
            # 当前句子 '脉/弦/大' -> 脉弦大
            cur_sentence = item[0].replace("/", "")
            # 前文词性 前文词性， 如果是训练数据的话，前文词性用[PLB]来做占位符号
            pre_part_of_speech = "[CLS]" if index == 0 else label[index - 1] if is_train_data else "[PLB]"
            # 前文词语
            pre_word = "[CLS]" if index == 0 else sentence[index - 1]
            # 当前词语
            cur_word = word[0]
            # 当前词性
            cur_part_of_speech = label[index]
            # 生成输入模型的pair
            prompt = generate_prompt(sentence=cur_sentence, word=cur_word, pre_part_of_speech=pre_part_of_speech,
                                     pre_word=pre_word, part_of_speech=cur_part_of_speech)
            # logddd.log(prompt)
            dataset.append([prompt.split("→")[0], prompt.split("→")[1].strip()])

    return dataset


def load_instance_data(standard_data: List[List[str]], tokenizer, Config, is_train_data: bool):
    """
      加载训练用的数据集
      :param standard_data: 标准格式的数据
      :param tokenizer:tokenizer对象
      :param Config: 模型配置类
      :param is_train_data: 是否是加载train数据，train 和test的数据加载有一定的偏差
    """
    # 每一条数据转换成的prompt列表 [[prompts],[prompts],...]
    instance_data = []

    for data in tqdm(standard_data, desc="load_instance_data:"):
        # 将一条数据转换成一系列的prompts
        prompts = build_a_list_of_prompts_not_split([data], is_train_data)
        # 遍历每一个prompt，将其转换为可以直接输入模型的数据
        prompt_texts = []
        prompt_labels = []
        for prompt in prompts:
            if len(str(prompt[1]).strip().replace("\n", "")) > 0:
                prompt_texts.append(prompt[0])
                prompt_labels.append(prompt[1])
        if len(prompt_texts) == 0:
            continue
        instance_data.append([prompt_texts, prompt_labels])

    return instance_data

    #     下面的代码是生成token的，目前只需要生成句子
    #     result = tokenizer(prompt_texts, return_tensors="pt", padding="max_length", max_length=Config.sentence_max_len)
    #     result["labels"] = [tokenizer.convert_tokens_to_ids(str(label).strip().replace("\n", "")) for label in
    #                         prompt_labels]
    #     # logddd.log(len(result["input_ids"]) == len(result["labels"]))
    #
    #     # 保存当前列的label
    #     label = copy.deepcopy(result["labels"])
    #     # 复制当前label过去
    #     labels = copy.deepcopy(result["input_ids"])
    #     for index, sentence_words in enumerate(result["input_ids"]):
    #         # 遍历当前句子的每一个word
    #         for word_index, word in enumerate(sentence_words):
    #             # 当前word 的id如果是等于mask的id
    #             if word == tokenizer.mask_token_id:
    #                 # 第index个句子，第word_index个词的id为 = 第index个label
    #                 # result["labels"][index][word_index] = label[index]
    #                 labels[index][word_index] = label[index]
    #             else:
    #                 labels[index][word_index] = -100
    #
    #     result["labels"] = labels
    #     # 删除不需要的key token_type_ids
    #     # del result["token_type_ids"]
    #
    #     instance_data.append(result)
    #
    # return instance_data


def load_plm(model_checkpoint):
    """
        加载预训练语言模型
    """
    # 获取模型配置
    model_config = BertConfig.from_pretrained(model_checkpoint)
    # 修改配置
    model_config.output_hidden_states = True

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenizer.add_special_tokens({'additional_special_tokens': Config.special_labels})
    if "bart" in model_checkpoint:
        from transformers import BartForConditionalGeneration
        model = BartForConditionalGeneration.from_pretrained(model_checkpoint, config=model_config)
    else:
        model = AutoModelForMaskedLM.from_pretrained(model_checkpoint, config=model_config)
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def wirte_jsonl(file_jsonl_path, datas):
    """
        一次性写入所有jsonl数据
    """
    import jsonlines
    # 写jsonl文件
    with jsonlines.open(file_jsonl_path, mode="a") as file_jsonl:
        for data in datas:
            file_jsonl.write(data)

def hangdel_datas(file_path):
    origin_data = joblib.load(file_path)
    _, tokenizer = load_plm("/Users/dailinfeng/Desktop/autoprompt_test/model/bert_large_chinese")

if __name__ == '__main__':
    test_data = joblib.load("test.data")[:1]
    print(test_data)
    model, tokenizer = load_plm("/Users/dailinfeng/Desktop/autoprompt_test/model/bert_large_chinese")
    train_data_instances = load_instance_data(test_data, tokenizer, Config, is_train_data=True)
