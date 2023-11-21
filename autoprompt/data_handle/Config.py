class Config(object):
    """
        配置类，保存配置文件
    """
    # batch_size
    batch_size = 1
    # 学习率
    learning_rate = 2e-5
    # epoch数
    num_train_epochs = 40
    # 句子的最大补齐长度
    # sentence_max_len = 2048
    sentence_max_len = 256
    # 结果文件存储位置
    predict_res_file = "/home/dlf/prompt/code/res_files/short_data_res_{}.txt"
    # 词性的类别数量
    # jw
    # class_nums = 18
    # ctb
    # class_nums = 42
    # class_nums = 42
    # ud
    class_nums = 15
    # 计算使用的device
    # device = "cpu"
    device = "cuda:0"
    # k折交叉验证
    kfold = 5
    # 1train9test 10折 train path
    # 1train9test 10折 test path
    # jw dataset
    # train_1_9_path = "/home/dlf/prompt/code/data/split_data/1_9_split/train_{idx}.data"
    # test_1_9_path = "/home/dlf/prompt/code/data/split_data/1_9_split/test_{idx}.data"

    # ctb dataset
    # train_1_9_path = "/home/dlf/prompt/code/data/ctb/split_data/1_9_split/train_{idx}.data"
    # test_1_9_path = "/home/dlf/prompt/code/data/ctb/split_data/1_9_split/test_{idx}.data"

    # 是否断点续训
    resume = False
    # few-shot 划分的数量
    few_shot = [5, 10, 15, 20, 25]
    # few_shot = [5, 10, 15, 20, 25, 50, 75, 100, 200, 500]
    # few_shot = [50, 75, 100, 200, 500]
    # few_shot = [50, 70]
    # few_shot = [5, 10, 15, 20, 25, 50, 75, 100, 200, 500]
    # few_shot = [ 25]
    # 测试集位置
    # jw
    # train_data_path = "/home/dlf/prompt/code/data/split_data/fold/{item}.data"
    # test_data_path = "/home/dlf/prompt/code/data/split_data/pos_seg_test.data"
    # CTB
    # test_data_path = "/home/dlf/prompt/code/data/ctb/split_data/few_shot/one_tentn_test_datas.data"
    # train_data_path = "/home/dlf/prompt/code/data/ctb/split_data/few_shot/fold/{item}.data"
    # test_data_path = "/home/dlf/prompt/code/data/ctb/split_data/few_shot/test_3000.data"
    # UD数据集
    test_data_path = "/home/dlf/prompt/code/data/ud/ud_en/test.data"
    train_data_path = "/home/dlf/prompt/code/data/ud/ud_en/fold/{item}.data"
    # log dir
    log_dir = "ctb_nohmm/"
    # train dataset template

    # train_data_path = "/home/dlf/prompt/code/data/split_data/{item}.data"
    # 截取句子的前n个词组成prompt,超过8个要oom
    # pre_n = 8
    # label
    # jw 数据集
    # special_labels = ["[PLB]", "NR", "NN", "AD", "PN", "OD", "CC", "DEG",
    #                   "SP", "VV", "M", "PU", "CD", "BP", "JJ", "LC", "VC",
    #                   "VA", "VE"]
    # ctb数据集
    # special_labels = ["[PLB]", "NR", "NN", "AD", "PN", "OD", "CC", "DEG",
    #                   "SP", "VV", "M", "PU", "CD", "BP", "JJ", "LC", "VC",
    #                   "VA", "VE",
    #                   "NT-SHORT", "AS-1", "PN", "MSP-2", "NR-SHORT", "DER",
    #                   "URL", "DEC", "FW", "IJ", "NN-SHORT", "BA", "NT", "MSP", "LB",
    #                   "P", "NOI", "VV-2", "ON", "SB", "CS", "ETC", "DT", "AS", "M", "X",
    #                   "DEV"
    #                   ]
    # UD 数据集
    special_labels = ["[PLB]", "PROPN", "SYM", "X", "PRON", "ADJ", "NOUN", "PART", "DET", "CCONJ", "ADP", "VERB", "NUM",
                      "PUNCT", "AUX", "ADV"]
    # 检查点的保存位置
    checkpoint_file = "/home/dlf/prompt/code/src/prompt/pths/ud-ch_{filename}.pth"
    pretrain_models = [
        "/Users/dailinfeng/Desktop/autoprompt_test/model/bert_large_chinese",
        "/home/dlf/prompt/code/model/bert_large_chinese",
        # "/home/dlf/prompt/code/model/medbert",
        #  "/home/dlf/prompt/code/model/bart-large"
    ]
