import argparse


def get_train_config():
    parse = argparse.ArgumentParser(description='common supervised learning config')

    # 项目配置参数
    parse.add_argument('-data-set',type=str,default='THP-small',help='数据集')
    parse.add_argument('-learn-name', type=str, default='new_train_04_Model', help='本次训练的名称')
    parse.add_argument('-max-len', type=int, default=90+2, help='max length of input sequences')
    parse.add_argument('-path-meta-data', type=str, default='../data/meta_data/', help='保存字典的位置')
    parse.add_argument('-save-best', type=bool, default=True, help='当得到更好的准确度是否要保存')
    parse.add_argument('-threshold', type=float, default=0.86, help='准确率阈值')
    parse.add_argument('-cuda', type=bool, default=True)
    parse.add_argument('-device', type=int, default=2)

    # 训练参数
    parse.add_argument('-lr', type=float, default=0.00025, help='学习率')
    parse.add_argument('-reg', type=float, default=0.0075, help='正则化lambda')
    parse.add_argument('-batch-size', type=int, default=32, help='一个batch中有多少个sample')
    parse.add_argument('-epoch', type=int, default=80, help='迭代次数')
    parse.add_argument('-k-fold', type=int, default=-1, help='k折交叉验证,-1代表只使用train-test方式')
    parse.add_argument('-num-class', type=int, default=2, help='类别数量')
    parse.add_argument('-train-way', type=int, default=2, help='类别数量')
    parse.add_argument('-interval-log', type=int, default=20, help='经过多少batch记录一次训练状态')
    parse.add_argument('-interval-valid', type=int, default=1, help='经过多少epoch对交叉验证集进行测试')
    parse.add_argument('-interval-T5_BERT_Model', type=int, default=1, help='经过多少epoch对测试集进行测试')

    # 模型参数
    # 通用
    parse.add_argument('-dim-embedding', type=int, default=128, help='词（残基）向量的嵌入维度')
    parse.add_argument('-num-layer', type=int, default=1, help='Transformer的Encoder模块的堆叠层数')
    parse.add_argument('-dropout', type=float, default=0.4, help='dropout率')
    parse.add_argument('-static', type=bool, default=False, help='嵌入是否冻结')

    # Transformer
    parse.add_argument('-num-head', type=int, default=16, help='多头注意力机制的头数')
    parse.add_argument('-dim-feedforward', type=int, default=32, help='词（残基）向量的嵌入维度')
    parse.add_argument('-dim-k', type=int, default=32, help='k/q向量的嵌入维度')
    parse.add_argument('-dim-v', type=int, default=32, help='v向量的嵌入维度')

    # TextCNN
    parse.add_argument('-dim-embedding-cnn', type=int, default=128, help='词（残基）向量的嵌入维度')
    parse.add_argument('-num-filter', type=int, default=32, help='卷积核的数量')
    parse.add_argument('-filter-sizes', type=str, default='1,2,4,8,16,24', help='卷积核的尺寸')

    config = parse.parse_args()
    return config
