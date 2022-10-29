import json
from snippets import *
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from bert4keras.snippets import lowercase_and_normalize
import random

# 基本参数
num_classes = 119
maxlen = 512
stride = 128
batch_size = 16
epochs = 20


def stride_split(i, q, c, a, s, t):
    """滑动窗口分割context
    """
    # 标准转换
    q = lowercase_and_normalize(q)  # 问题
    c = lowercase_and_normalize(c)  # 段落
    a = lowercase_and_normalize(a)  # 答案
    t = lowercase_and_normalize(t)
    e = s + len(a)  # 答案结束位置index
    # 滑窗分割
    results, n = [], 0
    max_c_len = maxlen - len(q) - 3
    while True:
        l, r = n * stride, n * stride + max_c_len
        if l <= s < e <= r:
            results.append((i, q, c[l:r], a, s - l, e - l, t))
        else:
            results.append((i, q, c[l:r], '', -1, -1, t))
        if r >= len(c):  # 如果左边的index大于context的长度，那么直接返回答案，否则，进行下一次的滑窗
            return results
        n += 1


def load_data(filename):
    """加载数据
    格式：[(id, 问题, 篇章, 答案, start, end)]
    """
    data = json.load(open(filename, 'r', encoding='utf-8-sig'))['data']
    D = []
    for d in data:
        for p in d['paragraphs']:
            for qa in p['qas']:
                for a in qa['answers']:
                    t = ''
                    for rand_num in range(1,3): #3个三元组
                        if 'triple_' + str(rand_num) in p['triple']:
                            t += p['triple']['triple_' + str(rand_num)][0]+p['triple']['triple_' + str(rand_num)][1]+p['triple']['triple_' + str(rand_num)][2]+'<SEP>'
                    # if 'triple_' + str(rand_num) in p['triple']:
                    D.extend(
                        stride_split(
                            qa['id'], qa['question'], p['context'], a['text'],
                            a['answer_start'], t
                        )
                    )
                    if a['answer_start'] == -1:
                        break
    return D

# 加载数据集
data_path = '../datasets/'
train_data = load_data(data_path + 'train_triple.json')
valid_data = load_data(data_path + 'dev_triple.json')


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_triple_ids, batch_question_ids, batch_context_ids= [], [], [], [], []
        batch_masks, batch_labels = [], []  # batch_label存放的是一个batch中，答案的起始位置和终止位置
        for is_end, (i, q, c, a, s, e, t) in self.sample(random):
            triple_ids = tokenizer.encode(t)[0]
            question_ids = tokenizer.encode(q)[0]
            context_ids = tokenizer.encode(c)[0]
            token_ids = tokenizer.encode(q)[
                0]  # tokenizer.encode输出文本对应token id和segment id，对q进行编码时，会在q的开头和尾部分别加上[CLS]和[SEP]作为开始和结束标签
            mask = [1] + [0] * len(token_ids[:-1])
            if s == -1:  # s=-1代表的是这个问题在文中找不到答案
                token_ids.extend(tokenizer.encode(c)[0][
                                 1:])  # 将文本c的token_id加入到原始的token_ids中去，因此token_ids为[CLS] q [SEP] c [SEP]所对应的id
                batch_labels.append([0, 0])
            else:  # 问题在文中是有答案的
                cl_ids = tokenizer.encode(c[:s])[0][
                         1:-1]  # c[:s]代表答案前面的文本，[1:-1]代表的是从index=1开始，到倒数第一个截止，因此cl_ids是去掉了[CLS]和[SEP]这两个标签的id,只是文本c的id
                a_ids = tokenizer.encode(c[s:e])[0][1:-1]  # a_ids代表答案的token_id
                cr_ids = tokenizer.encode(c[e:])[0][1:]  # cr_ids代表答案后面的文本的token_id，因此可能出现答案就在最后，cr_ids为空的现象
                start = len(token_ids) + len(cl_ids)  # 由于将问题放在了答案的前面，因此，需要重新计算答案的起始位置
                end = start + len(a_ids) - 1
                assert start <= end
                batch_labels.append([start, end])
                token_ids.extend(cl_ids + a_ids + cr_ids)
            mask.extend([1] * (len(token_ids[:-1]) - len(mask)) + [0])
            batch_triple_ids.append(triple_ids)
            batch_question_ids.append(question_ids)
            batch_context_ids.append(context_ids)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append([0] * len(token_ids))
            batch_masks.append(mask)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_triple_ids = sequence_padding(batch_triple_ids, length=maxlen)
                batch_question_ids = sequence_padding(batch_question_ids, length=maxlen)
                batch_context_ids = sequence_padding(batch_context_ids, length=maxlen)
                batch_token_ids = sequence_padding(batch_token_ids, length=maxlen)
                batch_segment_ids = sequence_padding(batch_segment_ids, length=maxlen)
                batch_masks = sequence_padding(batch_masks, length=maxlen)
                batch_labels = sequence_padding(batch_labels)
                yield [
                          batch_token_ids, batch_segment_ids, batch_masks, batch_triple_ids, batch_question_ids, batch_context_ids
                      ], batch_labels
                batch_token_ids, batch_segment_ids, batch_triple_ids, batch_question_ids, batch_context_ids = [], [], [], [], []
                batch_masks, batch_labels = [], []


train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)

if __name__ == '__main__':
    for i in train_generator:
        print(i)
        break
    for i in valid_generator:
        print(i)
        break
