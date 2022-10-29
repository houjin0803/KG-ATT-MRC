import model_custom_CQT
import dataloading_custom_CQT
import keras
import json
import numpy as np
from tqdm import tqdm
from itertools import groupby
from snippets import *

epochs = 20

# 调用自定义模型
model = model_custom_CQT.models_()
# 转换数据集
train_generator = dataloading_custom_CQT.train_generator
valid_generator = dataloading_custom_CQT.valid_generator
vaild_data = dataloading_custom_CQT.valid_data


class Evaluator(keras.callbacks.Callback):
    """保存验证集acc最好的模型
    """

    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = self.evaluate(vaild_data, valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('../weights/KG_ATT_MRC_cqt_webqa_4triple.weights')
        print(
            u'val_acc: %.5f, best_val_acc: %.5f\n' %
            (val_acc, self.best_val_acc)
        )
        print('logs:', logs)

    def evaluate(self, data, generator):
        Y_scores = np.empty((0, 1))
        Y_start_end = np.empty((0, 2), dtype=int)
        Y_true = np.empty((0, 2), dtype=int)
        for x_true, y_true in tqdm(generator, ncols=0):
            y_pred = model.predict(x_true)
            y_pred[:, 0] -= np.inf
            y_pred[:, :, 0] -= np.inf
            y_pred = y_pred.reshape((x_true[0].shape[0], -1))
            y_start_end = y_pred.argmax(axis=1)[:, None]
            y_scores = np.take_along_axis(y_pred, y_start_end, axis=1)
            y_start = y_start_end // x_true[0].shape[1]
            y_end = y_start_end % x_true[0].shape[1]
            y_start_end = np.concatenate([y_start, y_end], axis=1)
            Y_scores = np.concatenate([Y_scores, y_scores], axis=0)
            Y_start_end = np.concatenate([Y_start_end, y_start_end], axis=0)
            Y_true = np.concatenate([Y_true, y_true], axis=0)

        total, right, n = 0., 0., 0
        for k, g in groupby(data, key=lambda d: d[0]):  # 按qid分组
            g = len(list(g))
            i = Y_scores[n:n + g].argmax() + n  # 取组内最高分答案
            y_true, y_pred = Y_true[i], Y_start_end[i]
            if (y_pred == y_true).all():
                right += 1
            total += 1
            n += g

        return right / total


def group_data(test_data):
    """
    源代码groupby函数实现有问题，此处实现数据的分组功能，因为对原文进行了分割，即使用了滑动窗口，因此，可能出现一个文本，分成了多份，出现同样的问题，
    以及不同的答案或者有的问题根本没有答案，所以需要找出评分最高的答案
    """
    # 存放qid，包括重复的qid也存放进去
    question_id = []
    for item in test_data:
        question_id.append(item[0])  # item[0]代表的是question_id
    '''
    results的存储格式为：
    {
        'qid': 此qid出现的次数
    }
    '''
    results = {}
    for id in question_id:
        if id not in results:
            results[id] = 1
        else:
            results[id] += 1
    return results


def test_predict(in_file, out_file):
    """输出测试结果到文件
    结果文件可以提交到 https://www.cluebenchmarks.com 评测。
    """
    test_data = dataloading_custom_CQT.load_data(in_file)
    test_generator = dataloading_custom_CQT.data_generator(test_data)

    Y_scores = np.empty((0, 1))
    Y_start_end = np.empty((0, 2), dtype=int)
    for x_true, _ in tqdm(test_generator, ncols=0):
        y_pred = model.predict(x_true)
        y_pred[:, 0] -= np.inf
        y_pred[:, :, 0] -= np.inf
        y_pred = y_pred.reshape((x_true[0].shape[0], -1))
        y_start_end = y_pred.argmax(axis=1)[:, None]
        y_scores = np.take_along_axis(y_pred, y_start_end, axis=1)
        y_start = y_start_end // x_true[0].shape[1]
        y_end = y_start_end % x_true[0].shape[1]
        y_start_end = np.concatenate([y_start, y_end], axis=1)
        Y_scores = np.concatenate([Y_scores, y_scores], axis=0)
        Y_start_end = np.concatenate([Y_start_end, y_start_end], axis=0)

    results, n = {}, 0
    # for k, g in groupby(test_data, key=lambda d: d[0]):  # 按qid分组
    qid_array = group_data(test_data)
    for qid, qid_count in qid_array.items():
        g = qid_count
        i = Y_scores[n:n + g].argmax() + n  # 取组内最高分答案;argmax(a,axis)是指返回axis轴方向最大值的索引
        start, end = Y_start_end[i]
        q, c = test_data[i][1:3]
        q_tokens = tokenizer.tokenize(q)
        c_tokens = tokenizer.tokenize(c)[1:-1]
        mapping = tokenizer.rematch(c, c_tokens)  # 重匹配，直接在context取片段
        start, end = start - len(q_tokens), end - len(q_tokens)
        try:
            results[qid] = c[mapping[start][0]:mapping[end][-1] + 1]
            n += g
        except:
            print(qid)
            print(start, end)
            print(mapping)
            os._exit(0)
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    evaluator = Evaluator()
    history = model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )
    model.load_weights('../weights/KG_ATT_MRC_cqt_webqa_4triple.weights')
    test_predict(
        in_file=data_path + 'webqa_test_triple.json',
        out_file='../results/KG_ATT_MRC_cqt_webqa_4triple_results.json'
    )
