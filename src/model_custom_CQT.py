from snippets import *
from bert4keras.backend import keras
from bert4keras.layers import GlobalPointer
from keras import backend as K
from keras.engine.topology import Layer

# 基本参数
num_classes = 119
maxlen = 512
stride = 128
batch_size = 16
epochs = 20


class CustomMasking(keras.layers.Layer):
    """自定义mask（主要用于mask掉question部分）
    """

    def compute_mask(self, inputs, mask=None):
        return K.greater(inputs[1], 0.5)

    def call(self, inputs, mask=None):
        return inputs[0]

    def compute_output_shape(self, input_shape):
        return input_shape[0]



def globalpointer_crossentropy(y_true, y_pred):
    """给GlobalPointer设计的交叉熵
    """
    b, l = K.shape(y_pred)[0], K.shape(y_pred)[1]
    # y_true需要重新明确一下shape和dtype
    y_true = K.reshape(y_true, (b, 2))
    y_true = K.cast(y_true, 'int32')
    y_true = y_true[:, 0] * l + y_true[:, 1]
    # 计算交叉熵
    y_pred = K.reshape(y_pred, (b, -1))
    return K.mean(
        K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    )


def globalpointer_accuracy(y_true, y_pred):
    """给GlobalPointer设计的准确率
    """
    b, l = K.shape(y_pred)[0], K.shape(y_pred)[1]
    # y_true需要重新明确一下shape和dtype
    y_true = K.reshape(y_true, (b, 2))
    y_true = K.cast(y_true, 'int32')
    y_true = y_true[:, 0] * l + y_true[:, 1]
    # 计算准确率
    y_pred = K.reshape(y_pred, (b, -1))
    y_pred = K.cast(K.argmax(y_pred, axis=1), 'int32')
    return K.mean(K.cast(K.equal(y_true, y_pred), K.floatx()))

# 利用Keras构造互注意力机制层
class Mutual_MyLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Mutual_MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # 为该层创建一个可训练的权重
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3, input_shape[0][2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(Mutual_MyLayer, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        assert isinstance(x, list)
        x,y = x
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(y, self.kernel[2])
        QK = K.batch_dot(WQ,K.permute_dimensions(WK, [0, 2, 1]))
        QK = QK / (self.output_dim**0.5)
        QK = K.softmax(QK)
        V = K.batch_dot(QK,WV)
        return V

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return (shape_a[0],shape_a[1],self.output_dim)

# 利用Keras构造自注意力机制层
class Self_Attention(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Self_Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        # inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3, input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)

        super(Self_Attention, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])

        print("WQ.shape", WQ.shape)

        print("K.permute_dimensions(WK, [0, 2, 1]).shape", K.permute_dimensions(WK, [0, 2, 1]).shape)

        QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))

        QK = QK / (64 ** 0.5)

        QK = K.softmax(QK)

        print("QK.shape", QK.shape)

        V = K.batch_dot(QK, WV)

        return V

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)


def models_():
    input_triple = keras.layers.Input(shape=(None,))
    input_question = keras.layers.Input(shape=(None,))
    input_context = keras.layers.Input(shape=(None,))

    embeding_triple = keras.layers.Embedding(21128, 768)(input_triple)
    embeding_question = keras.layers.Embedding(21128, 768)(input_question)

    embeding_context = keras.layers.Embedding(21128, 768)(input_context)


    Attention_t2q = Mutual_MyLayer(768)([embeding_triple, embeding_question])
    Attention_c2q = Mutual_MyLayer(768)([embeding_context, embeding_question])
    last_q = keras.layers.Add()([Attention_t2q, Attention_c2q])

    Attention_q2t = Mutual_MyLayer(768)([embeding_question, embeding_triple])
    Attention_c2t = Mutual_MyLayer(768)([embeding_context, embeding_triple])
    last_t = keras.layers.Add()([Attention_q2t,Attention_c2t])

    Attention_q2c = Mutual_MyLayer(768)([embeding_question, embeding_context])
    Attention_t2c = Mutual_MyLayer(768)([embeding_triple, embeding_context])
    last_c = keras.layers.Add()([Attention_q2c,Attention_t2c])

    ks_layer = keras.layers.Multiply()([last_q, last_t, last_c])

    output = base.model.get_layer(last_layer).output
    output = keras.layers.Concatenate(-1)([ks_layer, output])

    output = keras.layers.Lambda(lambda x: x[..., int(x.shape[2])//2:])(output)

    masks_in = keras.layers.Input(shape=(None,))
    output = CustomMasking()([output, masks_in])

    # output = keras.layers.Lambda(lambda x: x, output_shape=lambda s: s)(output)
    # output = Self_Attention(768)(output)
    output = GlobalPointer(
        heads=1,
        head_size=base.attention_head_size,
        use_bias=False,
        kernel_initializer=base.initializer
    )(output)
    output = keras.layers.Lambda(lambda x: x[:, 0])(output)

    model = keras.models.Model(base.model.inputs + [masks_in, input_triple, input_question, input_context], output)
    model.summary()
    # 调用多个gpu
    model = keras.utils.multi_gpu_model(model, gpus=2)
    model.compile(loss=globalpointer_crossentropy, optimizer=optimizer, metrics=[globalpointer_accuracy])
    return model


if __name__=='__main__':
    print(models_())