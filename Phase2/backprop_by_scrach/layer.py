import numpy as np


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size


class Relu:
    """
    フォワード処理は
        xが1以上ならその値をそのまま後方に伝搬する
        xが0以下ならその値を全て0に置き換えて後方に伝搬する
    
    バックワード処理は
        self.maskの行列とdoutの行列の成分を比較し
            self.maskがFalseだった行列の成分と同じ場所のdoutの値はそのまま前方に伝搬
            self.maskがTrueだった行列の成分と同じ場所のdoutの値は0に置き換えて前方に伝搬
    """
    def __init__(self):
        self.mask = None

    def forward(self, x):
        """
        ex: 
            x = np.array([1, -3, 2, -4])
            self.mask = [False  True False  True]
            out[self.mask] = [-3, -4]
            out = [1, 0, 2, 0]
        """
        self.mask = (x<=0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        """
        ex:
            dout = np.array([100, 100, 200, 200])
            dout[self.mask] = [100 200]
            dx = [100   0 200   0]
        """
        dout[self.mask] = 0
        dx = dout
        return dx

sample = Relu()
out = sample.forward(np.array([1, -3, 2, -4]))
dout = sample.backward(np.array([100, 100, 200, 200]))


class Sigmoid:
    """
    dLdy === dout: 後方から前方に伝わってきた値
    self.out: 前方に伝搬した値
    
    フォワード処理はxを
        "1 / (1 + np.exp(-x))"に入れて前方に渡す
    
    バックワード処理は
        doutにシグモイド関数の微分(y) * (1-y)を掛けた値を前方に伝搬する
    """
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        # dLdy * y * (1 - y)
        dx = dout *  self.out * (1.0 - self.out)


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 損失
        self.y = None # softmaxの出力
        self.t = None # 教師データ (one hot vector)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx