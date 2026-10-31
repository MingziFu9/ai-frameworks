# 记录每一轮训练和验证成绩的小工具（照抄即可，不要求看懂）
import paddle


class History(paddle.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.h = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}

    def on_epoch_end(self, epoch, logs=None):        # 每轮训练结束时记下训练成绩
        self.h["loss"].append(float(logs["loss"][0]))
        self.h["acc"].append(float(logs["acc"]))

    def on_eval_end(self, logs=None):                # 每次验证结束时记下验证成绩
        self.h["val_loss"].append(float(logs["loss"][0]))
        self.h["val_acc"].append(float(logs["acc"]))
