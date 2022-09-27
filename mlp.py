from tensorflow.keras import callbacks, models, layers, losses, optimizers


class ResBlock(layers.Layer):
    def __init__(self, num_hidden):
        super(ResBlock, self).__init__()
        self.l1 = layers.Dense(num_hidden, activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.l2 = layers.Dense(num_hidden)
        self.relu = layers.ReLU()
        self.add = layers.Add()
        self.bn2 = layers.BatchNormalization()

    def call(self, x):
        fx = self.l1(x)
        fx = self.bn1(fx)
        fx = self.l2(fx)
        out = self.add([x, fx])
        out = self.relu(out)
        out = self.bn2(out)
        return out


def get_model():
    model = models.Sequential([
        layers.Dense(256, activation="relu"),
        ResBlock(256),
        ResBlock(256),
        ResBlock(256),
        ResBlock(256),
        ResBlock(256),
        ResBlock(256),
        # ResBlock(256),
        # ResBlock(256),
        # ResBlock(256),
        layers.Dense(1),
    ])
    return model