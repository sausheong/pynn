import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn
from mxnet.gluon.data import vision

warnings.simplefilter(action='ignore', category=DeprecationWarning)

# parameters
inputs, hiddens, outputs = 784, 200, 10
learning_rate = 0.01
epochs = 50
batch_size = 20

ctx = mx.cpu()

def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)
    
train_data = mx.gluon.data.DataLoader(vision.MNIST(train=True, transform=transform), batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(vision.MNIST(train=False, transform=transform), batch_size, shuffle=False)

def mlp():
    model = nn.Sequential()
    with model.name_scope():
        model.add(nn.Dense(hiddens, activation="sigmoid"))
        model.add(nn.Dense(outputs, activation="sigmoid"))
        dist = mx.init.Uniform(1/np.sqrt(float(hiddens)))
        model.collect_params().initialize(dist, ctx=ctx) 
    return model

def train():
    model = mlp()   
    loss = gluon.loss.L2Loss()
    optimizer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': learning_rate})

    for e in range(epochs):
        cumulative_error = 0
        for i, (data, labels) in enumerate(train_data):
            data = data.as_in_context(ctx).reshape((-1, inputs))
            labels = nd.one_hot(labels, 10, 1, 0).as_in_context(ctx)
            with autograd.record():
                output = model(data)
                error = loss(output, labels)
            error.backward()
            optimizer.step(data.shape[0])
            cumulative_error += nd.sum(error).asscalar()
        print("Epoch [%d/%d]: error: %.4f" % (e+1, epochs, cumulative_error/len(train_data)))    
    model.save_params("mxnet.model")

def predict():
    model = mlp()
    model.load_params("mxnet.model", ctx)
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(test_data):
        data = data.as_in_context(ctx).reshape((-1, inputs))
        label = label.as_in_context(ctx)
        output = model(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    print("accuracy: %.2f%%" % (acc.get()[1] * 100))

if __name__ == "__main__":   
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, default='predict' )
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.action == "predict":
        predict()
    if FLAGS.action == "train":
        train()