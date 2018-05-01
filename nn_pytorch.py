import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# parameters
inputs, hiddens, outputs = 784, 200, 10
learning_rate = 0.01
epochs = 50
batch_size = 20

transformation = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST('mnist/',train=True,transform=transformation, download=False)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST('mnist/',train=False,transform=transformation, download=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class mlp(nn.Module):
    def __init__(self):
        super(mlp, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.hidden_layer = nn.Linear(inputs, hiddens)
        self.output_layer = nn.Linear(hiddens, outputs)

    def forward(self, x):
        out = self.sigmoid(self.hidden_layer(x))
        out = self.sigmoid(self.output_layer(out))
        return out

    def name(self):
        return "mlp"

def train():
    model = mlp()
    loss = nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        avg_error = 0
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.view(-1, inputs))
            # Convert class label to one hot vector 
            one_hot = torch.FloatTensor(labels.size(0), 10).zero_()
            target = one_hot.scatter_(1, labels.view((labels.size(0),1)), 1)            
            target = Variable(target)
            # Compute loss and gradient
            optimizer.zero_grad()
            out = model(images)
            error = loss(out, target)
            error.backward()
            # Apply gradient
            optimizer.step()
            avg_error += error.data[0]
        # Average cost for epoch (over all training dataset samples)
        avg_error /= train_loader.dataset.train_data.shape[0]
        print ("Epoch [%d/%d], error: %.4f" %(epoch+1, epochs, avg_error))
    # Save model to file
    torch.save(model.state_dict(), 'model.pkl')

def predict():
    model = mlp()
    model.load_state_dict(torch.load('model.pkl'))
    correct, total = 0, 0
    for images, labels in test_loader:
        images = Variable(images.view(-1, inputs))
        out = model(images)
        _, predicted = torch.max(out.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('accuracy: %0.2f %%' % (100.0 * correct / total))   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, default='predict')
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.action == "predict":
        predict()
    if FLAGS.action == "train":
        train()