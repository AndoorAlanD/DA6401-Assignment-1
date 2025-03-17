import math
import random
import numpy as np
import matplotlib.pyplot as plt
import argparse
import wandb

from keras.datasets import mnist
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split

class Feed_Forward_Neural_Network():

  def __init__(self, config):
    self.W = []
    self.b = []

    self.config = config
    self.eta = self.config['lr']

    self.max_ep=self.config['epochs']
    self.b_s=self.config['batch_size']
    self.w_de=self.config['weight_decay']
    self.eps=self.config['epsilon']

    self.X_train = config["X_train"]
    self.y_train = config["y_train"]
    self.X_val = config["X_val"]
    self.y_val = config["y_val"]
    self.X_test = config["X_test"]
    self.y_test = config["y_test"]

    self.l = 2
    self.L = []
    self.L.append(self.X_train.shape[1])

    self.l += self.config['num_layers']
    for i in range(self.l-2):
      self.L.append(self.config['hidden_size'])
    self.L.append(np.max(self.y_train)+1)

    if(config['weight_init']=='random'):
      for i in range(self.l-1):
        self.W.append(np.random.randn(self.L[i+1], self.L[i]))
        self.b.append(np.random.randn(self.L[i+1]))
    else:
      for i in range(self.l-1):
        xav_std = np.sqrt(2 / (self.L[i+1] + self.L[i]))
        self.W.append(np.random.randn(self.L[i+1], self.L[i]) * xav_std)
        self.b.append(np.random.randn(self.L[i+1]))

    if self.config['activation']=='sigmoid':
      self.act =self.sigmoid
      self.act_der =self.sigmoid_der
    elif self.config['activation']=='tanh':
      self.act =self.tanh
      self.act_der =self.tanh_der
    elif self.config['activation']=='ReLU':
      self.act =self.relu
      self.act_der =self.relu_der
    else:
      self.act =self.identity
      self.act_der =self.identity_der

    optimizers = {
        'sgd': self.stochastic,
        'momentum': self.momentum,
        'nag': self.nesterov,
        'rmsprop': self.rmsprop,
        'adam': self.adam,
        'nadam': self.nadam
    }
    self.optimizer = optimizers[self.config['optimizer']](self.X_train, self.y_train)


  def identity(self, x):
    return x

  def identity_der(self, x):
    return np.ones(len(x))

  def sigmoid(self, x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

  def sigmoid_der(self, x):
    sg = self.sigmoid(x)
    return sg * (1 - sg)

  def tanh(self, x):
    return np.tanh(x)

  def tanh_der(self, x):
    new_x = self.tanh(x)
    return 1-(new_x*new_x)

  def relu(self, x):
    return np.maximum(0, x)

  def relu_der(self, x):
    return (x > 0).astype(int)

  def softmax(self, x):
    new_x = np.exp(np.clip(x,-500, 500))
    return  new_x / new_x.sum()

  def log_metrics(self, X1=None, y1=None, X2=None, y2=None, X3=None, y3=None):
    X1 = X1 if X1 is not None else self.X_train
    y1 = y1 if y1 is not None else self.y_train
    X2 = X2 if X2 is not None else self.X_val
    y2 = y2 if y2 is not None else self.y_val
    X3 = X3 if X3 is not None else self.X_test
    y3 = y3 if y3 is not None else self.y_test
    train_acc, train_loss = self.compute_accuracy_and_loss(X1, y1)
    val_acc, val_loss = self.compute_accuracy_and_loss(X2, y2)
    test_acc, test_loss = self.compute_accuracy_and_loss(X3, y3)
    print(test_acc)

    wandb.log({
        'train_accuracy': train_acc,
        'train_loss': train_loss,
        'val_accuracy': val_acc,
        'val_loss': val_loss,
        'test_accuracy': test_acc,
        'test_loss': test_loss
    })

  def log_confusion_matrix(self, X=None, y=None, dataset_name="Test"):
    X = X if X is not None else self.X_test
    y = y if y is not None else self.y_test
    y_true, y_pred = self.compute_predictions(X, y)
    wandb.log({
        f'{dataset_name}_confusion_matrix': wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true,
            preds=y_pred,
            class_names=[str(i) for i in range(np.max(y_true) + 1)]
        )
    })

  def compute_accuracy_and_loss(self, X, y):
    y_true, y_pred = self.compute_predictions(X, y)
    correct_predictions = sum(y_p == y_t for y_p, y_t in zip(y_pred, y_true))
    acc = (correct_predictions / len(y)) * 100.0
    if(self.config['loss'] == 'squared'):
      loss = self.squared_loss(X, y)
    else:
      loss = self.cross_entropy_loss(X, y)
    return acc, loss

  def compute_predictions(self, X, y):
    y_pred_labels = [np.argmax(self.feed_forward(x)[-1]) for x in X]
    return y, y_pred_labels

  def squared_loss(self, X, y):
    sq_loss = 0
    for (x_in, y_true) in zip(X, y):
      _, _, y_pr = self.feed_forward(x_in)
      y_pr[y_true] -= 1
      sq_loss += np.sum(y_pr**2)
    return sq_loss / len(y)

  def cross_entropy_loss(self, X, y):
    loss = 0
    for (x_in, y_true) in zip(X, y):
      _, _, y_out = self.feed_forward(x_in)
      loss -= np.log(y_out[y_true]+1e-10)
    return loss / len(y)

  def feed_forward(self, x, W=None, b=None):
    W = W if W is not None else self.W
    b = b if b is not None else self.b
    pre_a = []
    act_h = [x]

    for i in range(self.l-2):
      pre_a.append(np.dot(W[i], act_h[i]) + b[i])
      act_h.append(self.act(pre_a[-1]))
    pre_a.append(b[-1] + np.dot(W[-1], act_h[-1]))
    y_pred = self.softmax(pre_a[-1])
    return pre_a, act_h, y_pred

  def back_prop(self, a, h, y, y_pred, W=None):
    W = W if W is not None else self.W
    gr_a = []

    if(self.config['loss'] == 'mean_squared_error'):
      y_p = y_pred * (1-y_pred)
      y_pred[y] -= 1
      gr_a = np.array(y_p * y_pred)
    else:
      for i in range(len(y_pred)):
        if(i == y):
          gr_a.append(y_pred[i]-1)
        else:
          gr_a.append(y_pred[i])
      gr_a = np.array(gr_a)

    i = self.l-1
    gr_W, gr_b = [], []
    while i>0:
      gr_W.append(np.outer(gr_a, np.array(h[i-1])))
      gr_b.append(gr_a)
      if(i>1):
        gr_h = np.matmul(W[i-1].T, gr_a)
        gr_a = np.multiply(gr_h, self.act_der(a[i-2]))
      i -= 1

    return gr_W, gr_b

  def d_init(self):
    dw = [np.zeros_like(w) for w in self.W]
    db = [np.zeros_like(bi) for bi in self.b]
    return dw, db

  def momentum(self, X, y):
    be=self.config['momentum']
    uw, ub = self.d_init()
    ind=0

    for ep in range(self.max_ep):
      print(f"Epoch {ep+1}/{self.max_ep}")
      dw, db = self.d_init()

      for (x, y_true) in zip(X, y):
        a, h, y_pre = self.feed_forward(x)
        gr_W, gr_b = self.back_prop(a, h, y_true, y_pre)

        for i in range(self.l-1):
          dw[i] += gr_W[-1-i]
          db[i] += gr_b[-1-i]

        ind += 1
        if(ind == self.b_s):
          for i in range(self.l-1):
            uw[i] =be*uw[i]+self.eta*dw[i]
            ub[i] =be*ub[i]+self.eta*db[i]
            self.W[i] -= uw[i]+self.eta*self.w_de*self.W[i]
            self.b[i] -= ub[i]+self.eta*self.w_de*self.b[i]
          dw, db = self.d_init()
          ind=0

      self.log_metrics()
    self.log_confusion_matrix()

  def nesterov(self, X, y):
    be=self.config['momentum']
    vw, vb = self.d_init()
    ind = 0

    for ep in range(self.max_ep):
      print(f"Epoch {ep+1}/{self.max_ep}")
      dw, db = self.d_init()

      for i in range(self.l-1):
        self.W[i] -= be*vw[i]
        self.b[i] -= be*vb[i]

      for (x, y_true) in zip(X, y):
        a, h, y_pre = self.feed_forward(x)
        gr_W, gr_b = self.back_prop(a, h, y_true, y_pre)

        for i in range(self.l-1):
          dw[i] += gr_W[-1-i]
          db[i] += gr_b[-1-i]

        ind += 1
        if(ind == self.b_s):
          for i in range(self.l-1):
            dw[i] += self.w_de*self.W[i]
            vw[i] = be*vw[i] + self.eta*dw[i]
            vb[i] = be*vb[i] + self.eta*db[i]
            self.W[i] -= self.eta*dw[i]
            self.b[i] -= self.eta*db[i]
          dw, db = self.d_init()
          ind=0

      self.log_metrics()
    self.log_confusion_matrix()

  def stochastic(self, X, y):
    ind=0

    for ep in range(self.max_ep):
      print(f"Epoch {ep+1}/{self.max_ep}")
      dw, db = self.d_init()

      for (x, y_true) in zip(X, y):
        a, h, y_pre = self.feed_forward(x)
        gr_W, gr_b = self.back_prop(a, h, y_true, y_pre)

        for i in range(self.l-1):
          dw[i] += gr_W[-1-i]
          db[i] += gr_b[-1-i]

        ind += 1
        if(ind % self.b_s == 0):
          for i in range(len(self.W)):
            dw[i] += self.w_de*self.W[i]
            self.W[i] -= self.eta*np.array(dw[i])
            self.b[i] -= self.eta*np.array(db[i])
          dw, db = self.d_init()
          ind=0

      self.log_metrics()
    self.log_confusion_matrix()


  def rmsprop(self, X, y):
    be=self.config['beta']
    be=0.9
    vw, vb = self.d_init()
    ind=0

    for ep in range(self.max_ep):
      print(f"Epoch {ep+1}/{self.max_ep}")
      dw, db = self.d_init()

      for (x, y_true) in zip(X, y):
        a, h, y_pre = self.feed_forward(x)
        gr_W, gr_b = self.back_prop(a, h, y_true, y_pre)
        for i in range(self.l-1):
          dw[i] += gr_W[-1-i]
          db[i] += gr_b[-1-i]

        ind += 1
        if(ind == self.b_s):
          for i in range(self.l-1):
            dw[i] += self.w_de*self.W[i]
            vw[i]=be*vw[i]+(1-be)*(dw[i]**2)
            vb[i]=be*vb[i]+(1-be)*(db[i]**2)
            self.W[i] -= self.eta*dw[i]/(np.sqrt(vw[i])+self.eps)
            self.b[i] -= self.eta*db[i]/(np.sqrt(vb[i])+self.eps)
          dw, db = self.d_init()
          ind=0

      self.log_metrics()
    self.log_confusion_matrix()

  def adam(self, X, y):
    b1=self.config['beta1']
    b2=self.config['beta2']
    ind= 0

    mw, mb = self.d_init()
    vw, vb = self.d_init()
    mw_t, mb_t = self.d_init()
    vw_t, vb_t = self.d_init()

    for ep in range(self.max_ep):
      print(f"Epoch {ep+1}/{self.max_ep}")
      dw, db = self.d_init()

      for (x, y_true) in zip(X, y):
        a, h, y_pre = self.feed_forward(x)
        gr_W, gr_b = self.back_prop(a, h, y_true, y_pre)
        for i in range(self.l-1):
            dw[i] += gr_W[-1-i]
            db[i] += gr_b[-1-i]

        ind += 1
        if(ind == self.b_s):
          for i in range(self.l-1):
            dw[i] += self.w_de*self.W[i]
            mw[i]=b1*mw[i]+(1-b1)*dw[i]
            mb[i]=b1*mb[i]+(1-b1)*db[i]
            vw[i]=b2*vw[i]+(1-b2)*dw[i]**2
            vb[i]=b2*vb[i]+(1-b2)*db[i]**2
            mw_t[i]=mw[i]/(1-np.power(b1, ep+1))
            mb_t[i]=mb[i]/(1-np.power(b1, ep+1))
            vw_t[i]=vw[i]/(1-np.power(b2, ep+1))
            vb_t[i]=vb[i]/(1-np.power(b2, ep+1))
            self.W[i] -= self.eta*mw_t[i]/(np.sqrt(vw_t[i])+self.eps)
            self.b[i] -= self.eta*mb_t[i]/(np.sqrt(vb_t[i])+self.eps)
          dw, db = self.d_init()
          ind=0

      self.log_metrics()
    self.log_confusion_matrix()

  def nadam(self, X, y):
    b1=self.config['beta1']
    b2=self.config['beta2']
    mw, mb = self.d_init()
    vw, vb = self.d_init()
    mw_t, mb_t = self.d_init()
    vw_t, vb_t = self.d_init()
    ind= 0

    for ep in range(self.max_ep):
      print(f"Epoch {ep+1}/{self.max_ep}")
      dw, db = self.d_init()

      for (x, y_true) in zip(X, y):
        a, h, y_pre = self.feed_forward(x)
        gr_W, gr_b = self.back_prop(a, h, y_true, y_pre)
        for i in range(self.l-1):
            dw[i] += gr_W[-1-i]
            db[i] += gr_b[-1-i]

        ind += 1
        if(ind == self.b_s):
          for i in range(self.l-1):
            dw[i] += self.w_de*self.W[i]
            mw[i]=b1*mw[i]+(1-b1)*dw[i]
            mb[i]=b1*mb[i]+(1-b1)*db[i]
            vw[i]=b2*vw[i]+(1-b2)*dw[i]**2
            vb[i]=b2*vb[i]+(1-b2)*db[i]**2
            mw_t[i]=mw[i]/(1-np.power(b1, ep+1))
            mb_t[i]=mb[i]/(1-np.power(b1, ep+1))
            vw_t[i]=vw[i]/(1-np.power(b2, ep+1))
            vb_t[i]=vb[i]/(1-np.power(b2, ep+1))
            self.W[i] -= (self.eta/(np.sqrt(vw_t[i])+self.eps)) * (b1*mw_t[i] + (1-b1)*dw[i]/(1-b1**(ep+1)))
            self.b[i] -= (self.eta/(np.sqrt(vb_t[i])+self.eps)) * (b1*mb_t[i] + (1-b1)*db[i]/(1-b1**(ep+1)))
          dw, db = self.d_init()
          ind=0

      self.log_metrics()
    self.log_confusion_matrix()

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Feed Forward Neural Network.")
    
    parser.add_argument("-wp", "--wandb_project", type=str, default="Your wandb.ai project name")
    parser.add_argument("-we", "--wandb_entity", type=str, default="Your wandb.ai account name")
    parser.add_argument("-d", "--dataset", type=str, default="fashion_mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-l", "--loss", type=str, default="cross_entropy", choices=["mean_squared_error", "cross_entropy"])
    parser.add_argument("-o", "--optimizer", type=str, default="nadam",choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-m", "--momentum", type=float, default=0.99)
    parser.add_argument("-beta", "--beta", type=float, default=0.9)
    parser.add_argument("-beta1", "--beta1", type=float, default=0.9)
    parser.add_argument("-beta2", "--beta2", type=float, default=0.99)
    parser.add_argument("-eps", "--epsilon", type=float, default=0.0000001)
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0005)
    parser.add_argument("-w_i", "--weight_init", type=str, default="Xavier", choices=["random", "Xavier"])
    parser.add_argument("-nhl", "--num_layers", type=int, default=5)
    parser.add_argument("-sz", "--hidden_size", type=int, default=64)
    parser.add_argument("-a", "--activation", type=str, default="tanh", choices=["identity", "sigmoid", "tanh", "ReLU"])
    
    return vars(parser.parse_args())

def main():
  args = parse_args()
  
  wandb.init(project=args["wandb_project"], entity=args["wandb_entity"], config=args)
  
  if args["dataset"] == "mnist":
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
  else:
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)

  X_train = X_train.reshape(X_train.shape[0], 28*28) / 255
  X_test = X_test.reshape(X_test.shape[0], 28*28) /255
  X_val = X_val.reshape(X_val.shape[0], 28*28) /255

  config = {
    "lr": args["learning_rate"],
    "epochs": args["epochs"],
    "batch_size": args["batch_size"],
    "loss": args["loss"],
    "optimizer": args["optimizer"],
    "momentum": args["momentum"],
    "beta": args["beta"],
    "beta1": args["beta1"],
    "beta2": args["beta2"],
    "epsilon": args["epsilon"],
    "weight_decay": args["weight_decay"],
    "weight_init": args["weight_init"],
    "num_layers": args["num_layers"],
    "hidden_size": args["hidden_size"],
    "activation": args["activation"],
    "X_train": X_train,
    "y_train": y_train,
    "X_val": X_val,
    "y_val": y_val,
    "X_test": X_test,
    "y_test": y_test
  }

  wandb.run.name = (
        f"{config['epochs']}_{config['optimizer']}_{config['activation']}_{config['loss']}_{config['batch_size']}_{config['num_layers']}_"
        f"{config['lr']}_{config['weight_init']}"
    )

  nn = Feed_Forward_Neural_Network(config)
    
if __name__ == "__main__":
    main()
