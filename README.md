# DA6401-Assignment-1
Assignment 1 of DA6401, Introduction to Deep Learning

Link of Wandb.ai Report: https://wandb.ai/alandandoor-iit-madras/DL_A1/reports/DA6401-Assignment-1--VmlldzoxMTY1MzU3Nw?accessToken=59e8f9bcl90mf52dxibcwi5vy5g581zygthp3r1lpfc7j0k8vl1k4ezksx7xowqj

Link of GitHub: https://github.com/AndoorAlanD/DA6401-Assignment-1

Codes
-----
train.py  :- This code contains the whole code for the assignment and follow the code specifications mentioned.

train.ipynb :- This is the same code as 'train.py' but in google colab format. You can see how to run 'train.py' inside this file.

Question_1.ipynb :- This file contains the code for the question 1 in report. Here we are using the Fashion-MNIST dataset.

Question_2.ipynb :- This file contains the code for the question 2 in report. Here I have created a 'Feed_Forward_Neural_Network' class with forward and backward pass. I have also put a gradient descent optimization function in it. We can easily change the number of hidden layers and the number of neurons in hidden layers by changing the values inside the 'config'. It will output probability distribution over the 10 classes for the specified input. 

Question_3-8.ipynb :- This file contains the code for the questions 3 to 8 in report. The basic structure of code is same as 'Question_2.ipynb' but with all the optimisation functions from question 3 and all the hyperparameters from question 4. Here we make use of the sweep functionality of wandb.ai to iterate through multiple combinations of hyperparameters. Each run of the sweep is given proper naming. Here we are using Fashion-MNIST dataset and then splitting the train dataset into training and validation datasets at a ratio 9:1.

Question_10.ipynb :- This file contains the code for the question 10 in report. The code is same as the code 'Question_3-8.ipynb', the difference is that here we are using the MNIST dataset (not Fashion-MNIST) and instead of calling a sweep we are just calling 3 runs for the set of hyperparameters that gave best accuracy we I called the sweep in early codes.
