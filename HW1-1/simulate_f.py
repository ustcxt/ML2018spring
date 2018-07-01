import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt
from model import Deep
from model import Shallow
from train_op import train_op
from test import test_op
import os
EPOCH = 50000
# Generate data_set
def objective_function(x):
	return math.sin(5*x*math.pi)/(5*x*math.pi)
train_x = np.random.normal(loc=0.0,scale=1.0,size=(2000,1))
train_y = np.asarray([objective_function(entry) for entry in train_x],dtype=np.float)

# add Guassion noisy
#train_y = train_y+0.1*np.random.normal(size=np.shape(train_x))
#print(np.shape(train_x))
#print(0.9*np.random.normal(size=np.shape(train_x)))

if not os.path.exists("./results"):
	os.makedirs('./results')

print(train_x)
print(train_y)

fig = plt.figure()
ax = fig.add_subplot(221)
ax.set_title('train_set')
plt.xlim(0.0,1.0)
plt.xlabel('x')
plt.ylabel('y')
ax.scatter(train_x,train_y,c='r',marker='.',label = 'train_set')


shallow_model = Shallow()
deep_model = Deep()
shallow_dir = './Shallow'
deep_dir = './Deep'
loss_record = {}
shallow_name,loss_record_shallow=train_op(train_x,train_y,shallow_model,shallow_dir)
loss_record['shallow']=loss_record_shallow
deep_name,loss_record_deep=train_op(train_x,train_y,deep_model,deep_dir)
loss_record['deep']=loss_record_deep

x_axis = np.arange(EPOCH)+1
print(len(x_axis))
print(len(loss_record_shallow))


training_range = np.arange(-1.0,1.0,0.01)
traning_array = np.reshape(training_range,(len(training_range),1))
gt = [objective_function(entry) for entry in traning_array]
print(traning_array)
ax1 = fig.add_subplot(222)
ax1.set_title("gt")
plt.xlim(0.0,1.0)
plt.xlabel('x')
plt.ylabel('y') 
plt.plot(traning_array,gt,'g',label = 'ground-truth')

y_pred_shallow=test_op(traning_array,shallow_model,shallow_dir,shallow_name)
y_pred_deep = test_op(traning_array,deep_model,deep_dir,deep_name)
ax2 = fig.add_subplot(223)
ax2.set_title('test_shallow_model')
plt.xlim(0.0,1.0)
plt.xlabel('x')
plt.ylabel('y')
plt.plot(traning_array,y_pred_shallow,'r',label='Shallow-net prediction')
ax3 = fig.add_subplot(224)
ax3.set_title('test_deep_model')
plt.xlim(0.0,1.0)
plt.xlabel('x')
plt.ylabel('y')
plt.plot(traning_array,y_pred_deep,'b',label= 'Deep-net prediction')

plt.savefig("./results/result_1.png")

fig2 = plt.figure()
bx = fig2.add_subplot(111)
bx.set_title('results')
plt.xlim(0.0,1.0)
plt.plot(traning_array,gt,'g',label = 'ground-truth')
plt.plot(traning_array,y_pred_shallow,'r',label='Shallow-net prediction')
plt.plot(traning_array,y_pred_deep,'b',label= 'Deep-net prediction')
plt.legend()
plt.savefig('./results/result_2.png')

fig3 = plt.figure()
cx = fig3.add_subplot(111)
cx.set_title('Compare loss of shallow/deep')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(x_axis,loss_record['shallow'],label='Shallow model')
plt.plot(x_axis,loss_record['deep'],label='Deep model')
plt.legend()
plt.savefig('./results/Compare_loss.png')
plt.show()
	

