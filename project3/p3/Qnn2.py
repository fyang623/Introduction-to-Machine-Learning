
from nn import NN, Relu, Linear, SquaredLoss
from utils import data_loader, acc, save_plot, loadMNIST, onehot
from run_nn2 import *
from numpy import *
from matplotlib.pyplot import *
import util
import dr
import matplotlib.pyplot as plt



#model = NN(Relu(), SquaredLoss(), hidden_layers=[128,128])
#model.print_model()
x_train, label_train = loadMNIST('data/train-images.idx3-ubyte', 'data/train-labels.idx1-ubyte')
x_test, label_test = loadMNIST('data/t10k-images.idx3-ubyte', 'data/t10k-labels.idx1-ubyte')
y_train = onehot(label_train)
y_test = onehot(label_test) 
(P,Z,evals) = dr.pca(x_train.transpose(), 784)

x_train_new=[]
for i in range(6):
	x_train_new.append(dot(x_train.transpose(), Z[:,i]).reshape(60000,1))
#print(shape(x_train_new))
x_train_new = concatenate(x_train_new, axis=1).transpose()
#print(x_train_new.shape)

x_test_new=[]
for i in range(6):
	x_test_new.append(dot(x_test.transpose(), Z[:,i]).reshape(10000,1))
#print(shape(x_test_new))
x_test_new = concatenate(x_test_new, axis=1).transpose()
#print(x_test_new.shape)

Ks=[]
train_times=[]
test_times=[]
accs=[]

for K in [3, 4, 5, 6, 7, 8, 9, 10, 15, 30, 60, 120, 240, 480, 600, 784]:
	print('K = '+str(K))
	print(x_train.shape)
	print(x_test.shape)

	print(x_train_new[0:K,:].shape)


	model = NN(Relu(), SquaredLoss(), hidden_layers=[256, 256], input_d=K, output_d=10)
	model.print_model()

	lr = 1e-2
	max_epoch = 20
	batch_size = 128
	training_data = {"X":x_train_new[0:K,:], "Y":y_train}
	dev_data = {"X":x_test_new[0:K,:], "Y":y_test}

	train_time,test_time,acc = train(model, training_data, dev_data, lr, batch_size, max_epoch)
	print((train_time,test_time,acc))
	Ks.append(K)
	train_times.append(train_time)
	test_times.append(test_time)
	accs.append(acc)

print(Ks)
print(train_times)
print(test_times)
print(accs)
'''
Ks=[3, 4, 5, 6, 7, 8, 9, 10, 15, 30, 60, 120, 240, 480, 600, 784]
train_times=[18.74735426902771, 18.801438570022583, 18.822139024734497, 18.81931781768799, 18.506865739822388, 17.57029914855957, 18.008158922195435, 18.40490961074829, 18.173954725265503, 19.952720642089844, 21.19170618057251, 24.506177186965942, 28.566027641296387, 37.71127200126648, 42.53033947944641, 48.718310832977295]
test_times=[2.2796733379364014, 2.6452300548553467,  2.287336826324463, 2.7969236373901367, 2.299431324005127, 2.3354525566101074, 2.189162015914917, 2.2348473072052, 2.2607719898223877, 2.4500458240509033, 2.5265352725982666, 2.839808464050293, 3.4424517154693604, 4.222909450531006, 4.710484027862549, 3.420621395111084]
accs=[0.5142, 0.6249, 0.7412, 0.8125, 0.8308, 0.8383, 0.8677, 0.8975, 0.9073, 0.9356, 0.9352, 0.9374, 0.9407, 0.9431, 0.9474, 0.9471]
'''
#plt.subplot(i)
#i += 1
plt.xlabel('number of dimensions')
plt.ylabel('training accuracy')
plt.plot(Ks, accs, label='accuracy')
#plt.text('ff')
plt.xscale('log')
plt.grid(True)

# Now add the legend with some customizations.
legend = plt.legend(loc='lower right', shadow=True)

# The frame is matplotlib.patches.Rectangle instance surrounding the legend.
frame = legend.get_frame()
frame.set_facecolor('0.90')

# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('large')

for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width

plt.title('acc vs. dim')
plt.show()

plt.xlabel('number of dimensions')
plt.ylabel('time(s)')
plt.plot(Ks, train_times, label='training time')
plt.plot(Ks, test_times, label='test time')
#plt.text('ff')
plt.xscale('log')
plt.grid(True)

# Now add the legend with some customizations.
legend = plt.legend(loc='lower right', shadow=True)

# The frame is matplotlib.patches.Rectangle instance surrounding the legend.
frame = legend.get_frame()
frame.set_facecolor('0.90')

# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('large')

for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width

plt.title('time vs. dim')
plt.show()

