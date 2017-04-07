import cPickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc


class LinearClassifier:

	def __init__(self, W, b, learning_rate, reg_strength):
		self.name = "Abstract"
		self.W = W
		self.b = b
		self.learning_rate = learning_rate
		self.reg_strength = reg_strength

	def evaluate_classifier(self, X):
		raise NotImplementedError("Should have implemented this")

	def compute_cost(self, X, y):
		raise NotImplementedError("Should have implemented this")

	def compute_accuracy(self, X, y):
		scores = self.evaluate_classifier(X)
		acc = np.mean(np.argmax(scores, axis=1) == y)
		return acc

	def compute_gradients(self, X, y, scores):
		raise NotImplementedError("Should have implemented this")

	def gradient_check(self):
		raise NotImplementedError("Should have implemented this")

	def mini_batch_gradient_descent(self, X, y, n_batch):
		for j in range(X.shape[0] / n_batch): 
			start = j * n_batch 
			end = (j + 1) * n_batch - 1	
			batch_X = X[start:end, ] 
			batch_y = y[start:end, ]
			scores = self.evaluate_classifier(batch_X)
			grad_W, grad_b = self.compute_gradients(batch_X, batch_y, scores)
			self.W -= self.learning_rate * grad_W
			self.b -= self.learning_rate * grad_b 

	def vanilla_gradient_descent(self, train_data, train_labels, valid_data,
			valid_labels, n_epochs, n_batch):
		# Losses
		train_loss = np.zeros(n_epochs)
		valid_loss = np.zeros(n_epochs) 
		# Accuracies
		train_acc = np.zeros(n_epochs)
		valid_acc = np.zeros(n_epochs)
		# Train
		for ep in range(n_epochs):
			#print "Epoch ", ep + 1
			self.mini_batch_gradient_descent(train_data, train_labels, n_batch)
			train_loss[ep] = self.compute_cost(train_data, train_labels)
			valid_loss[ep] = self.compute_cost(valid_data, valid_labels)
			train_acc[ep] = self.compute_accuracy(train_data, train_labels)
			valid_acc[ep] = self.compute_accuracy(valid_data, valid_labels)
		return train_loss, valid_loss, train_acc, valid_acc

	def plot_loss(self, train_loss, valid_loss, n_epochs):
		epochs = np.arange(1, n_epochs + 1)
		plt.plot(epochs, train_loss, label="training loss")
		plt.plot(epochs, valid_loss, label="validation loss")
		plt.xlabel("Epochs")
		plt.ylabel("Loss")
		plt.legend(loc='upper right', shadow=True)
		plt.show()

	def plot_accuracy(self, train_acc, valid_acc, n_epochs):
		epochs = np.arange(1, n_epochs + 1)
		plt.plot(epochs, train_acc, label="training accuracy")
		plt.plot(epochs, valid_acc, label="validation accuracy")
		plt.xlabel("Epochs")
		plt.ylabel("Accuracy")
		plt.legend(loc='lower right', shadow=True)
		plt.show()

	def plot_loss_acc(self, train_loss, valid_loss, train_acc, valid_acc, n_epochs):
		epochs = np.arange(1, n_epochs + 1)
		fig, axes = plt.subplots(1, 2, figsize=(15, 5))
		# Loss
		axes[0].plot(epochs, train_loss, label="training loss")
		axes[0].plot(epochs, valid_loss, label="validation loss")
		axes[0].set_xlabel("Epochs")
		axes[0].set_ylabel("Loss")
		axes[0].legend(loc='upper right', shadow=True)
		# Accuracy
		axes[1].plot(epochs, train_acc, label="training accuracy")
		axes[1].plot(epochs, valid_acc, label="validation accuracy")
		axes[1].set_xlabel("Epochs")
		axes[1].set_ylabel("Accuracy")
		axes[1].legend(loc='lower right', shadow=True)
		plt.show()

	def visualize_weight_matrix(self):
		d = unpickle("cifar-10-batches-py/batches.meta")
		label_names = d["label_names"]
		fig, axes = plt.subplots(1, 10, figsize=(16, 16))
		for i in range(10):
			img = self.W[:, i].reshape(3, 32, 32);
			img = np.divide(img - np.amin(img), np.amax(img) - np.amin(img))
			axes[i].set_axis_off()
			axes[i].set_title("{}".format(label_names[i]))
			axes[i].imshow(np.transpose(img, (1, 2, 0)), interpolation='bicubic')
		plt.show()

	def __repr__(self):
		return ("{} linear classifier\n\tweight matrix [{} x {}]\n\tbias vector"
		" [{} x {}]\n\tlearning rate {}\n\tregularization stregth {}").format(
		self.name, self.W.shape[0], self.W.shape[1], self.b.shape[0], 
		self.b.shape[1], self.learning_rate, self.reg_strength)


class SoftmaxClassifier(LinearClassifier):

	def __init__(self, W, b, learning_rate, reg_strength):
		LinearClassifier.__init__(self, W, b, learning_rate, reg_strength)
		self.name = "Softmax"

	def evaluate_classifier(self, X):
		# Evaluate the class scores (unnormalized log probabilities of the classes)
		scores = np.dot(X, self.W) + self.b # [N x K]
		# Compute the normalized class probabilities
		exp_scores = np.exp(scores)
		probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
		return probs

	def compute_cost(self, X, y):
		probs = self.evaluate_classifier(X)
		correct_logprobs = -np.log(probs[range(probs.shape[0]), y])
		data_cost = np.sum(correct_logprobs) / probs.shape[0]
		reg_cost = self.reg_strength * np.sum(self.W * self.W)
		cost = data_cost + reg_cost
		return cost

	def compute_gradients(self, X, y, probs): 
		# Compute the gradient on the scores
		grad_scores = probs
		grad_scores[range(X.shape[0]), y] -= 1
		grad_scores /= X.shape[0]
		# Backpropate the gradient to the parameters (W,b)
		grad_W = np.dot(X.T, grad_scores)
		grad_b = np.sum(grad_scores, axis=0, keepdims=True)
		# Add regularization gradient
		grad_W += self.reg_strength * self.W 
		return grad_W, grad_b



def unpickle(file):
    """
        Unpickle function from http://www.cs.utoronto.ca/~kriz/cifar.html
    """
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


def load_batch(filename, batch=None):
	"""
		Read in data from one or more CIFAR-10 batch files specified by batch
		and return the image and label data in separate variables
	"""
	if batch is None:
		d = unpickle(filename)
		x = np.float64(d['data'])
		y = np.uint8(d['labels'])		
	if isinstance(batch, int):
		d = unpickle(filename +  str(batch))
		x = np.float64(d['data'])
		y = np.uint8(d['labels'])
	if isinstance(batch, tuple):
		x = []
		y = []
		for i in range(batch[0], batch[1] + 1):
			d = unpickle(filename + str(i))
			x.append(d['data'])
			y.append(d['labels'])
		x = np.float64(np.concatenate(x))
		y = np.uint8(np.concatenate(y))
	# Convert the data to be between 0 and 1
	x /= np.amax(np.abs(x))
	return x, y



if __name__ == '__main__':

	# Load training, validation, and test data
	train_data, train_labels = load_batch('cifar-10-batches-py/data_batch_', 1)
	valid_data, valid_labels = load_batch('cifar-10-batches-py/data_batch_', 2)
	test_data, test_labels = load_batch('cifar-10-batches-py/test_batch')

	# Initialize parameters as Gaussian random values (mean = 0 and s.d. = 0.01)
	K = np.amax(train_labels) + 1
	N, d = train_data.shape
	W = np.random.normal(0, 0.01, (d, K))
	b = np.random.normal(0, 0.01, (1, K))

	# Set hyperparameters
	learning_rate = 0.01 # eta
	reg_strength = 0 # lambda

	# Create classifier instance
	softmax = SoftmaxClassifier(W, b, learning_rate, reg_strength)
	print softmax

	# Set other gradient descent parameters
	n_epochs = 20
	n_batch = 100

	train_loss, valid_loss, train_acc, valid_acc = \
		softmax.vanilla_gradient_descent(train_data, train_labels, valid_data, \
		valid_labels, n_epochs, n_batch)

	# Plot loss
	softmax.plot_loss(train_loss, valid_loss, n_epochs)

	# Plot accuracy
	softmax.plot_accuracy(train_acc, valid_acc, n_epochs)

	# visualize W
	softmax.visualize_weight_matrix()

	# Compute the accuracy of the learnt classifier on the test data
	accuracy = softmax.compute_accuracy(test_data, test_labels)
	print accuracy





# to do
# gradient check 
# double check mini batch GD 
# read more about vanilla/ stochastic/ minibatch GD
# add comments
# implement SVM classifier class
# improvements from the lab
# add text to notebook


# MY IMPROVEMENTS
# center data to 0 so normalize to [-1 1] (not [0 1])









