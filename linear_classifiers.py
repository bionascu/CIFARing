# Last changed: 04/11/2017
# Author: Beatrice Ionascu (bionascu@kth.se)

# DD2424 Assignment 1

import cPickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc


class LinearClassifier:

	def __init__(self, W, b, learning_rate, reg_strength):
		self.name = "Abstract"
		self.W = W
		self.b = b
		self.learning_rate = learning_rate
		self.reg_strength = reg_strength
		self.bestW = None
		self.bestb = None

	def evaluate_classifier(self, X):
		# Evaluate the class scores 
		scores = np.dot(self.W, X) + self.b # [K x N]
		return scores

	def compute_cost(self, X, y):
		raise NotImplementedError("Should have implemented this")

	def compute_accuracy(self, X, y):
		scores = self.evaluate_classifier(X)
		acc = np.mean(np.argmax(scores, axis=0) == y)
		return acc

	def compute_gradients(self, X, y, scores): 
		raise NotImplementedError("Should have implemented this")

	def gradient_check(self):
		raise NotImplementedError("Should have implemented this")

	def mini_batch_gradient_descent(self, X, y, n_batch):
		for j in range(X.shape[1] / n_batch): 
			start = j * n_batch 
			end = (j + 1) * n_batch	
			batch_X = X[:, start:end] 
			batch_y = y[start:end]
			scores = self.evaluate_classifier(batch_X)
			grad_W, grad_b = self.compute_gradients(batch_X, batch_y, scores)
			self.W -= self.learning_rate * grad_W
			self.b -= self.learning_rate * grad_b 

	def vanilla_gradient_descent(self, train_data, train_labels, valid_data,
			valid_labels, n_epochs, n_batch, decay=False):
		# Losses
		train_loss = np.zeros(n_epochs+1)
		valid_loss = np.zeros(n_epochs+1) 
		# Accuracies
		train_acc = np.zeros(n_epochs+1)
		valid_acc = np.zeros(n_epochs+1)

		train_loss[0] = self.compute_cost(train_data, train_labels)
		valid_loss[0] = self.compute_cost(valid_data, valid_labels)
		train_acc[0] = self.compute_accuracy(train_data, train_labels)
		valid_acc[0] = self.compute_accuracy(valid_data, valid_labels)
		# Train
		bestacc = -1
		for ep in range(n_epochs):
			# print "Epoch ", ep + 1 
			self.mini_batch_gradient_descent(train_data, train_labels, n_batch)
			train_loss[ep+1] = self.compute_cost(train_data, train_labels)
			valid_loss[ep+1] = self.compute_cost(valid_data, valid_labels)
			train_acc[ep+1] = self.compute_accuracy(train_data, train_labels)
			valid_acc[ep+1] = self.compute_accuracy(valid_data, valid_labels)
			if (decay):
				self.learning_rate *= 0.9
				# print self.learning_rate
			if (valid_acc[ep+1] > bestacc):
				# print "best: ", ep+1
				bestacc = valid_acc[ep+1]
				self.bestW = np.copy(self.W)
				self.bestb = np.copy(self.b)
		return train_loss, valid_loss, train_acc, valid_acc

	def __repr__(self):
		return ("{} linear classifier\n\tweight matrix [{} x {}]\n\tbias vector"
		" [{} x {}]\n\tlearning rate {}\n\tregularization stregth {}").format(
		self.name, self.W.shape[0], self.W.shape[1], self.b.shape[0], 
		self.b.shape[1], self.learning_rate, self.reg_strength)


class SoftmaxClassifier(LinearClassifier):

	def __init__(self, W, b, learning_rate, reg_strength):
		LinearClassifier.__init__(self, W, b, learning_rate, reg_strength)
		self.name = "Softmax"

	def softmax(self, scores):
		"""
		Interpret class scores as unnormalized log probabilities of the classes
		and compute the normalized class probabilities.
		"""
		scores -= np.max(scores) # avoid overflow by shifting scores to be < 0
		exp_scores = np.exp(scores)
		probs = exp_scores / np.sum(exp_scores, axis=0, keepdims=True) # [K x N]
		return probs

	def evaluate_classifier(self, X):
		"""
		Redefine method to evaluate the class scores using a probabilistic 
		interpretation.
		"""
		scores = np.dot(self.W, X) + self.b # [K x N]
		return self.softmax(scores)

	def compute_cost(self, X, y):
		"""
		Compute the cross-entropy between the estimated class probabilities and 
		the true distribution (= probability mass is on the correct class).
		"""
		probs = self.evaluate_classifier(X) # [K x N]
		correct_logprobs = -np.log(probs[y, range(probs.shape[1])])
		data_cost = np.mean(correct_logprobs)
		reg_cost = self.reg_strength * np.sum(self.W * self.W)
		cost = data_cost + reg_cost
		return cost

	def compute_gradients(self, X, y, scores): 
		# Compute the gradient on the scores
		grad_scores = scores
		grad_scores[y, range(X.shape[1])] -= 1
		grad_scores /= X.shape[1]
		# Backpropate the gradient to the parameters (W,b)
		grad_W = np.dot(grad_scores, X.T)
		grad_b = np.sum(grad_scores, axis=1, keepdims=True)
		# Add regularization gradient
		grad_W += 2 * self.reg_strength * self.W 
		return grad_W, grad_b


class SvmClassifier(LinearClassifier):

	def __init__(self, W, b, learning_rate, reg_strength, form="SVM"):
		LinearClassifier.__init__(self, W, b, learning_rate, reg_strength)
		self.name = "SVM"
		# The margin hyperparameter can safely be set to 1.0 in all cases
		# because it controls the same tradeoff as the reg_strength
		self.margin = 1.0 
		self.form = form

	def compute_cost(self, X, y):
		"""
		Compute the Multiclass SVM loss (hinge loss) or the L2-SVM loss (squared 
		hinge loss).
		"""
		N = X.shape[1]
		scores = self.evaluate_classifier(X) # [K x N]
		margins = np.maximum(0, scores - scores[y, range(N)] + self.margin)
		if self.form == "L2-SVM":
			margins = np.power(margins, 2)
			margins[y, range(N)] -= np.power(self.margin, 2)
		else:
			margins[y, range(N)] -= self.margin
		data_cost = np.mean(np.sum(margins, axis=0))
		reg_cost = self.reg_strength * np.sum(self.W * self.W)
		cost = data_cost + reg_cost
		return cost

	def compute_gradients(self, X, y, scores): 
		margins = scores - scores[y, range(X.shape[1])] + self.margin
		# Compute the gradient corresponding to the wrong class
		grad_scores = np.where(margins >= 0, 1, 0)
		# Compute the gradient on the correct scores by counting the number of 
		# classes that didn't meet the desired margin
		grad_scores[y, range(X.shape[1])] -= np.sum(grad_scores, axis=0)
		# Backpropate the gradient to the parameters (W,b)
		grad_W = np.dot(grad_scores, X.T)
		grad_b = np.sum(grad_scores, axis=1, keepdims=True)
		# Add regularization gradient
		grad_W += 2 * self.reg_strength * self.W 
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
	and return the image and label data in separate variables.
	"""
	if batch is None:
		d = unpickle(filename)
		X = np.float64(d['data'])
		y = np.uint8(d['labels'])		
	if isinstance(batch, int):
		d = unpickle(filename +  str(batch))
		X = np.float64(d['data'])
		y = np.uint8(d['labels'])
	if isinstance(batch, tuple):
		X = []
		y = []
		for i in range(batch[0], batch[1] + 1):
			d = unpickle(filename + str(i))
			X.append(d['data'])
			y.append(d['labels'])
		X = np.float64(np.concatenate(X))
		y = np.uint8(np.concatenate(y))
	# Convert the data to be between 0 and 1
	X /= np.amax(np.abs(X))
	return X.T, y.T


def split_train_data(data, labels, valid_size):
	train_data = data[:, :-valid_size]
	train_labels = labels[:-valid_size]
	valid_data = data[:, -valid_size:]
	valid_labels = labels[-valid_size:]
	return train_data, train_labels, valid_data, valid_labels

def plot_loss(train_loss, valid_loss, n_epochs):
	epochs = np.arange(n_epochs + 1)
	plt.plot(epochs, train_loss, label="training loss")
	plt.plot(epochs, valid_loss, label="validation loss")
	plt.xlabel("Epochs")
	plt.ylabel("Loss")
	plt.legend(loc='upper right', shadow=True)
	plt.show()

def plot_accuracy(train_acc, valid_acc, n_epochs):
	epochs = np.arange(n_epochs + 1)
	plt.plot(epochs, train_acc, label="training accuracy")
	plt.plot(epochs, valid_acc, label="validation accuracy")
	plt.xlabel("Epochs")
	plt.ylabel("Accuracy")
	plt.legend(loc='lower right', shadow=True)
	plt.show()

def plot_loss_acc(train_loss, valid_loss, train_acc, valid_acc, n_epochs):
	epochs = np.arange(n_epochs + 1)
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

def visualize_weight_matrix(W):
	d = unpickle("cifar-10-batches-py/batches.meta")
	label_names = d["label_names"]
	fig, axes = plt.subplots(1, 10, figsize=(16, 16))
	for i in range(10):
		img = W[i, ].reshape(3, 32, 32);
		img = np.divide(img - np.amin(img), np.amax(img) - np.amin(img))
		axes[i].set_axis_off()
		axes[i].set_title("{}".format(label_names[i]))
		axes[i].imshow(np.transpose(img, (1, 2, 0)), interpolation='bicubic')
	plt.show()


if __name__ == '__main__':

	# Load training, validation, and test data
	train_data, train_labels = load_batch('cifar-10-batches-py/data_batch_', 1)
	valid_data, valid_labels = load_batch('cifar-10-batches-py/data_batch_', 2)
	test_data, test_labels = load_batch('cifar-10-batches-py/test_batch')

	# Initialize parameters as Gaussian random values (mean = 0 and s.d. = 0.01)
	K = np.amax(train_labels) + 1
	d, N = train_data.shape
	W = np.random.normal(0, 0.01, (K, d))
	b = np.random.normal(0, 0.01, (K, 1))

	# Set hyperparameters
	learning_rate = 0.00001 # eta
	reg_strength = 1 # lambda

	# Set other gradient descent parameters
	n_epochs = 40
	n_batch = 100

	# Create Softmax classifier instance
	softmax = SoftmaxClassifier(np.copy(W), np.copy(b), learning_rate, reg_strength)
	print softmax
	train_loss, valid_loss, train_acc, valid_acc = \
		softmax.vanilla_gradient_descent(train_data, train_labels, valid_data, \
		valid_labels, n_epochs, n_batch)
	# Plot loss
	plot_loss(train_loss, valid_loss, n_epochs)
	# Plot accuracy
	plot_accuracy(train_acc, valid_acc, n_epochs)
	# visualize W
	visualize_weight_matrix(softmax.W)
	# Compute the accuracy of the learnt classifier on the test data
	accuracy = softmax.compute_accuracy(test_data, test_labels)
	print accuracy

	# Create SVM classifier instance
	svm = SvmClassifier(np.copy(W), np.copy(b), learning_rate, reg_strength, "SVM")
	print svm
	train_loss2, valid_loss2, train_acc2, valid_acc2 = \
		svm.vanilla_gradient_descent(train_data, train_labels, valid_data, \
		valid_labels, n_epochs, n_batch)
	plot_loss(train_loss2, valid_loss2, n_epochs)
	plot_accuracy(train_acc2, valid_acc2, n_epochs)
	visualize_weight_matrix(svm.W)
	accuracy2 = svm.compute_accuracy(test_data, test_labels)
	print accuracy2



	"""Optimization"""
	# Load training, validation, and test data
	test_data, test_labels = load_batch('cifar-10-batches-py/test_batch')
	data, labels = load_batch('cifar-10-batches-py/data_batch_', (1, 5))

	train_data, train_labels, valid_data, valid_labels = \
		split_train_data(data, labels, 1000)

	train_data -= np.mean(train_data, axis = 0)
	valid_data -= np.mean(valid_data, axis = 0)
	test_data -= np.mean(test_data, axis = 0)

	# Initialize parameters as Gaussian random values (mean = 0 and s.d. = 0.01)
	K = np.amax(train_labels) + 1
	d, N = train_data.shape
	W = np.random.randn(K, d) / np.sqrt(d)
	b = np.zeros((K, 1))

	# Set hyperparameters
	learning_rate = 0.01 # eta
	reg_strength = 0 # lambda

	# Set other gradient descent parameters
	n_epochs = 40
	n_batch = 100


	# Create classifier instance
	softmax = SoftmaxClassifier(np.copy(W), np.copy(b), learning_rate, reg_strength)
	print softmax
	train_loss, valid_loss, train_acc, valid_acc = \
		softmax.vanilla_gradient_descent(train_data, train_labels, valid_data, \
		valid_labels, n_epochs, n_batch, True)
	# Plot loss
	plot_loss(train_loss, valid_loss, n_epochs)
	# Plot accuracy
	plot_accuracy(train_acc, valid_acc, n_epochs)
	# visualize W
	visualize_weight_matrix(softmax.W)
	# Compute the accuracy of the learnt classifier on the test data
	accuracy = softmax.compute_accuracy(test_data, test_labels)
	print accuracy

	visualize_weight_matrix(softmax.bestW)
	bestsoftmax = SoftmaxClassifier(np.copy(softmax.bestW), np.copy(softmax.bestb), \
		learning_rate, reg_strength)
	bestaccuracy = bestsoftmax.compute_accuracy(test_data, test_labels)
	print bestaccuracy














