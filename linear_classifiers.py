import cPickle
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

# sns.set()

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
	# Stack 2D arrays (images) into a single 3D array for processing
	# x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
	return x, y


def evaluate_classifier(X, W, b):
	# Evaluate the class scores (unnormalized log probabilities of the classes)
	scores = np.dot(X, W) + b # [N x K]
	# Compute the normalized class probabilities
	exp_scores = np.exp(scores)
	probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
	return probs


def compute_cost(X, y, W, b, reg_strength):
	probs = evaluate_classifier(X, W, b)
	correct_logprobs = -np.log(probs[range(probs.shape[0]), y])
	data_cost = np.sum(correct_logprobs) / probs.shape[0]
	reg_cost = reg_strength * np.sum(W * W)
	cost = data_cost + reg_cost
	return cost


def compute_accuracy(X, y, W, b):
	probs = evaluate_classifier(X, W, b)
	acc = np.mean(np.argmax(probs, axis=1) == y)
	return acc


def compute_gradients(X, y, probs, W, reg_strength): # todo implement gradient check
	# Compute the gradient on the scores
	grad_scores = probs
	grad_scores[range(X.shape[0]), y] -= 1
	grad_scores /= X.shape[0]
	# Backpropate the gradient to the parameters (W,b)
	grad_W = np.dot(X.T, grad_scores)
	grad_b = np.sum(grad_scores, axis=0, keepdims=True)
	# Add regularization gradient
	grad_W += reg_strength * W 
	return grad_W, grad_b


def mini_batch_gradient_descent(X, y, W, b, learning_rate, reg_strength):
	probs = evaluate_classifier(X, W, b)
	grad_W, grad_b = compute_gradients(X, y, probs, W, reg_strength)
	return W - learning_rate * grad_W, b - learning_rate * grad_b 



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

	# Other gradient descent parameters
	n_epochs = 40
	n_batch = 100

	# Losses
	train_loss = np.zeros(n_epochs)
	valid_loss = np.zeros(n_epochs) 
	train_acc = np.zeros(n_epochs)
	valid_acc = np.zeros(n_epochs)

	for ep in range(n_epochs):
		print "Epoch ", ep + 1
		for j in range (N / n_batch): # todo make sure this is how mini batch gradient descent works
			start = j * n_batch 
			end = (j + 1) * n_batch - 1	
			batch_data = train_data[start:end, ] 
			batch_labels = train_labels[start:end, ]
			W, b = mini_batch_gradient_descent(batch_data, batch_labels, W, b, 
				learning_rate, reg_strength)
		train_loss[ep] = compute_cost(train_data, train_labels, W, b, reg_strength)
		valid_loss[ep] = compute_cost(valid_data, valid_labels, W, b, reg_strength)
		train_acc[ep] = compute_accuracy(train_data, train_labels, W, b)
		valid_acc[ep] = compute_accuracy(valid_data, valid_labels, W, b)

	# Plot loss
	epochs = np.arange(1, n_epochs+1)
	plt.plot(epochs, train_loss, label="training loss")
	plt.plot(epochs, valid_loss, label="validation loss")
	plt.xlabel("Epochs")
	plt.ylabel("Loss")
	plt.legend(loc='upper right', shadow=True)
	plt.show()

	# Plot accuracy
	plt.plot(epochs, train_acc, label="training accuracy")
	plt.plot(epochs, valid_acc, label="validation accuracy")
	plt.xlabel("Epochs")
	plt.ylabel("Accuracy")
	plt.legend(loc='lower right', shadow=True)
	plt.show()

	# Compute the accuracy of the learnt classifier on the test data
	accuracy = compute_accuracy(test_data, test_labels, W, b)
	print accuracy





# to do
# visualize w
# gradient check 
# double check mini batch GD 
# read more about vanilla/ stochastic/ minibatch GD
# add comments
# implement softmax and 


# MY IMPROVEMENTS
# center data to 0 so normalize to [-1 1] (not [0 1])









