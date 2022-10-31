#X = np.hstack((X, np.ones((X.shape[0], 1))))  # add bias
#print(shape(X), shape(W1), shape(b1))
z1 = np.dot(X, W1) + b1  # compute the dot
a1 = relu(z1)  # use activation function
#a1 = np.hstack((a1, np.ones((a1.shape[0], 1))))  # add bias
z2 = np.dot(a1, W2) + b2  # compute the dot
a2 = softmax(z2)  # predict the probabilities
### END CODE

### YOUR CODE HERE - BACKWARDS PASS - compute derivatives of all weights and bias, store them in d_w1, d_w2, d_b1, d_b2
d_w2 = np.dot(a1.T, a2 - labels) + c * W2
d_b2 = np.sum(a2 - labels, axis=0)
d_a1 = np.dot(a2 - labels, W2.T)
d_z1 = d_a1 * (z1 > 0)
d_w1 = np.dot(X.T, d_z1) + c * W1
d_b1 = np.sum(d_z1, axis=0)
# add fake axis to bs
d_b1 = d_b1[:, np.newaxis]
d_b2 = d_b2[:, np.newaxis]
# average cross entropy cost
cost = -np.mean(np.sum(labels * np.log(a2), axis=1)) + c * (np.sum(W1 * W1) + np.sum(W2 * W2))
# print shapes
print('d_w1', d_w1.shape)
print('d_b1', d_b1.shape)
print('d_w2', d_w2.shape)
print('d_b2', d_b2.shape)