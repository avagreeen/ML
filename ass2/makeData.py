from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot
from pandas import DataFrame
from sklearn.datasets import make_circles

def genBlob():
# generate 2d classification dataset
	X, y = make_blobs(n_samples=100, centers=3, n_features=2)
# scatter plot, dots colored by class value
	df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
	colors = {0:'red', 1:'blue', 2:'green'}
	fig, ax = pyplot.subplots()
	grouped = df.groupby('label')
	for key, group in grouped:
		group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
	pyplot.show()

def genCircle():
	X, y = make_circles(n_samples=100, noise=0.05)
	# scatter plot, dots colored by class value
	df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
	colors = {0:'red', 1:'blue'}
	fig, ax = pyplot.subplots()
	grouped = df.groupby('label')
	for key, group in grouped:
		group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
	pyplot.show()