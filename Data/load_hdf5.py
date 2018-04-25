dataset = h5py.File('Data/train_test_data.h5', 'r')

print('Keys : %s' % dataset.keys())

test_X = np.array(dataset["test_X"][:])
test_Y = np.array(dataset["test_Y"][:])
test_Y = np.reshape(test_Y, (len(test_Y),1))

train_X = np.array(dataset["train_X"][:])
train_Y = np.array(dataset["train_Y"][:])
train_Y = np.reshape(train_Y, (len(train_Y),1))


print('train_X is a {} array and has : {} examples' .format(train_X.shape, len(train_X)))
print('train_Y is a {} array and has {} labels' .format(train_Y.shape, len(train_Y)))

print('test_X is a {} array and has : {} examples' .format(test_X.shape, len(test_X)))
print('test_Y is a {} array and has {} labels' .format(test_Y.shape, len(test_Y)))