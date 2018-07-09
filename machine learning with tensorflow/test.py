import tensorflow as tf
import numpy as np

m1 = [[1.0, 2.0], [3.0, 4.0]]
m2 = np.array([[1.0, 2.0], [3.0, 4.0]])
m3 = tf.constant([[1.0, 2.0], [3.0, 4.0]])

print(type(m1))
print(type(m2))
print(type(m3))

t1 = tf.convert_to_tensor(m1)
print(type(t1))


# create tf session to run code
# allows change settings without changing code
x = tf.constant([[1., 2.]])
neg_op = tf.negative(x)
with tf.Session() as sess:
    result = sess.run(neg_op)
    print(result)