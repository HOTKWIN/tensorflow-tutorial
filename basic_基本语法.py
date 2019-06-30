import tensorflow as tf
import numpy as np
import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

#create new data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

### create tensorflow structure start ###

#定义变量
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))

#计算预测值
y = Weights * x_data +biases

#loss function
loss = tf.reduce_mean(tf.square(y-y_data))

#优化器
optimizer = tf.train.GradientDescentOptimizer(0.5)

#训练目标loss最小化
train = optimizer.minimize(loss)

#初始化变量
init = tf.global_variables_initializer()

#创建session,进行参数初始化
sess = tf.Session()
sess.run(init)

#开始训练200步，20步输出一下两个参数
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(Weights),sess.run(biases))

### create tensorflow structure end ###



#tf中的Session,可用session.run来运行框架中某一个点的功能
import tensorflow as tf

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],[2]])

product = tf.matmul(matrix1,matrix2)

sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)



#tf中的Variable
import tensorflow as tf

#定义变量，给定初始值和name
state = tf.Variable(0,name = "counter")
print(state.name)

one = tf.constant(1)

new_value = tf.add(state,one)
update = tf.assign(state,new_value)

#此处只是定义，必须用session.run来执行
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))



#TF placeholder
import tensorflow as tf

input1 = tf.placeholder(dtype=tf.float32)
input2 = tf.placeholder(dtype=tf.float32)

output = tf.multiply(input1,input2)

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[3.],input2:[5]}))