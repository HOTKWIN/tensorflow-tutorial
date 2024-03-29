import tensorflow as tf
import numpy as np
import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

#保存
# W = tf.Variable([[1,2,3],[3,4,5]],dtype=tf.float32,name='weights')
# b = tf.Variable([[1,2,3]],dtype=tf.float32,name='biases')
#
# init = tf.global_variables_initializer()
#
# '''
# 保存时, 首先要建立一个 tf.train.Saver() 用来保存, 提取变量.
# 再创建一个名为my_net的文件夹, 用这个 saver 来保存变量到这个目录"my_net/save_net.ckpt".
# '''
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     sess.run(init)
#     save_path = saver.save(sess,"my_net/save_net.ckpt")
#     print("Save to path:",save_path)



#重载
"""重载变量
先重新定义为你的变量定义相同的形状
"""

W = tf.Variable(np.arange(6).reshape(2,3),name='weights',dtype=tf.float32)
b = tf.Variable(np.arange(3).reshape(1,3),name='biases',dtype=tf.float32)

# not need init step
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,"my_net/save_net.ckpt")
    print("weights:",sess.run(W))
    print("biases:",sess.run(b))


