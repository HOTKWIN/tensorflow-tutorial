import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


digits = load_digits()
X = digits.data
Y = digits.target
Y = LabelBinarizer().fit_transform(Y)
print(Y.shape)
trainx,testx,trainy,testy = train_test_split(X,Y,test_size=0.3)


def add_layer(input,input_size,output_size,layer_name,activation_function=None):
    pass

xs = tf.placeholder(tf.float32,[None,64])
ys = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)

l1 = add_layer(xs,64,50,"l1",activation_function=tf.nn.tanh)
prediction = add_layer(l1,50,10,"l2",activation_function=tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))
tf.summary.scalar("loss",cross_entropy)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    #合并所有的summary
    merged = tf.summary.merge_all()
    #得到summary的FileWriter
    train_writer = tf.summary.FileWriter('logs/train/',sess.graph)
    test_writer = tf.summary.FileWriter('logs/test/', sess.graph)
    sess.run(init)
    for i in range(1000):
        sess.run(train_step,feed_dict = {xs:trainx,ys:trainy,keep_prob:0.5})
        if i%50 == 0:
            print(sess.run(merged,feed_dict={xs:trainx,ys:trainy}))
            #这里要运行merged
            train_loss = sess.run(merged,feed_dict={xs:trainx,ys:trainy,keep_prob:0.5})
            test_loss = sess.run(merged,feed_dict={xs:testx,ys:testy,keep_prob:0.5})
            #将loss写入FileWriter中
            train_writer.add_summary(train_loss,i)
            test_writer.add_summary(test_loss,i)