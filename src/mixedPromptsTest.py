import tensorflow as tf

sess = tf.Session()
#Alice: 14862
with sess.as_default():
    logits = tf.constant([[-10,-9,-8,-7, -99]], dtype = tf.float32)
    logitValues,index = tf.nn.top_k(logits, k=3)
    X = tf.constant([[464, 41118, 7893, 14862, 5811, 11, 366, 2061, 338, 534, 1438, 1701, 1375, 7429, 11, 366, 3666,
                     1438, 318]], dtype = tf.int32)
    alice = X[0,3]
    bob = X[0,4]
    shapeX = tf.concat([X[0,0:4], X[0,5:]], axis=0)
    size = X.shape.as_list()
    size2 = tf.shape(X)


    print(size2.eval())
    # print(tf.get_default_session())
print(size[1])
# print(tf.get_default_session())
# print(tf.get_static_value(size))
