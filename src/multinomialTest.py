import tensorflow as tf


sess = tf.Session()
with sess.as_default():
    logits = tf.constant([[-10,-9,-8,-7, -99]], dtype = tf.float32)
    topK,other = tf.nn.top_k(logits, k=3)
    print(topK)
    print(tf.shape(topK).eval())
    print(other)
    print(tf.shape(other).eval())
    print("--------------------------------------------------------")

    samples = tf.multinomial(logits, num_samples=3, output_dtype=tf.int32)
    softmax = tf.nn.softmax(logits)
    index = tf.where(logits>-10)
    print(index.eval())
    print(tf.shape(logits).eval())
    print("9999999999999999")
    newTF = logits[0,:]
    print(newTF.eval())
    final = tf.gather_nd(logits, index)
    print(final.eval())
    att = tf.zeros(shape = (1,3), dtype=tf.float32)
    final = tf.expand_dims(final, axis=0)
    print("==============================================")
    print(tf.shape(att).eval())
    print(tf.shape(final).eval())
    print("==============================================")
    comb = tf.concat([att, final], axis=0)
    print(comb.eval())

    indexAtt = index[:,1]
    print(indexAtt.eval())
    a = tf.constant([], dtype=tf.int64)
    print(a.eval())
    print(tf.shape(a).eval())
    comb = tf.concat([a, indexAtt], axis = 0)
    print(comb.eval())
    print(tf.shape(comb).eval())

