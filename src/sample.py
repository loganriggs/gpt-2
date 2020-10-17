import tensorflow as tf

import model

def top_k_logits(logits, k):
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        return tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )
    return tf.cond(
       tf.equal(k, 0),
       lambda: logits,
       lambda: _top_k(),
    )


def sample_sequence(*, hparams, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, softmax_length=10):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = tf.fill([batch_size, 1], start_token)

    def step(hparams, tokens, past=None, mix_prompt=False):
        lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE, mix_prompt=mix_prompt)

        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        presents = lm_output['present']
        presents.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))
        return {
            'logits': logits,
            'presents': presents,
        }

    with tf.name_scope('sample_sequence'):
        # Don't feed the last context token -- leave that to the loop below
        # TODO: Would be slightly faster if we called step on the entire context,
        # rather than leaving the last token transformer calculation to the while loop.
        context_output = step(hparams, context[:, :-1],mix_prompt=True)

        def body(past, prev, output, index, softmax):
            next_outputs = step(hparams, prev[:, tf.newaxis], past=past)
            logits = next_outputs['logits'][:, -1, :]  / tf.to_float(temperature)
            #Calculate Softmax of top N, before we cancel them to -1e10
            softmax_length_logits, index_loop = tf.nn.top_k(logits, k=softmax_length)
            softmax_loop = tf.nn.softmax(softmax_length_logits)

            logits = top_k_logits(logits, k=top_k)
            samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)

            #
            # #Index of Logits
            # indexOfLogits = tf.where(logits>-1e9)
            # indexAsList = indexOfLogits[:,1]
            # indexAsList = tf.expand_dims(indexAsList, axis=0)
            # #Softmax of top k logits
            # kLogits = tf.gather_nd(logits, indexOfLogits)
            # softmaxOfLogits = tf.nn.softmax(kLogits)
            # softmaxOfLogits = tf.expand_dims(softmaxOfLogits, axis=0)
            #We want the word distribution of the last one.
            #I think multinomial is converting logits to an index. Maybe convert num_samples = 10? Then Softmax?
            return [
                tf.concat([past, next_outputs['presents']], axis=-2),
                tf.squeeze(samples, axis=[1]),
                tf.concat([output, samples], axis=1),
                tf.concat([index, index_loop], axis=0),
                tf.concat([softmax, softmax_loop], axis=0)
            ]

        def cond(*args):
            return True

        #Filler Index and Softmax Tensors for first loop
        #Therefor, skip first row in output
        fillerIndex = tf.zeros(shape=[1,softmax_length], dtype=tf.int32)
        fillerSoftmax = tf.zeros(shape=[1,softmax_length], dtype=tf.float32)


        _, _, tokens, index, softmax = tf.while_loop(
            cond=cond, body=body,
            maximum_iterations=length,
            loop_vars=[
                context_output['presents'],
                context[:, -1],
                context,
                fillerIndex,
                fillerSoftmax,
            ],
            shape_invariants=[
                tf.TensorShape(model.past_shape(hparams=hparams, batch_size=batch_size)),
                tf.TensorShape([batch_size]),
                tf.TensorShape([batch_size, None]),
                tf.TensorShape([None,None]),
                tf.TensorShape([None,None]),
            ],
            back_prop=False,
        )

        return tokens, index, softmax
