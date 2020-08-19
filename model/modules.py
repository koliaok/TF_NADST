import tensorflow as tf

class LayerNorm():
    '''Applies layer normalization. See https://arxiv.org/abs/1607.06450.
    inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    scope: Optional scope for `variable_scope`.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    def __init__(self, inputs, epsilon=1e-6, scope="ln"):
        with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]
            mean, variance = tf.compat.v1.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.compat.v1.get_variable("beta", params_shape, initializer=tf.compat.v1.zeros_initializer())
            gamma = tf.compat.v1.get_variable("gamma", params_shape, initializer=tf.compat.v1.ones_initializer())
            self.normalized = gamma*((inputs - mean)/(variance + epsilon))+beta

    def get_layer_norm(self):
        return self.normalized



def mask(inputs, key_masks=None, type=None):
    """Masks paddings on keys or queries to inputs
    inputs: 3d tensor. (h*N, T_q, T_k)
    key_masks: 3d tensor. (N, 1, T_k)
    type: string. "key" | "future"

    e.g.,
    >> inputs = tf.zeros([2, 2, 3], dtype=tf.float32)
    >> key_masks = tf.constant([[0., 0., 1.],
                                [0., 1., 1.]])
    >> mask(inputs, key_masks=key_masks, type="key")
    array([[[ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09],
        [ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09]],

       [[ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09],
        [ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09]],

       [[ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09],
        [ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09]],

       [[ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09],
        [ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09]]], dtype=float32)
    """

    padding_num = -2 ** 32 + 1
    if type in ("k", "key", "keys"):
        key_masks = tf.compat.v1.to_float(key_masks)
        t_ret = tf.compat.v1.shape(inputs)[0] // tf.compat.v1.shape(key_masks)[0]
        key_masks = tf.compat.v1.tile(key_masks, [tf.compat.v1.shape(inputs)[0] // tf.compat.v1.shape(key_masks)[0], 1])  # (h*N, seqlen)
        key_masks = tf.compat.v1.expand_dims(key_masks, 1)  # (h*N, 1, seqlen)
        outputs = inputs + key_masks * padding_num
    elif type in ("f", "future", "right"):
        diag_vals = tf.compat.v1.ones_like(inputs[0, :, :])  # (T_q, T_k)
        tril = tf.compat.v1.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
        future_masks = tf.compat.v1.tile(tf.compat.v1.expand_dims(tril, 0), [tf.compat.v1.shape(inputs)[0], 1, 1])  # (N, T_q, T_k)

        paddings = tf.compat.v1.ones_like(future_masks) * padding_num
        outputs = tf.compat.v1.where(tf.compat.v1.equal(future_masks, 0), paddings, inputs)
    else:
        print("Check if you entered type correctly!")

    return outputs


def multihead_attention(queries, keys, values, key_masks,
                        num_heads=8,
                        dropout_rate=0,
                        training=True,
                        scope="multihead_attention",
                        nomalize=True,
                        no_mask=False):
    '''Applies multihead attention. See 3.2.2
    queries: A 3d tensor with shape of [N, T_q, d_model].
    keys: A 3d tensor with shape of [N, T_k, d_model].
    values: A 3d tensor with shape of [N, T_k, d_model].
    key_masks: A 2d tensor with shape of [N, key_seqlen]
    num_heads: An int. Number of heads.
    dropout_rate: A floating point number.
    training: Boolean. Controller of mechanism for dropout.
    causality: Boolean. If true, units that reference the future are masked.
    scope: Optional scope for `variable_scope`.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    d_model = queries.get_shape().as_list()[-1]
    with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
        # Linear projections
        Q = tf.compat.v1.layers.dense(queries, d_model, use_bias=True)  # (N, T_q, d_model)
        K = tf.compat.v1.layers.dense(keys, d_model, use_bias=True)  # (N, T_k, d_model)
        V = tf.compat.v1.layers.dense(values, d_model, use_bias=True)  # (N, T_k, d_model)

        # Split and concat

        Q_ = tf.compat.v1.concat(tf.compat.v1.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, d_model/h)
        K_ = tf.compat.v1.concat(tf.compat.v1.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)
        V_ = tf.compat.v1.concat(tf.compat.v1.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)

        # Attention
        outputs, softmax_out = scaled_dot_product_attention(Q_, K_, V_, key_masks,
                                               dropout_rate, training, no_mask, scope=scope)

        # Restore shape
        outputs = tf.compat.v1.concat(tf.compat.v1.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, d_model)
        outputs = tf.compat.v1.layers.Dense(units=d_model, use_bias=True)(outputs)
        if training:
            outputs = tf.compat.v1.layers.dropout(outputs, rate=dropout_rate, training=training)
        # Residual connection
        outputs += queries

        # Normalize
        if nomalize:
            return outputs
        else:
            return outputs, softmax_out


def scaled_dot_product_attention(Q, K, V, key_masks, dropout_rate=0.,
                                 training=True, no_mask=False,
                                 scope="scaled_dot_product_attention"):
    '''See 3.2.1.
    Q: Packed queries. 3d tensor. [N, T_q, d_k].
    K: Packed keys. 3d tensor. [N, T_k, d_k].
    V: Packed values. 3d tensor. [N, T_k, d_v].
    key_masks: A 2d tensor with shape of [N, key_seqlen]
    causality: If True, applies masking for future blinding
    dropout_rate: A floating point number of [0, 1].
    training: boolean for controlling droput
    scope: Optional scope for `variable_scope`.
    '''
    with tf.compat.v1.variable_scope(scope+"dot_product_attention", reuse=tf.compat.v1.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]

        # dot product
        outputs = tf.compat.v1.matmul(Q, tf.compat.v1.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)

        # scale
        outputs /= d_k ** 0.5

        # key masking
        if not no_mask:
            outputs = mask(outputs, key_masks=key_masks, type="key")

        # softmax
        softmax_out = tf.compat.v1.nn.softmax(outputs)

        # dropout
        outputs = tf.compat.v1.layers.dropout(softmax_out, rate=dropout_rate, training=training)

        # weighted sum (context vectors)
        outputs = tf.compat.v1.matmul(outputs, V)  # (N, T_q, d_v)

    return outputs, softmax_out

def get_layer_connection(in_txt0, in_mask0, in_txt1, in_mask1, in_txt2, in_mask2, number_head, dropout, line_parameter, fertility=True, training=True):
    """
    3 multi-head attention and loop output for 3
    """
    out = None
    for i in range(3):
        if out is not None: in_txt0 = out
        out = sublayer(in_txt0, in_mask0, in_txt1, in_mask1, in_txt2, in_mask2,
                       number_head=number_head, dropout=dropout, line_parameter=line_parameter, fertility=fertility, training=training)
    out = LayerNorm(out).get_layer_norm()
    return out

def sublayer(slot_domain_encoder, slot_domain_encoder_mask, delex_encoder=None,
             delex_encoder_mask=None, context_encoder=None, context_encoder_mask=None, number_head=16, dropout=0.2, line_parameter=256, fertility=True, training=True):
    if fertility:
        scope_name = 'fertility_'
        no_mask_check = True
    else:
        scope_name = 'state_'
        no_mask_check =False

    slot_domain_encoder = LayerNorm(slot_domain_encoder).get_layer_norm()
    out = multihead_attention(slot_domain_encoder, slot_domain_encoder, slot_domain_encoder, slot_domain_encoder_mask,
                              num_heads=number_head, dropout_rate=dropout, training=training, scope= scope_name + "slot_domain", no_mask=no_mask_check)
    if delex_encoder is not None:
        out = LayerNorm(out).get_layer_norm()
        out = multihead_attention(out, delex_encoder, delex_encoder, delex_encoder_mask, num_heads=number_head,
                                  dropout_rate=dropout, training=training, scope= scope_name + "delex")
        if context_encoder is not None:
            out = LayerNorm(out).get_layer_norm()
            out = multihead_attention(out, context_encoder, context_encoder, context_encoder_mask, num_heads=number_head,
                                      dropout_rate=dropout, training=training, scope= scope_name + "context")
    out = ff(inputs=out, num_units=[line_parameter*4, line_parameter], dropout_rate=dropout, train=training, scope=scope_name+"positionwise_feedforward")
    return out



def ff(inputs, num_units, dropout_rate, train, scope="positionwise_feedforward"):
    '''position-wise feed forward net. See 3.3

    inputs: A 3d tensor with shape of [N, T, C].
    num_units: A list of two integers.
    scope: Optional scope for `variable_scope`.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
        # Inner layer
        outputs = tf.compat.v1.layers.dense(inputs, num_units[0], activation=tf.nn.relu)
        # Outer layer
        dropout_output = tf.compat.v1.layers.dropout(outputs, rate=dropout_rate, training=train)
        outputs = tf.compat.v1.layers.dense(dropout_output, num_units[1])
        outputs = tf.compat.v1.layers.dropout(outputs, rate=dropout_rate, training=train)
        # Residual connection
        outputs += inputs

    return outputs


def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See 5.4 and https://arxiv.org/abs/1512.00567.
    inputs: 3d tensor. [N, T, V], where V is the number of vocabulary.
    epsilon: Smoothing rate.

    For example,

    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1],
       [0, 1, 0],
       [1, 0, 0]],

      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)

    outputs = label_smoothing(inputs)

    with tf.Session() as sess:
        print(sess.run([outputs]))

    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],

       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]
    ```
    '''
    V = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / V)

def noam_opt(d_model, global_step, factor, warmup_steps=4000.):
    '''Noam scheme learning rate decay
    init_lr: initial learning rate. scalar.
    global_step: scalar.
    warmup_steps: scalar. During warmup_steps, learning rate increases
        until it reaches init_lr.
    '''
    step = tf.compat.v1.cast(global_step + 1, dtype=tf.float32)
    return factor * (d_model ** (-0.5) * tf.compat.v1.minimum(step * warmup_steps ** (-1.5), step ** (-0.5)))