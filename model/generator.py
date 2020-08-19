from model.modules import multihead_attention

import tensorflow as tf

class Generator():
    def __init__(self, vocab, w=None, scope_name=None):
        self.scope_name = scope_name
        with tf.compat.v1.variable_scope(self.scope_name, reuse=tf.compat.v1.AUTO_REUSE):
            self.share_weight = False
            self.out_size = vocab
            self.project = tf.compat.v1.keras.layers.Dense(vocab)
            if w is not None:
                self.share_weight = True
                self.project = w

    def get_slot_gen(self, input):
        with tf.compat.v1.variable_scope(self.scope_name, reuse=tf.compat.v1.AUTO_REUSE):
            if self.share_weight:
                return tf.compat.v1.matmul(input, tf.compat.v1.transpose(self.project, [1, 0]))
            else:
                return self.project(input), self.out_size

class PointerGenerator():
    def __init__(self, state_generator, scope_name=None):
        self.state_generator = state_generator
        self.scope_name = scope_name

    def get_point_generator(self, out_states, encoded_context2, encoded_in_domainslots2, contexts, context_mask):
        with tf.compat.v1.variable_scope(self.scope_name, reuse=tf.compat.v1.AUTO_REUSE):
            vocab_attn = self.state_generator.get_slot_gen(out_states) #sate decoder의 output과 vocab 사이즈 만큼 matmul 계
            encoded_context = encoded_context2
            encoded_in_domainslots = encoded_in_domainslots2
            #state decoder의 Output과 입력된 Context의 Multihead attention을 통해서 Context에서 중요한 단어의 Attention
            out, pointer_attn = multihead_attention(out_states, encoded_context, encoded_context,
                                      context_mask,
                                      num_heads=1, dropout_rate=0.0, training=True,
                                      scope='point_generator',
                                      nomalize=False)

            p_vocab = tf.compat.v1.nn.softmax(vocab_attn, axis=-1)#Voca softmax 결과
            #context의 index 값을 뽑아냄
            context_index = tf.compat.v1.broadcast_to(tf.compat.v1.expand_dims(contexts, axis=1),
                                                      [tf.shape(pointer_attn)[0], tf.shape(pointer_attn)[1], tf.shape(pointer_attn)[2]])
            p_context_ptr = tf.compat.v1.zeros_like(p_vocab) #p_vocab zero tensor 생성

            p_context_ptr = self.get_scatter_add(p_context_ptr, context_index, pointer_attn) #vocab zero tensor 안에 context vector의 index 위치 attention된 값을 더해서 넣음
            expanded_pointer_attn = tf.tile(tf.compat.v1.expand_dims(pointer_attn, axis=-1), # context attention값의 shape확장
                                             tf.constant([1, 1, 1, encoded_context.get_shape().as_list()[-1]]))
            context_vec = tf.compat.v1.reduce_sum(tf.compat.v1.broadcast_to(tf.compat.v1.expand_dims(encoded_context, axis=1), # context에 attention 값  곱하고 더하
                                     [tf.shape(expanded_pointer_attn)[0], tf.shape(expanded_pointer_attn)[1],
                                      tf.shape(expanded_pointer_attn)[2], tf.shape(expanded_pointer_attn)[3]]) * expanded_pointer_attn, axis=2)
            p_gen_vec = tf.compat.v1.concat([out_states, encoded_in_domainslots, context_vec], axis=-1)# 논문의  Vgen 값을 구

            pgen = tf.compat.v1.keras.layers.Dense(1)(p_gen_vec) # Vgen의  weight 계산
            vocab_pointer_switches = tf.compat.v1.nn.sigmoid(tf.compat.v1.broadcast_to(pgen,
                                                               [tf.shape(p_context_ptr)[0], tf.shape(p_context_ptr)[1],
                                                                tf.shape(p_context_ptr)[2]])) # 결국 state 생성시 context전체에 중요한 내용 반영하기 위한 비율 결정
            p_out = (1 - vocab_pointer_switches) * p_context_ptr + vocab_pointer_switches * p_vocab


        return tf.compat.v1.log(p_out), p_out.shape[-1]

    def get_scatter_add(self, tensor, indices, updates):
        original_tensor = tensor
        # expand index value from vocab size
        indices = tf.compat.v1.reshape(indices, shape=[-1, tf.shape(indices)[-1]])
        indices_add = tf.compat.v1.expand_dims(tf.range(0, tf.shape(indices)[0], 1)*(tf.shape(tensor)[-1]), axis=-1)
        indices += indices_add

        # resize
        tensor = tf.compat.v1.reshape(tensor, shape=[-1])
        indices = tf.compat.v1.reshape(indices, shape=[-1, 1])
        updates = tf.compat.v1.reshape(updates, shape=[-1])

        #check_
        """
        update = tensor.shape[indices.shape[-1]:]
        res = indices.shape[:-1] + update
        """
        #same Torch scatter_add_
        scatter = tf.compat.v1.tensor_scatter_nd_add(tensor, indices, updates)
        scatter = tf.compat.v1.reshape(scatter, shape=[tf.shape(original_tensor)[0], tf.shape(original_tensor)[1], -1])
        return scatter

