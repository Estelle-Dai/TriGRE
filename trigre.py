from models import *
from helper import *
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from bert_embedding import BertEmbedding



"""
Abbreviations used in variable names:
	Type:  Entity type side informatoin
	ProbY, RelAlias: Relation alias side information
NOTE: View this file with tab size 8.
"""


class RESIDE(Model):

    # 分batch
    def getBatches(self, data, shuffle=True):
        if shuffle: random.shuffle(data)
        for chunk in getChunks(data, self.p.batch_size):
            batch = ddict(list)
            num = 0
            for i, bag in enumerate(chunk):
                batch['X'].append(bag['X'])
                batch['Pos1'].append(bag['Pos1'])
                batch['Pos2'].append(bag['Pos2'])
                batch['Pos3'].append(bag['Pos3'])
                batch['POS'].append(bag['POS'])
                batch['DepEdges'].append(bag['DepEdges'])
                batch['CoEdges'].append(bag['CoEdges'])
                batch['Y'].append(bag['Y'])
                old_num = num
                num += len(bag['X'])
                batch['sent_num'].append([old_num, num, i])
            yield batch



    # 读数据
    def load_data(self):
        data = pickle.load(open(self.p.dataset, 'rb'))
        self.voc2id = data['voc2id']
        self.max_pos = data['max_pos']  # Maximum position distance
        self.num_class = len(data['rel2id'])
        self.num_deLabel = 1

        # 获取词列表
        self.wrd_list = list(self.voc2id.items())
        self.wrd_list.sort(key=lambda x: x[1])
        self.wrd_list, _ = zip(*self.wrd_list)
        self.data = data

    # 预定义
    def add_placeholders(self):
        self.input_x = tf.placeholder(tf.int32, shape=[None, None], name='input_data')  # Tokens ids of sentences
        self.input_y = tf.placeholder(tf.int32, shape=[None, None], name='input_labels')  # Actual relation of the bag
        self.input_pos1 = tf.placeholder(tf.int32, shape=[None, None], name='input_pos1')  # Position ids wrt entity 1
        self.input_pos2 = tf.placeholder(tf.int32, shape=[None, None], name='input_pos2') 		# Position ids wrt entity 2
        self.input_pos3 = tf.placeholder(tf.int32, shape=[None, None], name='input_pos3') 		# Position ids wrt entity 3
        self.input_pos = tf.placeholder(tf.int32, shape=[None, None], name='input_pos')  # tokens
        self.x_len = tf.placeholder(tf.int32, shape=[None], name='input_len')  # Number of words in sentences in a batch
        self.seq_len = tf.placeholder(tf.int32, shape=(), name='seq_len')  # Max number of tokens in sentences in a batch
        self.total_sents = tf.placeholder(tf.int32, shape=(), name='total_sents')  # Total number of sentences in a batch
        self.sent_num = tf.placeholder(tf.int32, shape=[None, 3],
                                       name='sent_num')  # Stores which sentences belong to which bag
        self.de_adj_ind = tf.placeholder(tf.int64, shape=[self.num_deLabel, None, None, 2],
                                         name='de_adj_ind')  # Dependency graph information (Storing only indices and data)
        self.de_adj_data = tf.placeholder(tf.float32, shape=[self.num_deLabel, None, None], name='de_adj_data')
        self.de_adj_ind2 = tf.placeholder(tf.int64, shape=[self.num_deLabel, None, None, 2],
                                         name='de_adj_ind2')  # Coreference graph information (Storing only indices and data)
        self.de_adj_data2 = tf.placeholder(tf.float32, shape=[self.num_deLabel, None, None], name='de_adj_data2')
        self.dropout = tf.placeholder_with_default(self.p.dropout, shape=(),
                                                   name='dropout')  # Dropout used in GCN Layer
        self.rec_dropout = tf.placeholder_with_default(self.p.rec_dropout, shape=(),
                                                       name='rec_dropout')  # Dropout used in Bi-LSTM

    # Padding
    def padData(self, data, seq_len):
        temp = np.zeros((len(data), seq_len), np.int32)
        mask = np.zeros((len(data), seq_len), np.float32)

        for i, ele in enumerate(data):
            temp[i, :len(ele)] = ele[:seq_len]
            mask[i, :len(ele)] = np.ones(len(ele[:seq_len]), np.float32)

        return temp, mask

    # 生成one-hot
    def getOneHot(self, data, num_class, isprob=False):
        temp = np.zeros((len(data), num_class), np.int32)
        for i, ele in enumerate(data):
            for rel in ele:
                if isprob:
                    temp[i, rel - 1] = 1
                else:
                    temp[i, rel] = 1
        return temp

    # Padding
    def pad_dynamic(self, X, pos1, pos2, pos3, pos):
        seq_len = 0

        x_len = np.zeros((len(X)), np.int32)

        for i, x in enumerate(X):
            seq_len = max(seq_len, len(x))
            x_len[i] = len(x)

        x_pad, _ = self.padData(X, seq_len)
        pos1_pad, _ = self.padData(pos1, seq_len)
        pos2_pad, _ = self.padData(pos2, seq_len)
        pos3_pad, _ = self.padData(pos3, seq_len)
        pos_pad, _ = self.padData(pos, seq_len)



        return x_pad, x_len, pos1_pad, pos2_pad, pos3_pad, pos_pad, seq_len

    def create_feed_dict(self, batch, wLabels=True, dtype='train'):  # Where putting dropout for train?
        X, Y, pos1, pos2, pos3, pos ,sent_num = batch['X'], batch['Y'], batch['Pos1'], batch[
            'Pos2'],batch['Pos3'], batch['POS'], batch['sent_num']
        total_sents = len(batch['X'])
        x_pad, x_len, pos1_pad, pos2_pad, pos3_pad, pos_pad, seq_len = self.pad_dynamic(
            X, pos1, pos2, pos3, pos)

        y_hot = self.getOneHot(Y, self.num_class)

        feed_dict = {}
        feed_dict[self.input_x] = np.array(x_pad)
        feed_dict[self.input_pos1] = np.array(pos1_pad)
        feed_dict[self.input_pos2] = np.array(pos2_pad)
        feed_dict[self.input_pos3] = np.array(pos3_pad)
        feed_dict[self.input_pos] = np.array(pos_pad)
        feed_dict[self.x_len] = np.array(x_len)
        feed_dict[self.seq_len] = seq_len
        feed_dict[self.total_sents] = total_sents
        feed_dict[self.sent_num] = sent_num


        if wLabels: feed_dict[self.input_y] = y_hot

        feed_dict[self.de_adj_ind], \
        feed_dict[self.de_adj_data] = self.get_adj(batch['DepEdges'], total_sents, seq_len, self.num_deLabel)
        feed_dict[self.de_adj_ind2], \
        feed_dict[self.de_adj_data2] = self.get_adj(batch['CoEdges'], total_sents, seq_len, self.num_deLabel)

        if dtype != 'train':
            feed_dict[self.dropout] = 1.0
            feed_dict[self.rec_dropout] = 1.0
        else:
            feed_dict[self.dropout] = self.p.dropout
            feed_dict[self.rec_dropout] = self.p.rec_dropout

        return feed_dict

    # Stores the adjacency matrix as indices and data for feeding to TensorFlow
    def get_adj(self, edgeList, batch_size, max_nodes, max_labels):
        max_edges = 0
        for edges in edgeList:
            max_edges = max(max_edges, len(edges))

        adj_mat_ind = np.zeros((max_labels, batch_size, max_edges, 2), np.int64)
        adj_mat_data = np.zeros((max_labels, batch_size, max_edges), np.float32)

        for lbl in range(max_labels):
            for i, edges in enumerate(edgeList):
                in_ind_temp, in_data_temp = [], []
                for j, (src, dest, _, _) in enumerate(edges):
                    adj_mat_ind[lbl, i, j] = (src, dest)
                    adj_mat_data[lbl, i, j] = 1.0

        return adj_mat_ind, adj_mat_data

    # GCN Layer Implementation
    def GCNLayer(self, gcn_in,  # Input to GCN Layer
                 in_dim,  # Dimension of input to GCN Layer
                 gcn_dim,  # Hidden state dimension of GCN
                 batch_size,  # Batch size
                 max_nodes,  # Maximum number of nodes in graph
                 max_labels,  # Maximum number of edge labels
                 adj_ind,  # Adjacency matrix indices
                 adj_data,  # Adjacency matrix data (all 1's)
                 w_gating=True,  # Whether to include gating in GCN
                 num_layers=1,  # Number of GCN Layers
                 name="GCN"):
        out = []
        out.append(gcn_in)

        for layer in range(num_layers):
            gcn_in = out[
                -1]  # out contains the output of all the GCN layers, intitally contains input to first GCN Layer
            if len(out) > 1: in_dim = gcn_dim  # After first iteration the in_dim = gcn_dim

            with tf.name_scope('%s-%d' % (name, layer)):
                act_sum = tf.zeros([batch_size, max_nodes, gcn_dim])
                for lbl in range(max_labels):

                    # Defining the layer and label specific parameters
                    with tf.variable_scope('label-%d_name-%s_layer-%d' % (lbl, name, layer)) as scope:
                        w_in = tf.get_variable('w_in', [in_dim, gcn_dim],
                                               initializer=tf.keras.initializers.glorot_normal(),
                                               regularizer=self.regularizer)
                        w_out = tf.get_variable('w_out', [in_dim, gcn_dim],
                                                initializer=tf.keras.initializers.glorot_normal(),
                                                regularizer=self.regularizer)
                        w_loop = tf.get_variable('w_loop', [in_dim, gcn_dim],
                                                 initializer=tf.keras.initializers.glorot_normal(),
                                                 regularizer=self.regularizer)
                        b_in = tf.get_variable('b_in', initializer=np.zeros([1, gcn_dim]).astype(np.float32),
                                               regularizer=self.regularizer)
                        b_out = tf.get_variable('b_out', initializer=np.zeros([1, gcn_dim]).astype(np.float32),
                                                regularizer=self.regularizer)

                        if w_gating:
                            w_gin = tf.get_variable('w_gin', [in_dim, 1],
                                                    initializer=tf.keras.initializers.glorot_normal(),
                                                    regularizer=self.regularizer)
                            w_gout = tf.get_variable('w_gout', [in_dim, 1],
                                                     initializer=tf.keras.initializers.glorot_normal(),
                                                     regularizer=self.regularizer)
                            w_gloop = tf.get_variable('w_gloop', [in_dim, 1],
                                                      initializer=tf.keras.initializers.glorot_normal(),
                                                      regularizer=self.regularizer)
                            b_gin = tf.get_variable('b_gin', initializer=np.zeros([1]).astype(np.float32),
                                                    regularizer=self.regularizer)
                            b_gout = tf.get_variable('b_gout', initializer=np.zeros([1]).astype(np.float32),
                                                     regularizer=self.regularizer)

                    # Activation from in-edges
                    with tf.name_scope('in_arcs-%s_name-%s_layer-%d' % (lbl, name, layer)):
                        inp_in = tf.tensordot(gcn_in, w_in, axes=[2, 0]) + tf.expand_dims(b_in, axis=0)

                        def map_func1(i):
                            adj_mat = tf.SparseTensor(adj_ind[lbl, i], adj_data[lbl, i],
                                                      [tf.cast(max_nodes, tf.int64), tf.cast(max_nodes, tf.int64)])
                            adj_mat = tf.sparse_transpose(adj_mat)
                            return tf.sparse_tensor_dense_matmul(adj_mat, inp_in[i])

                        in_t = tf.map_fn(map_func1, tf.range(batch_size), dtype=tf.float32)

                        if self.p.dropout != 1.0: in_t = tf.nn.dropout(in_t, keep_prob=self.p.dropout)

                        if w_gating:
                            inp_gin = tf.tensordot(gcn_in, w_gin, axes=[2, 0]) + tf.expand_dims(b_gin, axis=0)

                            def map_func2(i):
                                adj_mat = tf.SparseTensor(adj_ind[lbl, i], adj_data[lbl, i],
                                                          [tf.cast(max_nodes, tf.int64), tf.cast(max_nodes, tf.int64)])
                                adj_mat = tf.sparse_transpose(adj_mat)
                                return tf.sparse_tensor_dense_matmul(adj_mat, inp_gin[i])

                            in_gate = tf.map_fn(map_func2, tf.range(batch_size), dtype=tf.float32)
                            in_gsig = tf.sigmoid(in_gate)
                            in_act = in_t * in_gsig
                        else:
                            in_act = in_t

                    # Activation from out-edges
                    with tf.name_scope('out_arcs-%s_name-%s_layer-%d' % (lbl, name, layer)):
                        inp_out = tf.tensordot(gcn_in, w_out, axes=[2, 0]) + tf.expand_dims(b_out, axis=0)

                        def map_func3(i):
                            adj_mat = tf.SparseTensor(adj_ind[lbl, i], adj_data[lbl, i],
                                                      [tf.cast(max_nodes, tf.int64), tf.cast(max_nodes, tf.int64)])
                            return tf.sparse_tensor_dense_matmul(adj_mat, inp_out[i])

                        out_t = tf.map_fn(map_func3, tf.range(batch_size), dtype=tf.float32)
                        if self.p.dropout != 1.0: out_t = tf.nn.dropout(out_t, keep_prob=self.p.dropout)

                        if w_gating:
                            inp_gout = tf.tensordot(gcn_in, w_gout, axes=[2, 0]) + tf.expand_dims(b_gout, axis=0)

                            def map_func4(i):
                                adj_mat = tf.SparseTensor(adj_ind[lbl, i], adj_data[lbl, i],
                                                          [tf.cast(max_nodes, tf.int64), tf.cast(max_nodes, tf.int64)])
                                return tf.sparse_tensor_dense_matmul(adj_mat, inp_gout[i])

                            out_gate = tf.map_fn(map_func4, tf.range(batch_size), dtype=tf.float32)
                            out_gsig = tf.sigmoid(out_gate)
                            out_act = out_t * out_gsig
                        else:
                            out_act = out_t

                    # Activation from self-loop
                    with tf.name_scope('self_loop'):
                        inp_loop = tf.tensordot(gcn_in, w_loop, axes=[2, 0])
                        if self.p.dropout != 1.0: inp_loop = tf.nn.dropout(inp_loop, keep_prob=self.p.dropout)

                        if w_gating:
                            inp_gloop = tf.tensordot(gcn_in, w_gloop, axes=[2, 0])
                            loop_gsig = tf.sigmoid(inp_gloop)
                            loop_act = inp_loop * loop_gsig
                        else:
                            loop_act = inp_loop

                    # Aggregating activations
                    act_sum += in_act + out_act + loop_act

                gcn_out = tf.nn.relu(act_sum)
                out.append(gcn_out)

        return out
        
    def normalize(self,inputs, epsilon = 1e-8,scope="ln",reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]
            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.Variable(tf.zeros(params_shape),dtype=tf.float32)
            gamma = tf.Variable(tf.ones(params_shape),dtype=tf.float32)
            normalized = (inputs - mean) / ((variance + epsilon) ** (0.5))
            outputs = gamma * normalized + beta
        return outputs

    def self_attention(self, keys, num_units, scope='multihead_attention', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            Q = tf.nn.relu(tf.layers.dense(keys, num_units, kernel_initializer=tf.keras.initializers.glorot_normal()))
            K = tf.nn.relu(tf.layers.dense(keys, num_units, kernel_initializer=tf.keras.initializers.glorot_normal()))
            V = tf.nn.relu(tf.layers.dense(keys, num_units, kernel_initializer=tf.keras.initializers.glorot_normal()))
            Q_ = tf.concat(tf.split(Q, self.p.num_heads, axis=2), axis=0)
            K_ = tf.concat(tf.split(K, self.p.num_heads, axis=2), axis=0)
            V_ = tf.concat(tf.split(V, self.p.num_heads, axis=2), axis=0)
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
            key_masks = tf.tile(key_masks, [self.p.num_heads, 1])
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(keys)[1], 1])
            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)
            outputs = tf.nn.softmax(outputs)
            query_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
            query_masks = tf.tile(query_masks, [self.p.num_heads, 1])
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])
            outputs *= query_masks
            outputs = tf.nn.dropout(outputs, keep_prob=self.p.dropout)
            outputs = tf.matmul(outputs, V_)
            outputs = tf.concat(tf.split(outputs, self.p.num_heads, axis=0), axis=2)
            outputs += keys
            outputs = self.normalize(outputs)
        return outputs

    def add_model(self):
        in_wrds, in_pos1, in_pos2, in_pos3,in_pos = self.input_x, self.input_pos1, self.input_pos2,self.input_pos3, self.input_pos

        with tf.variable_scope('Embeddings') as scope:
            # model = gensim.models.KeyedVectors.load_word2vec_format(self.p.embed_loc, binary=False)
            # embed_init = getEmbeddings(model, self.wrd_list, 200)
            # _wrd_embeddings = tf.get_variable('embeddings', initializer=embed_init, trainable=True,
            #                                   regularizer=self.regularizer)
            # wrd_pad = tf.zeros([1, 200])
            # wrd_embeddings = tf.concat([wrd_pad, _wrd_embeddings], axis=0)

            # bert
            bert_embedding = BertEmbedding()
            embed_init_list = []

            for wrd in self.wrd_list:
                sent = []
                sent.append(wrd)
                wrd_emb = bert_embedding(sent)
                embed_init_list.append(wrd_emb[0][1][0])
            
            embed_init = np.array(embed_init_list)
            _wrd_embeddings = tf.get_variable('embeddings', initializer=embed_init, trainable=True,
                                              regularizer=self.regularizer)
            wrd_pad = tf.zeros([1, self.p.embed_dim])
            wrd_embeddings = tf.concat([wrd_pad, _wrd_embeddings], axis=0)



            pos1_embeddings = tf.get_variable('pos1_embeddings', [300, self.p.pos_dim],
                                              initializer=tf.keras.initializers.glorot_normal(), trainable=True,
                                              regularizer=self.regularizer)
            pos2_embeddings = tf.get_variable('pos2_embeddings', [300, self.p.pos_dim],
                                              initializer=tf.keras.initializers.glorot_normal(), trainable=True,
                                              regularizer=self.regularizer)
            pos3_embeddings = tf.get_variable('pos3_embeddings', [300, self.p.pos_dim],
                                              initializer=tf.keras.initializers.glorot_normal(), trainable=True,
                                              regularizer=self.regularizer)
            pos_embeddings = tf.get_variable('POS_embeddings', [42, self.p.POS_dim],
                                              initializer=tf.keras.initializers.glorot_normal(), trainable=True,
                                              regularizer=self.regularizer)
            



        wrd_embed = tf.nn.embedding_lookup(wrd_embeddings, in_wrds)
        pos1_embed = tf.nn.embedding_lookup(pos1_embeddings, in_pos1)
        pos2_embed = tf.nn.embedding_lookup(pos2_embeddings, in_pos2)
        pos3_embed = tf.nn.embedding_lookup(pos3_embeddings, in_pos3)
        pos_embed = tf.nn.embedding_lookup(pos_embeddings, in_pos)

        
        embeds = tf.concat([wrd_embed, pos1_embed, pos2_embed, pos3_embed,pos_embed], axis=2)

        # bilstm
        with tf.variable_scope('Bi-LSTM') as scope:
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.p.lstm_dim, name='FW_GRU'),
                                                    output_keep_prob=self.rec_dropout)
            bk_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(self.p.lstm_dim, name='BW_GRU'),
                                                    output_keep_prob=self.rec_dropout)
            val, state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bk_cell, embeds, sequence_length=self.x_len,
                                                         dtype=tf.float32)
           
            lstm_out = tf.concat((val[0], val[1]), axis=2)
            # print(lstm_out)
            # print("*"*55)
            lstm_out_dim = self.p.lstm_dim * 2

        # 句法依赖图
        de_out1 = self.GCNLayer(gcn_in = lstm_out, 		in_dim 	= lstm_out_dim,  gcn_dim=self.p.de_gcn_dim,
                                batch_size = self.total_sents, 	max_nodes   = self.seq_len, 		max_labels = self.num_deLabel,
                                 adj_ind 	= self.de_adj_ind, 	adj_data    = self.de_adj_data, 	w_gating   = self.p.wGate,
                                 num_layers 	= self.p.de_layers, 	name 	    = "GCN_DE")
        de_out1 	   = de_out1[-1]

        # 实体共现图
        de_out2 = self.GCNLayer(gcn_in = lstm_out, 		in_dim 	= lstm_out_dim,  gcn_dim=self.p.de_gcn_dim,
                                batch_size = self.total_sents, 	max_nodes   = self.seq_len, 		max_labels = self.num_deLabel,
                                 adj_ind 	= self.de_adj_ind2, 	adj_data    = self.de_adj_data2, 	w_gating   = self.p.wGate,
                                 num_layers 	= self.p.de_layers, 	name 	    = "GCN_DE2")
        de_out2 	   = de_out2[-1]

        # attention连接g2
        with tf.variable_scope('self_attention_g1') as scope:
            g1 = self.self_attention(de_out1,self.p.de_gcn_dim)

        # attention连接g2
        with tf.variable_scope('self_attention_g2') as scope:
            g2 = self.self_attention(de_out2,self.p.de_gcn_dim)

        # attention连接长序列
        with tf.variable_scope('self_attention_lstm') as scope:
            h = self.self_attention(lstm_out,lstm_out_dim)


        with tf.variable_scope('interaction') as scope:
            weight=tf.Variable(tf.random_normal([self.p.de_gcn_dim, self.p.de_gcn_dim]), name='aspect_w')

            bias = tf.Variable(tf.zeros([self.p.de_gcn_dim]), name='aspect_b')

            aspect_avg1 = tf.reduce_mean(de_out1, 1)
            aspect_avg2 = tf.reduce_mean(de_out2, 1)

            print('aspect_avg.get_shape():' ,aspect_avg1.shape)
            print('aspect_avg.get_shape():' ,aspect_avg2.shape)

            context_att1 = tf.nn.softmax(tf.nn.tanh(tf.einsum('ijk,kl,ilm->ijm', g1, weight,
                                                         tf.expand_dims(aspect_avg2, -1)) + bias))
            context_att2 = tf.nn.softmax(tf.nn.tanh(tf.einsum('ijk,kl,ilm->ijm', g2, weight,
                                                         tf.expand_dims(aspect_avg1, -1)) + bias))

            context_rep1 = tf.reduce_sum(context_att1 * g1, 1)
            context_rep2 = tf.reduce_sum(context_att2 * g2, 1)
            context_12 = tf.concat((context_rep1, context_rep2),axis=-1)

            weight_lstm=tf.Variable(tf.random_normal([lstm_out_dim, lstm_out_dim]), name='aspect_wl')
            bias_lstm = tf.Variable(tf.zeros([lstm_out_dim]), name='aspect_bl')
            context_att = tf.nn.softmax(tf.nn.tanh(tf.einsum('ijk,kl,ilm->ijm', h, weight_lstm,
                                                         tf.expand_dims(context_12, -1)) + bias_lstm))

            context_rep = tf.reduce_sum(context_att * h, 1)



        with tf.variable_scope('FC1') as scope:
            w_rel = tf.get_variable('w_rel', [lstm_out_dim, self.num_class],
                                    initializer=tf.keras.initializers.glorot_normal(), regularizer=self.regularizer)
            b_rel = tf.get_variable('b_rel', initializer=np.zeros([self.num_class]).astype(np.float32),
                                    regularizer=self.regularizer)
            nn_out = tf.nn.xw_plus_b(context_rep, w_rel, b_rel)

        with tf.name_scope('Accuracy') as scope:
            prob     = tf.nn.softmax(nn_out)
            y_pred   = tf.argmax(prob, 	   axis=1)
            y_actual = tf.argmax(self.input_y, axis=1)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_actual), tf.float32))

        return nn_out, accuracy


    def add_loss(self, nn_out):
        with tf.name_scope('Loss_op'):
            loss  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=nn_out, labels=self.input_y))
        return loss

    def add_optimizer(self, loss):
        with tf.name_scope('Optimizer'):
            if self.p.opt == 'adam' and not self.p.restore:
                optimizer = tf.train.AdamOptimizer(self.p.lr)
            else:
                optimizer = tf.train.GradientDescentOptimizer(self.p.lr)
            train_op  = optimizer.minimize(loss)
        return train_op

    def __init__(self, params):
        self.p  = params
        self.logger = get_logger(self.p.name, self.p.log_dir, self.p.config_dir)

        self.logger.info(vars(self.p)); pprint(vars(self.p))
        self.p.batch_size = self.p.batch_size

        if self.p.l2 == 0.0: 	self.regularizer = None
        else: 			self.regularizer = tf.keras.regularizers.l2(self.p.l2)

        self.load_data()
        self.add_placeholders()

        nn_out, self.accuracy = self.add_model()
        
        self.loss      	= self.add_loss(nn_out)
        self.logits  	= tf.nn.softmax(nn_out)
        self.train_op   = self.add_optimizer(self.loss)

        tf.summary.scalar('accmain', self.accuracy)
        self.merged_summ = tf.summary.merge_all()
        self.summ_writer = None

    # Evaluate model on valid/test data
    def predict_test(self, sess, data, wLabels=True, shuffle=False, label='Evaluating on Test'):
        losses, accuracies, results, y_pred, y, logit_list, y_actual_hot = [], [], [], [], [], [], []
        bag_cnt = 0

        for step, batch in enumerate(self.getBatches(data, shuffle)):

            loss, logits, accuracy = sess.run([self.loss, self.logits, self.accuracy], feed_dict = self.create_feed_dict(batch, dtype='test'))
            losses.    append(loss)
            accuracies.append(accuracy)

            pred_ind      = logits.argmax(axis=1)
            logit_list   += logits.tolist()
            y_actual_hot += self.getOneHot(batch['Y'], self.num_class).tolist()
            y_pred       += pred_ind.tolist()
            y 	     += np.argmax(self.getOneHot(batch['Y'], self.num_class), 1).tolist()
            bag_cnt      += len(batch['sent_num'])

            results.append(pred_ind)

            if step % 100 == 0:
                self.logger.info('{} ({}/{}):\t{:.5}\t{:.5}\t{}'.format(label, bag_cnt, len(self.data['test']), np.mean(accuracies ) *100, np.mean(losses), self.p.name))
                print('{} ({}/{}):\t{:.5}\t{:.5}\t{}'.format(label, bag_cnt, len(self.data['test']), np.mean(accuracies ) *100, np.mean(losses), self.p.name))
        self.logger.info('Test Accuracy: {}'.format(np.mean(accuracies ) *100))
        print('Test Accuracy: {}'.format(np.mean(accuracies ) *100))

        return np.mean(losses), results,  np.mean(accuracies ) *100, y, y_pred, logit_list, y_actual_hot
    
    # Runs one epoch of training
    def run_epoch(self, sess, data, epoch, shuffle=True):
        losses, accuracies = [], []
        bag_cnt = 0

        for step, batch in enumerate(self.getBatches(data, shuffle)):
            feed = self.create_feed_dict(batch)
            summary_str, loss, accuracy, _ = sess.run([self.merged_summ,self.loss, self.accuracy, self.train_op], feed_dict=feed)
            #print(printmax.shape)
            losses.    append(loss)
            accuracies.append(accuracy)

            bag_cnt += len(batch['sent_num'])

            if step % 10 == 0:
                self.logger.info('E:{} Train Accuracy ({}/{}):\t{:.5}\t{:.5}\t{}\t{:.5}'.format(epoch, bag_cnt, len(self.data['train']), np.mean
                                                                                                    (accuracies ) *100, np.mean(losses), self.p.name, self.best_train_acc))
                print('E:{} Train Accuracy ({}/{}):\t{:.5}\t{:.5}\t{}\t{:.5}'.format(epoch, bag_cnt, len(self.data['train']), np.mean
                                                                                                    (accuracies ) *100, np.mean(losses), self.p.name, self.best_train_acc))
                self.summ_writer.add_summary(summary_str, epoch *len(self.data['train']) + bag_cnt)

        accuracy = np.mean(accuracies) * 100.0
        self.logger.info('Training Loss:{}, Accuracy: {}'.format(np.mean(losses), accuracy))
        print('Training Loss:{}, Accuracy: {}'.format(np.mean(losses), accuracy))
        return np.mean(losses), accuracy



    def result_evaluation(self, y_test, pred_test):

        pred_matrix = np.zeros((len(pred_test) ,2) ,dtype=np.int8)
        y_matrix = np.zeros((len(pred_test) ,2) ,dtype=np.int8)


        for i in range(len(y_test)):
            pred_matrix[i][pred_test[i]] = 1
            y_matrix[i][y_test[i]] = 1

        count_matrix =np.zeros((2 ,3))
        for class_idx in range(1 ,2):

            count_matrix[class_idx][0] = np.sum \
                (np.array(pred_matrix[:, class_idx]) * np.array(y_matrix[:, class_idx])  )  # tp
            count_matrix[class_idx][1] = np.sum \
                (np.array(pred_matrix[:, class_idx]) * (1 - np.array(y_matrix[:, class_idx]))  )  # fp
            count_matrix[class_idx][2] = np.sum \
                ((1 - np.array(pred_matrix[:, class_idx])) * np.array(y_matrix[:, class_idx])  )  # fn

        sumtp = sumfp = sumfn =0

        for i in range(1 ,2):
            sumtp +=count_matrix[i][0]
            sumfp +=count_matrix[i][1]
            sumfn +=count_matrix[i][2]

        precision = recall = f1 =0

        if (sumtp + sumfp) == 0:
            precision = 0.
        else:
            precision = float(sumtp) / (sumtp + sumfp)

        if (sumtp + sumfn) == 0:
            recall = 0.
        else:
            recall = float(sumtp) / (sumtp + sumfn)

        if (precision + recall) == 0.:
            f1 = 0.
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return precision ,recall ,f1
 

    # Trains the model and finally evaluates on test
    def fit(self, sess):
        self.summ_writer = tf.summary.FileWriter('tf_board/{}'.format(self.p.name), sess.graph)
        saver     = tf.train.Saver(max_to_keep=4)
        save_dir  = 'checkpoints/{}/'.format(self.p.name); make_dir(save_dir)
        res_dir   = 'results/{}/'.format(self.p.name);     make_dir(res_dir)
        save_path = os.path.join(save_dir, 'best_model')

        # Restore previously trained model
        if self.p.restore:
            saver.restore(sess, save_path)
        self.f1, self.best_train_acc = 0.0, 0.0
      


        if not self.p.only_eval:
            for epoch in range(self.p.max_epochs):
                train_loss, train_acc = self.run_epoch(sess, self.data['train'], epoch)
                #train_loss, train_acc = self.run_epoch(sess, self.data['train'], epoch)
                self.logger.info \
                      ('[Epoch {}]: Training Loss: {:.5}, Training Acc: {:.5}\n'.format(epoch, train_loss, train_acc))
                # val_loss, val_pred, val_acc, y, y_pred, logit_list, y_hot = self.predict_valid(sess,self.data['valid'])
                test_loss, test_pred, test_acc, y, y_pred, logit_list, y_hot = self.predict_test(sess, self.data['test'])
                print("实际：")
                print(y)
                print("预测：")
                print(y_pred)
                test_prec, test_rec, test_f1  = self.result_evaluation(y, y_pred)
                print("!!!!")
                print(test_acc,test_prec,test_rec,test_f1)
                print("!!!!")
                    # self.logger.info('Final results: Prec:{} | Rec:{} | F1:{}'.format(test_prec, test_rec, test_f1))
                # Store the model with least train loss
                if test_acc > self.best_train_acc:
                    self.best_train_acc   = test_acc
                    saver.save(sess=sess, save_path=save_path)
                # self.logger.info('[Epoch {}]: Training Loss: {:.5}, Training Acc: {:.5}, Valid Loss: {:.5}, Valid Acc: {:.5} Best Acc: {:.5}\n'.format(epoch, train_loss, train_acc, val_loss, val_acc, self.best_val_acc))

        self.logger.info('Running on Test set')
        saver.restore(sess, save_path)
        test_loss, test_pred, test_acc, y, y_pred, logit_list, y_hot = self.predict_test(sess, self.data['test'])
        self.logger.info('Final results: test_acc:{}'.format(test_acc))
        # test_loss, test_pred, test_acc_1, y, y_pred, logit_list, y_hot = self.predict_test(sess, self.data['test2'])
        # self.logger.info('Final results: test_acc_1:{}'.format(test_acc_1))
        # test_loss, test_pred, test_acc_2, y, y_pred, logit_list, y_hot = self.predict_test(sess, self.data['test3'])
        # self.logger.info('Final results: test_acc_2:{}'.format(test_acc_2))
        #self.logger.info('Average results:Prec:{:.5} | Rec:{:.5} | F1:{:.5}'.format(np.mean(np.array(Prec_score))*100,np.mean(np.array(Rec_score))*100,np.mean(np.array(F1_score))*100))

    
if __name__== "__main__":

    parser = argparse.ArgumentParser \
        (description='DDI Relation Extraction')

    parser.add_argument('-data', 	 dest="dataset", 	required=True,							help='Dataset to use')
    parser.add_argument('-gpu', 	 dest="gpu", 		default='0',							help='GPU to use')
    parser.add_argument('-nGate', 	 dest="wGate", 		action='store_false',   					help='Include edgewise-gating in GCN')
    parser.add_argument('-lstm_dim', dest="lstm_dim", 	default=200,   	type=int, 					help='Hidden state dimension of Bi-LSTM')
    parser.add_argument('-pos_dim',  dest="pos_dim", 	default=50, 			type=int, 			help='Dimension of positional embeddings')
    parser.add_argument('-POS_dim', dest="POS_dim", default=50, type=int, help='Dimension of cui embeddings')
    parser.add_argument('-de_dim',   dest="de_gcn_dim", 	default=200,   			type=int, 			help='Hidden state dimension of GCN over dependency tree')
    parser.add_argument('-de_layer', dest="de_layers", 	default=1,   			type=int, 			help='Number of layers in GCN over dependency tree')
    parser.add_argument('-drop',	 dest="dropout", 	default=0.6,  			type=float,			help='Dropout for full connected layer')
    parser.add_argument('-rdrop',	 dest="rec_dropout", 	default=0.6,  			type=float,			help='Recurrent dropout for LSTM')
    parser.add_argument('-num_units', dest="num_units", 	default=200,   			type=int, 			help='Number of self_attention')
    parser.add_argument('-num_heads', dest="num_heads", 	default=8,   			type=int, 			help='Number of head')
    parser.add_argument('-lr',	 dest="lr", 		default=0.001,  		type=float,			help='Learning rate')
    parser.add_argument('-rho',	 dest="rho", 		default=0.95,  		type=float,			help='Learning rate')
    parser.add_argument('-epsilon',	 dest="epsilon", 		default=1e-08,  		type=float,			help='Learning rate')
    parser.add_argument('-l2', 	 dest="l2", 		default=0.001,  		type=float, 			help='L2 regularization')
    parser.add_argument('-epoch', 	 dest="max_epochs", 	default=50,   			type=int, 			help='Max epochs')
    parser.add_argument('-batch', 	 dest="batch_size", 	default=20,   			type=int, 			help='Batch size')
    parser.add_argument('-chunk', 	 dest="chunk_size", 	default=1000,   		type=int, 			help='Chunk size')
    parser.add_argument('-restore',	 dest="restore", 	action='store_true', 						help='Restore from the previous best saved model')
    parser.add_argument('-only_eval' ,dest="only_eval", 	action='store_true', 						help='Only Evaluate the pretrained model (skip training)')
    parser.add_argument('-opt',	 dest="opt", 		default='adam', 						help='Optimizer to use for training')
    parser.add_argument('-name', 	 dest="name", 		default='test_ ' +str(uuid.uuid4()),				help='Name of the run')
    parser.add_argument('-seed', 	 dest="seed", 		default=1234, 			type=int,			help='Seed for randomization')
    parser.add_argument('-logdir',	 dest="log_dir", 	default='./log/', 						help='Log directory')
    parser.add_argument('-config',	 dest="config_dir", 	default='./config/', 						help='Config directory')
    parser.add_argument('-embed_loc',dest="embed_loc", 	default='./glove/glove.6B.200d_word2vec.txt', 			help='Log directory')
    parser.add_argument('-embed_dim' ,dest="embed_dim", 	default=768, type=int,						help='Dimension of embedding')
    args = parser.parse_args()

    if not args.restore: args.name = args.name
    
    # Set GPU to use
    set_gpu(args.gpu)

    # Set seed
    tf.set_random_seed(args.seed)
    #random.seed(args.seed)
    np.random.seed(args.seed)

    # Create model computational graph
    model  = RESIDE(args)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth =True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        model.fit(sess)
