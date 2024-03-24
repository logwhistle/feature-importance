import numpy as np
import tensorflow as tf
from tensorflow.python.ops import state_ops

'''1. 构造模型, 模型由5个sparse特征及20个dense特征组成'''
np.random.seed(0)
hidden_size = 16

with tf.variable_scope('models', reuse=tf.AUTO_REUSE) as scope:

    # 5 sparse features, each size is 4. take dense tensor for example, which is sparse tensor in real world
    fe0 = tf.placeholder(dtype=tf.float32, name='sparse_feature0')
    fe1 = tf.placeholder(dtype=tf.float32, name='sparse_feature1')
    fe2 = tf.placeholder(dtype=tf.float32, name='sparse_feature2')
    fe3 = tf.placeholder(dtype=tf.float32, name='sparse_feature3')
    fe4 = tf.placeholder(dtype=tf.float32, name='sparse_feature4')

    # 20 dense features

    dfe = tf.placeholder(dtype=tf.float32, name='dense_feature')
    label1 = tf.placeholder(dtype=tf.float32, name='label1')
    label2 = tf.placeholder(dtype=tf.float32, name='label2')
    inputs = tf.concat([fe0, fe1, fe2, fe3, fe4, dfe], 1)

    # layer 1
    # input_shape = inputs.shape[-1].value
    w0 = tf.get_variable(name='weight', initializer=tf.variance_scaling_initializer, shape=[40, hidden_size])
    b0 = tf.get_variable(name='bias', initializer=tf.zeros_initializer(), shape=[hidden_size])
    h1 = tf.nn.elu(tf.add(tf.tensordot(inputs, w0, axes=1), b0))

    # MTL layer
    w_l1 = tf.get_variable(name='weight_label1', initializer=tf.variance_scaling_initializer, shape=[hidden_size, 1])
    b_l1 = tf.get_variable(name='bias_label1', initializer=tf.zeros_initializer(), shape=[1])
    y1 = tf.add(tf.tensordot(h1, w_l1, axes=1), b_l1)
    w_l2 = tf.get_variable(name='weight_label2', initializer=tf.variance_scaling_initializer, shape=[hidden_size, 1])
    b_l2 = tf.get_variable(name='bias_label2', initializer=tf.zeros_initializer(), shape=[1])
    y2 = tf.add(tf.tensordot(h1, w_l2, axes=1), b_l2) # no tf.nn.sigmoid here, which will do in sigmoid_cross_entropy_with_logits
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label1, logits=y1, name='l1_loss') + tf.nn.sigmoid_cross_entropy_with_logits(labels=label2, logits=y2, name='l2_loss'))


'''2. 模型训练op'''
opt = tf.train.GradientDescentOptimizer(0.01)
tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='models/')
grad_var = opt.compute_gradients(loss, var_list=tvars)
global_step = tf.Variable(0, trainable=False)
train_op = opt.apply_gradients(grad_var, global_step=global_step)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


'''3. 构造训练数据集, 不能和placeholder的key重名, 类型一致, shape一致'''
# set batch_size = 32, features dim = sparse_features_dim + dense_features_dim
feed_dict = {}

# 遍历生成5个sparse features
for i in range(5):
    globals()['fe%s_data'%i] = (i+1)/2 * np.random.random([33, 4]) + i + 1
    globals()['fe%s_data'%i] = globals()['fe%s_data'%i].astype(np.float32)
    feed_dict[globals()['fe%s'%i]] = globals()['fe%s_data'%i][1:33, :]

# 一次性生成20个dense features
dfe_data = 3 * np.random.random([33, 20]) + 2
dfe_data = dfe_data.astype(np.float32)
feed_dict[dfe] = dfe_data[1:33, :]

input = np.concatenate(list(feed_dict.values()), 1)
l1 = np.round(1 / (1 + np.mean(-input * input, axis=1, keepdims=True)))
l1 = l1.astype(np.float32)

l2 = np.round(1 / (1 + np.mean(-2 * input ** 2 + 0.5, axis=1, keepdims=True)))
l2 = l2.astype(np.float32)

feed_dict[label1] = l1
feed_dict[label2] = l2


'''4. 构造测试集, 不能和placeholder的key重名, 类型一致, shape一致'''
# generate test data
feed_dict_test = {}

for i in range(5):
    globals()['fe%s_test'%i] = globals()['fe%s_data'%i][0:1, :]
    feed_dict_test[globals()['fe%s'%i]] = globals()['fe%s_test'%i]

# 一次性生成20个dense features
dfe_test = dfe_data[0:1, :]
feed_dict_test[dfe] = dfe_test
input_test = np.concatenate(list(feed_dict_test.values()), 1)
feed_dict_test[label1] = (np.round(1 / (1 + np.mean(-input_test * input_test, axis=1, keepdims=True)))).astype(np.float32)
feed_dict_test[label2] = (np.round(1 / (1 + np.mean(-2 * input_test ** 2 + 0.5, axis=1, keepdims=True)))).astype(np.float32)

l = sess.run([loss], feed_dict=feed_dict_test)


'''5. 特征重要性分析前: a.打印训练前loss b.训练45轮 c.打印训练后的loss'''
print('before train', l)
for i in range(45):
    _, l, step = sess.run([train_op, loss, global_step], feed_dict=feed_dict)                             
    print(l, step)

l = sess.run([loss], feed_dict=feed_dict_test)
print('after train', l)


'''6. 特征重要性分析, 方法一: shuffle方法  注意dense features的shuffle粒度为1, 而sparse features的shuffle粒度是embedding size'''
def shuffle(dense_tensor, size=20, prob=1.0, mod='dense', sess=None):
    # dense_tensor = 3 * np.random.random([4, 5]) + 2
    row_num = tf.cast(tf.shape(dense_tensor)[0], dtype=tf.int32)
    col_num = tf.cast(tf.shape(dense_tensor)[1], dtype=tf.int32)

    def process_col():
        base_range = tf.range(0, row_num, dtype=tf.int32)
        random_num = tf.random.uniform(shape=(row_num,))
        mask = tf.less_equal(random_num, tf.constant(prob))
        idx_masked = tf.cast(tf.where(mask), dtype=tf.int32)
        idx_masked_shuffled = tf.random.shuffle(tf.squeeze(idx_masked, 1))
        idx_masked_shuffled_scat = tf.scatter_nd(idx_masked, idx_masked_shuffled, [row_num])
        idx_shuffled = tf.where(mask, idx_masked_shuffled_scat, base_range)
        return tf.gather(base_range, idx_shuffled)

    if mod == 'dense':
        random_index_mapping = tf.map_fn(fn=lambda d: process_col(), elems=tf.range(0, col_num))
        temp = []
        for num in range(size):
            temp.append(tf.gather(dense_tensor[:, num], random_index_mapping[num]))
        shuffled_tensor = tf.stack(temp, axis=1)

    elif mod == 'sparse':
        # sparse通常为sparse tensor，本demo为了简化所以均用dense tensor，此处为了适配增加针对sparse tensor的方法
        sparse_idx = tf.where(tf.not_equal(dense_tensor, 0))
        sparse_tensor = tf.SparseTensor(sparse_idx, tf.gather_nd(dense_tensor, sparse_idx), [row_num, col_num])
        random_index_mapping = tf.map_fn(fn=lambda d: process_col(), elems=tf.range(0, col_num))
        shuffled_index = tf.map_fn(fn=lambda d: tf.concat([[random_index_mapping[d[1], d[0]]], d[1:]], axis=0), elems=sparse_tensor.indices)
        shuffled_tensor_convert = tf.sparse.reorder(tf.SparseTensor(shuffled_index, sparse_tensor.values, sparse_tensor.dense_shape))
        # shuffled_tensor = tf.sparse_to_dense(shuffled_tensor_convert.indices, sparse_tensor.dense_shape, sparse_tensor.values)
        shuffled_tensor = tf.sparse.to_dense(shuffled_tensor_convert)
        if sess != None:
            print(sess.run([shuffled_index, sparse_tensor.values, sparse_tensor.dense_shape, shuffled_tensor]))
    else:
        print('plz use the correct mod')
    return shuffled_tensor, random_index_mapping

# choose 5th and 10th dense features to check importance
for idx in [5, 10]:
    shuffled_tensor, random_index_mapping = shuffle(dfe_data, mod='dense')
    # is_shuffle = (random_index_mapping[idx] == idx)
    new_dfe_data = sess.run(tf.concat([dfe_data[:, :idx], shuffled_tensor[:, idx:idx+1], dfe_data[:, idx+1:]], 1))
    # test again
    dfe_test = new_dfe_data[0:1, :]
    feed_dict_test[dfe] = dfe_test
    l = sess.run([loss], feed_dict=feed_dict_test)
    print('after shuffle %d: '%idx, l)

# choose 2nd sparse feature to check its importance
sidx = 2
shuffled_tensor, random_index_mapping = shuffle(globals()['fe%s_data'%str(sidx)], mod='sparse')
tmp = sess.run(shuffled_tensor)
feed_dict_test[globals()['fe%s'%str(sidx)]] = tmp[0:1, :]
l = sess.run([loss], feed_dict=feed_dict_test)
print('after shuffle %d: '%sidx, l)


'''7. 特征重要性分析, 方法二: 泰勒展开方式, shuffle方法的一阶近似'''
def feature_grad(emb, grad):
    from tensorflow.python.training import moving_averages
    n, d = 32, len(emb[0])
    m,v = tf.nn.moments(tf.cast(emb, tf.float32), axes=0, keep_dims=True)
    print(sess.run([tf.shape(m-embs)]))
    score = tf.reduce_mean(tf.abs((m-emb) * grad), axis=0) #[n, d]
    score= tf.reshape(score, [n, d])
    score = tf.reduce_mean(score, axis=1) #[n]
    moving_score = tf.get_local_variable("score", initializer=tf.zeros_initializer, shape=[n], dtype=tf.float32)
    sess.run(moving_score.initializer)
    assign = moving_averages.assign_moving_average(moving_score, score, 0.9, zero_debias=False)
    tf.add_to_collection("EXTRA_OPS", assign)
    # sess.run(tf.initialize_all_variables())
    return assign, moving_score, m

sidx = 2
embs = globals()['fe%s_data'%sidx][1:33, :].astype(np.float32)
grads = tf.gradients(loss, globals()['fe%s'%sidx])
assign, moving_score, m = feature_grad(embs, grads)
_, moving_score, grad, m = sess.run([assign, moving_score, grads, m], feed_dict=feed_dict)
# print(sess.run(score))
print(moving_score)


