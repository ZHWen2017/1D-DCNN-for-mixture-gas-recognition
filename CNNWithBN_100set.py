import tensorflow as tf
import pickle
import numpy as np
import matplotlib.pyplot as plt
import  os
from sklearn.metrics import roc_auc_score, accuracy_score

Batch_size = 64
Max_steps = 5000
Learning_rate = 1e-4
Num_class = 4
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
DECAY_STEPS = 10
REGULARAZTION_RATE = 0.0001
MOVING_AVERAGE_DECAY = 0.99
keep_prob = 0.9
length = 100
##The data preprocessing

train_path = "/home/wzh/PycharmProjects/data/multilabel/multilabel_Xavier/Cross_valid/BNModel/train_set.txt"
test_path = "/home/wzh/PycharmProjects/data/multilabel/multilabel_Xavier/Cross_valid/BNModel/test_set.txt"
f1 = open(train_path, "rb")
gas_train = pickle.load(f1)
gas_train['label'] = np.array(gas_train['label'])
gas_train['gas'] = np.array(gas_train['gas'])
f1.close()

f2 = open(test_path, "rb")
gas_test = pickle.load(f2)
gas_test['label'] = np.array(gas_test['label'])
gas_test['gas'] = np.array(gas_test['gas'])
f2.close()
# print(len(gas_train['label']))
gas_test['label'] = np.array(gas_test['label'])


print(gas_train['gas'].shape)
print(gas_train['label'].shape)
  ##The normalization of training dataset
for i in range(len(gas_train['gas'])):
    gas_temp = gas_train['gas'][i]
    mean_value = np.mean(gas_temp, axis=0)
    std_value = np.std(gas_temp, axis=0)
    # print(mean_value.shape, std_value.shape)
    for j in range(len(mean_value)):
        gas_temp[:, j] = (gas_temp[:, j] - mean_value[j]) / std_value[j]
# print(gas_train['gas'][3].shape)

   ###The normalization of test dataset
for i in range(len(gas_test['gas'])):
    gas_temp1 = gas_test['gas'][i]
    gas_mean = np.mean(gas_temp1, axis=0)
    gas_std = np.std(gas_temp1, axis=0)
    for j in range(len(gas_mean)):
        gas_temp1[:, j] = (gas_temp1[:, j] - gas_mean[j]) / gas_std[j]

# print(gas_test['gas'][3])

gas_train['gas'] = np.reshape(gas_train['gas'], gas_train['gas'].shape+(1,))
gas_test['gas'] = np.reshape(gas_test['gas'], gas_test['gas'].shape+(1,))
print(gas_train['gas'].shape)


def print_activation(t):
    print(t.op.name, ' ', t.get_shape().as_list()) #显示名称和tensor的尺寸

def gen_batch(X, Y, batch_size):
    x_batch = []
    y_batch = []
    for index in np.random.choice(len(X), size=batch_size):
        x_batch.append(X[index])
        y_batch.append(Y[index])
    x_batch = np.array(x_batch)
    y_batch = np.array(y_batch)
    return x_batch, y_batch

def batch_epoch(X, Y, batch_size):
    for index in range(len(X)-batch_size):
        start_i = index
        end_i = index + batch_size
        yield X[start_i:end_i], Y[start_i:end_i]

def get_weight(shape, lamba):
    var = tf.Variable(tf.truncated_normal(shape, stddev=1.0), dtype=tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lamba)(var))
    return var

def metric(predictions, targets):
    auc = []
    for i in range(predictions.shape[1]):
        temp_auc = roc_auc_score(targets[:, i], predictions[:, i])
        auc.append(temp_auc)
    auc_mean = np.mean(auc)
    return auc, auc_mean

def metric_acc(targets, predictions):
    acc = []
    for i in range(predictions.shape[1]):
        temp_acc = accuracy_score(targets[:, i], predictions[:, i])
        acc.append(temp_acc)
    acc_mean = np.mean(acc)
    return acc, acc_mean

def xavier_init(shape, fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform(shape=shape, minval=low, maxval=high, dtype=tf.float32)

def classification(a, b):
    score_dict = {'n_class': [], 'true_index_1': [], 'true_index_0': [], 'pre_index_1': [], 'pre_index_0': []}
    for i in range(len(a)):
        score_dict['n_class'].append(int(np.sum(a[i])))
        tmp0 = []
        tmp1 = []
        for j in range(len(a[i])):
            if a[i, j] == 1:
                tmp1.append(j)

            else:
                tmp0.append(j)
        score_dict['true_index_1'].append(tmp1)
        score_dict['true_index_0'].append(tmp0)

    c = -np.sort(-b, axis=1)

    i = 0
    for ci in score_dict['n_class']:
        # print(ci)
        chose_index_1 = c[i, :(ci)]
        chose_index_0 = c[i, (ci):]
        temp2 = []
        temp3 = []
        for j1 in range(len(chose_index_1)):
            for k1 in range(len(b[i])):
                if b[i, k1] == chose_index_1[j1]:
                    temp2.append(k1)

        for j0 in range(len(chose_index_0)):
            for k0 in range(len(b[i])):
                if b[i, k0] == chose_index_0[j0]:
                    temp3.append(k0)
        i += 1
        score_dict['pre_index_1'].append(temp2)
        score_dict['pre_index_0'].append(temp3)
    n_count = 0
    for i in range(len(score_dict['n_class'])):
        if score_dict['true_index_1'][i] == score_dict['pre_index_1'][i]:
            n_count += 1
    return n_count

def batch_normal(xs, n_out, ph_train):
    with tf.variable_scope('bn'):
        batch_mean, batch_var = tf.nn.moments(xs, axes=[0])
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]))
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]))
        epsilon = 1e-3

        ema = tf.train.ExponentialMovingAverage(decay=0.5)
        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = tf.cond(ph_train, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        xs_norm = tf.nn.batch_normalization(xs, mean, var, beta, gamma, epsilon)
    return xs_norm

g1 = tf.Graph()
with g1.as_default():
    ##the feed forward net
    X_holder0 = tf.placeholder(tf.float32, [None, 1, length, 1])
    X_holder1 = tf.placeholder(tf.float32, [None, 1, length, 1])
    X_holder2 = tf.placeholder(tf.float32, [None, 1, length, 1])
    X_holder3 = tf.placeholder(tf.float32, [None, 1, length, 1])
    X_holder4 = tf.placeholder(tf.float32, [None, 1, length, 1])
    X_holder5 = tf.placeholder(tf.float32, [None, 1, length, 1])
    X_holder6 = tf.placeholder(tf.float32, [None, 1, length, 1])
    X_holder7 = tf.placeholder(tf.float32, [None, 1, length, 1])
    X_holder8 = tf.placeholder(tf.float32, [None, 1, length, 1])
    X_holder9 = tf.placeholder(tf.float32, [None, 1, length, 1])
    X_holder10 = tf.placeholder(tf.float32, [None, 1, length, 1])
    X_holder11 = tf.placeholder(tf.float32, [None, 1, length, 1])
    X_holder12 = tf.placeholder(tf.float32, [None, 1, length, 1])
    X_holder13 = tf.placeholder(tf.float32, [None, 1, length, 1])
    X_holder14 = tf.placeholder(tf.float32, [None, 1, length, 1])
    X_holder15 = tf.placeholder(tf.float32, [None, 1, length, 1])
    y_holder = tf.placeholder(tf.float32, [None, 3])
    phase_train = tf.placeholder(tf.bool, name='phase_train')



    def Conv(gas):
        with tf.variable_scope('conv1') as scope:
            weight1 = tf.Variable(xavier_init([1, 16, 1, 8], 100, 100), trainable=True)
            # weight1 = tf.get_variable(name='weight1', shape=[1, 4, 1, 4], initializer=tf.contrib.layers.xavier_initializer())
            bias1 = tf.Variable(tf.constant(0.0, shape=[8]))
            # bias1 = tf.get_variable(name='bias1', shape=[4], initializer=tf.zeros_initializer())
            # bias1 = avg_class.average(bias1)
            conv1 = tf.nn.conv2d(gas, weight1, [1, 1, 1, 1], padding='VALID') + bias1
            # if phase_train == True:
            #     conv1 = tf.nn.dropout(conv1, 0.25)
            conv1_bn = batch_normal(conv1, 8, phase_train)
            conv1_out = tf.nn.relu(conv1_bn)

            print_activation(conv1_out)

            # axis = list(range(len(conv1.get_shape()) - 1))
            # mean1, vars1 = tf.nn.moments(conv1, axis)
            # nor1 = tf.nn.batch_normalization(conv1, mean1, vars1)
            # pool1 = tf.nn.avg_pool(conv1, [1, 1, 2, 1], [1, 1, 2, 1], 'VALID')
        return conv1_out, weight1, bias1

    c0, w0, b0= Conv(X_holder0)
    c1, w1, b1= Conv(X_holder1)
    c2, w2, b2= Conv(X_holder2)
    c3, w3, b3= Conv(X_holder3)
    c4, w4, b4= Conv(X_holder4)
    c5, w5, b5= Conv(X_holder5)
    c6, w6, b6= Conv(X_holder6)
    c7, w7, b7= Conv(X_holder7)
    c8, w8, b8= Conv(X_holder8)
    c9, w9, b9 = Conv(X_holder9)
    c10, w10, b10 = Conv(X_holder10)
    c11, w11, b11 = Conv(X_holder11)
    c12, w12, b12 = Conv(X_holder12)
    c13, w13, b13 = Conv(X_holder13)
    c14, w14, b14 = Conv(X_holder14)
    c15, w15, b15 = Conv(X_holder15)
    Conv1 = tf.concat([c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15], axis=3)
    print_activation(Conv1)

    with tf.name_scope('Conv2') as scope:
        # weight2 = tf.Variable(tf.random_uniform(shape=[1, 3, 256, 256], minval=-1, maxval=1), trainable=True)
        # weight2 = tf.Variable(xavier_init([1, 3, 64, 64], 50, 50), trainable=True)
        # weight2 = avg_class.average(weight2)
        # weight2 = get_weight([1, 3, 256, 512], REGULARAZTION_RATE)
        # weight2 = tf.get_variable("weight2", shape=[1, 3, 256, 128],
        #                           initializer=tf.contrib.layers.xavier_initializer(seed=1))
        weight2 = tf.get_variable("weight2", shape=[1, 3, 128, 64],
                                  initializer=tf.glorot_uniform_initializer(seed=1))
        # bias2 = tf.Variable(tf.constant(0.0, shape=[64]))
        # bias2 = avg_class.average(bias2)
        bias2 = tf.get_variable("bias2", shape=[64], initializer=tf.zeros_initializer())
        Conv2 = tf.nn.conv2d(Conv1, weight2, [1, 1, 1, 1], padding='VALID') + bias2
        # if phase_train == True:
        #     Conv2 = tf.nn.dropout(Conv2, 0.25)
        Conv2_bn = batch_normal(Conv2, 64, phase_train)
        Conv2_out = tf.nn.relu(Conv2_bn)
        print_activation(Conv2_out)
        # pool2 = tf.nn.avg_pool(conv2, [1, 1, 2, 1], [1, 1, 2, 1], 'VALID')


    with tf.name_scope('Conv3') as scope:
        # weight3 = get_weight([1, 3, 512, 512], REGULARAZTION_RATE)
        # weight3 = tf.Variable(tf.random_uniform(shape=[1, 3, 256, 256], minval=-1, maxval=1),trainable=True)
        # weight3 = tf.Variable(xavier_init([1, 3, 64, 128], 50, 50), trainable=True)
        # weight3 = tf.get_variable("weight3", shape=[1, 3, 128, 128],
        #                           initializer=tf.contrib.layers.xavier_initializer(seed=1))
        weight3 = tf.get_variable("weight3", shape=[1, 3, 64, 64],
                                  initializer=tf.glorot_uniform_initializer(seed=1))
        bias3 = tf.Variable(tf.constant(0.0, shape=[64]))
        # weight3 = avg_class.average(weight3)
        # bias3 = avg_class.average(bias3)
        Conv3 = tf.nn.conv2d(Conv2_out, weight3, [1, 1, 1, 1], padding='VALID') + bias3
        if phase_train == True:
            Conv3 = tf.nn.dropout(Conv3, 0.25)
        Conv3_bn = batch_normal(Conv3, 64, phase_train)
        Conv3_out = tf.nn.relu(Conv3_bn)
        print_activation(Conv3_out)

        # pool = tf.nn.dropout(pool, 0.9)

    #
    with tf.name_scope('Conv4') as scope:
        # weight3 = get_weight([1, 3, 512, 512], REGULARAZTION_RATE)
        # weight_ = tf.Variable(tf.random_uniform(shape=[1, 2, 256, 512], minval=-1, maxval=1, dtype=tf.float32),trainable=True)
        # weight_1 = tf.Variable(xavier_init([1, 3, 128, 256], 100, 50), trainable=True)
        # weight_1 = tf.get_variable("weight_1", shape=[1, 3, 128, 256],
        #                           initializer=tf.contrib.layers.xavier_initializer(seed=0))
        weight_1 = tf.get_variable("weight_1", shape=[1, 3, 64, 128],
                                   initializer=tf.glorot_uniform_initializer(seed=1))
        bias_1 = tf.Variable(tf.constant(0.0, shape=[128]))
        # weight3 = avg_class.average(weight3)
        # bias3 = avg_class.average(bias3)
        conv_ = tf.nn.conv2d(Conv3_out, weight_1, [1, 1, 1, 1], padding='VALID') + bias_1
        # if phase_train == True:
        #     conv_ = tf.nn.dropout(conv_, 0.25)
        conv_bn = batch_normal(conv_, 128, phase_train)
        conv_out = tf.nn.relu(conv_bn)
        # print_activation(conv_)
        pool_ = tf.nn.avg_pool(conv_out,[1, 1, 4, 1], [1, 1, 3, 1], 'VALID')
        print_activation(pool_)

    # #     print_activation(conv_)



    # flatten = tf.concat([conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7], axis=1)
    # flatten = tf.cast(flatten, tf.float32)
    # dim = flatten.shape[1].value
    flatten = tf.reshape(pool_, [-1, 26*128])
    dim = flatten.shape[1].value


    with tf.name_scope('fc1') as scope:

        # weight4 = tf.Variable(tf.random_uniform([dim, 1024], -1, 1),trainable=True)
        # weight4 = tf.Variable(xavier_init([dim, 256], 50, 50), trainable=True)
        # weight4 = get_weight([dim, 1024], REGULARAZTION_RATE)
        # weight4 = tf.get_variable("weight4", shape=[dim, 256],
        #                           initializer=tf.contrib.layers.xavier_initializer(seed=1))
        weight4 = tf.get_variable("weight4", shape=[dim, 64],
                                  initializer=tf.glorot_uniform_initializer())
        bias4 = tf.Variable(tf.constant(0.001, shape=[64]))

        fc1 = (tf.matmul(flatten, weight4) + bias4)
        # if phase_train == True:
        #     fc1 = tf.nn.dropout(fc1, 0.25)
        fc1_bn = batch_normal(fc1, 64, phase_train)
        fc1_out = tf.nn.relu(fc1_bn)
        # print_activation(fc1)


    with tf.name_scope('fc2') as scope:
        # weight5 = tf.Variable(tf.random_uniform([1024, 512], -1, 1))
        # weight5 = tf.Variable(xavier_init([256, 128], 50, 100))
        # weight5 = tf.get_variable("weight5", shape=[256, 128],
        #                           initializer=tf.contrib.layers.xavier_initializer(seed=1))
        weight5 = tf.get_variable("weight5", shape=[64, 32],
                                  initializer=tf.glorot_uniform_initializer())
        bias5 = tf.Variable(tf.constant(0.0, shape=[32]), trainable=True)
        fc2 = (tf.matmul(fc1_out, weight5) + bias5)
        if phase_train == True:
            fc2 = tf.nn.dropout(fc2, 0.25)
        fc2_bn = batch_normal(fc2, 32, phase_train)
        # print_activation(fc2)
        fc2_out = tf.nn.relu(fc2_bn)

    with tf.name_scope('output') as scope:
        # weight_ = tf.Variable(tf.random_uniform([512, 3], -1, 1),trainable=True)
        # weight_ = tf.Variable(xavier_init([128, 3], 50, 100), trainable=True)
        # weight_ = tf.get_variable("weight_", shape=[128, 3],
        #                           initializer=tf.contrib.layers.xavier_initializer(seed=1))
        weight_ = tf.get_variable("weight_", shape=[32, 3],
                                  initializer=tf.glorot_uniform_initializer())
        # weight5 = get_weight([1024, 5], REGULARAZTION_RATE)

        bias_ = tf.Variable(tf.constant(0.001, shape=[3]))
        logits = (tf.matmul(fc2_out, weight_) + bias_)
        if phase_train == True:
            logits = tf.nn.dropout(logits, 0.5)
        logits = batch_normal(logits, 3, phase_train)
        # print(logits)
        output = tf.nn.sigmoid(logits)
        predictions = tf.round(output)
    # regularation =  tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)(weight3)
    cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y_holder), 1))
    # cross_entropy = cross_entropy +  regularation
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=LEARNING_RATE_BASE, global_step=global_step,
                                               decay_steps=DECAY_STEPS, decay_rate=LEARNING_RATE_DECAY, staircase=True)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)



    saver = tf.train.Saver()
ckpt_dir = "./ckpt_dir"
if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)

test_gas_batch, test_label_batch = gas_test['gas'], gas_test['label']
flag_test = 0
feed_dict1 = {X_holder0: test_gas_batch[:, 0:1, :, :], X_holder1: test_gas_batch[:, 1:2, :, :],
                     X_holder2: test_gas_batch[:, 2:3, :, :], X_holder3: test_gas_batch[:, 3:4, :, :],
                     X_holder4: test_gas_batch[:, 4:5, :, :], X_holder5: test_gas_batch[:, 5:6, :, :],
                     X_holder6: test_gas_batch[:, 6:7, :, :], X_holder7: test_gas_batch[:, 7:8, :, :],
                      X_holder8: test_gas_batch[:, 8:9, :, :], X_holder9: test_gas_batch[:, 9:10, :, :],
                      X_holder10: test_gas_batch[:, 10:11, :, :], X_holder11: test_gas_batch[:, 11:12, :, :],
                      X_holder12: test_gas_batch[:, 12:13, :, :], X_holder13: test_gas_batch[:, 13:14, :, :],
                      X_holder14: test_gas_batch[:, 14:15, :, :], X_holder15: test_gas_batch[:, 15:16, :, :],
                      y_holder: test_label_batch, phase_train: False}

# print(feed_dict1[X_holder1],feed_dict1[X_holder12])
with tf.Session(graph=g1) as sess:
    losses = []
    train_aucs = []
    test_losses = []
    test_aucs = []
    test_acc_per = []
    valid_accs = []
    train_accs = []
    acc_best = 0
    epochElec = 0
    sess.run(tf.global_variables_initializer())
    for epoch in range(Max_steps):

        gas_batch, label_batch = gen_batch(gas_train['gas'], gas_train['label'], Batch_size)

        # print(gas_batch.shape)
        feed_dict = {X_holder0:gas_batch[:, 0:1, :, :], X_holder1:gas_batch[:, 1:2, :, :], X_holder2:gas_batch[:, 2:3, :, :],
                     X_holder3: gas_batch[:, 3:4, :, :], X_holder4:gas_batch[:, 4:5, :, :],X_holder5:gas_batch[:, 5:6, :, :],
                     X_holder6:gas_batch[:, 6:7, :, :], X_holder7:gas_batch[:, 7:8, :, :],
                     X_holder8: gas_batch[:, 8:9, :, :], X_holder9: gas_batch[:, 9:10, :, :],
                     X_holder10: gas_batch[:, 10:11, :, :], X_holder11: gas_batch[:, 11:12, :, :], X_holder12: gas_batch[:, 12:13, :, :],
                     X_holder13: gas_batch[:, 13:14, :, :], X_holder14: gas_batch[:, 14:15, :, :], X_holder15: gas_batch[:, 15:16, :, :],
                     y_holder:label_batch, phase_train: True}
        # print(sess.run(conv0, feed_dict=feed_dict).shape)
        sess.run(train_step, feed_dict=feed_dict)

        y_train, loss= sess.run([predictions, cross_entropy], feed_dict=feed_dict)
        y_test, loss_test = sess.run([predictions, cross_entropy], feed_dict=feed_dict1)
        # print(y_train)
        test_count = 0
        train_count = 0
        for i in range(len(y_test)):
            if (y_test[i] == test_label_batch[i]).all():
                test_count += 1
        for j in range(len(y_train)):
            if (y_train[j] == label_batch[j]).all():
                train_count+=1
        accuracy_train = train_count / len(y_train)
        accuracy_test = test_count / len(y_test)
        # test_auc, test_auc_mean = metric(test_out, test_label_batch)
        acc_test, test_acc_mean = metric_acc(y_test, test_label_batch)

        form0 = [epoch, loss, loss_test, test_count, accuracy_train, accuracy_test]
        losses.append(loss)
        # test_aucs.append(test_auc_mean)
        test_acc_per.append(acc_test)
        valid_accs.append(accuracy_test)
        # train_aucs.append(AUC_train)
        test_losses.append(loss_test)
        train_accs.append(accuracy_train)
        if epoch % 5 == 0:
            print('Epoch #{}, Training loss = {:.4f}, test loss={:.4f}, '
                  'correct count = {:.1f}, train accuracy = {:.5f}, vaild accuracy = {:.5f}'.format(*form0))
            print('acc_per: ', acc_test)
        # print(train_acc)
        # print('The training set sigmoid output: \n', y_train)
        if accuracy_test >= acc_best:
            acc_best = accuracy_test
            saver.save(sess, ckpt_dir+'/BNmodel.ckpt')
            epochElec = epoch
            # print('model updated')


    # print('Epoch #{}, Training loss = {:.4f}, test loss={:.4f}, test_mean_auc = {:.5f}, '
    #       'correct count = {:.1f}, vaild accuracy = {:.5f}'.format(*form0))
    # saver.save(sess, ckpt_dir + '/model.ckpt')
test_acc_per = np.array(test_acc_per)
print('\n the best test accuracy(auc): %.5f, best loss: %.4f, epoch: %d' %(acc_best, test_losses[int(epochElec)], epochElec))

Loss = {}; accuracy = {}
Loss['train_loss'] = losses; Loss['test_loss'] = test_losses
accuracy['train_accuary'] = train_accs; accuracy['test_accuracy'] = valid_accs
accuracy['best acc'] = acc_best

epoch_seq = np.arange(1, Max_steps+1,10)

fig1 = plt.figure(figsize=(6,6))
fig2 = plt.figure(figsize=(6,6))
ax1 = fig1.add_subplot(1, 1, 1)
ax2 = fig2.add_subplot(1, 1, 1)

ax1.plot(epoch_seq, losses[::10], 'k-', label = 'Training Set')
ax1.plot(epoch_seq, test_losses[::10], 'r-', label = 'Test Set')
ax1.set_title('Loss')
ax1.set_xlabel('Iterations')
ax1.set_ylabel('loss')
ax1.legend(loc = 'best')


# ax2.plot(epoch_seq, test_acc_per[:, 0], 'k--', label = 'Ethylene accuracy')
# ax2.plot(epoch_seq, test_acc_per[:, 1], 'b--', label = 'CO accuracy')
# ax2.plot(epoch_seq, test_acc_per[:, 2], 'g--', label = 'CO accuracy')
ax2.plot(epoch_seq, valid_accs[::10], 'r-', label = 'Valid accuracy')
# ax2.plot(epoch_seq, train_accs[::5], 'g-', label = 'Train accuracy')
##set ticks
my_y_ticks = np.arange(0, 1.1, 0.1)
# my_y_ticks = np.arange(0, Max_steps, int(Max_steps/10000))
ax2.set_yticks(my_y_ticks)
# ax2.set_yticks(my_y_ticks)
x_label_name = 'Iterations (best acc %.5f)' % acc_best
ax2.set_title('Accuracy')
ax2.set_xlabel(x_label_name)
ax2.set_ylabel('accuracy')
ax2.legend(loc = 'best')


# name='/home/wzh/PycharmProjects/data/multilabel/multilabel_Xavier/Cross_valid/BNModel/result/xavier_80.eps'
# plt.savefig(name, format='eps')
plt.show()

loss_dir = '/home/wzh/PycharmProjects/data/multilabel/multilabel_Xavier/Cross_valid/BNModel/result/100_loss.txt'
acc_dir = '/home/wzh/PycharmProjects/data/multilabel/multilabel_Xavier/Cross_valid/BNModel/result/100_accuracy.txt'

f3 = open(loss_dir, 'wb')
pickle.dump(Loss, f3)
f3.close()

f4 = open(acc_dir, 'wb')
pickle.dump(accuracy, f4)
f4.close()