import tensorflow as tf
import pickle
import numpy as np
import matplotlib.pyplot as plt
import  os
from sklearn.metrics import roc_auc_score, accuracy_score
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

def CNN_process(hold_i, Max_steps, gas_train, gas_test, hidden_nums):

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

                # print_activation(conv1_out)

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
        # print_activation(Conv1)

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
            # print_activation(Conv2_out)
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
            # print_activation(Conv3_out)

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
            # print_activation(pool_)

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
            weight5 = tf.get_variable("weight5", shape=[64, hidden_nums],
                                      initializer=tf.glorot_uniform_initializer())
            bias5 = tf.Variable(tf.constant(0.0, shape=[hidden_nums]), trainable=True)
            fc2 = (tf.matmul(fc1_out, weight5) + bias5)
            if phase_train == True:
                fc2 = tf.nn.dropout(fc2, 0.25)
            fc2_bn = batch_normal(fc2, hidden_nums, phase_train)
            # print_activation(fc2)
            fc2_out = tf.nn.relu(fc2_bn)

        with tf.name_scope('output') as scope:
            # weight_ = tf.Variable(tf.random_uniform([512, 3], -1, 1),trainable=True)
            # weight_ = tf.Variable(xavier_init([128, 3], 50, 100), trainable=True)
            # weight_ = tf.get_variable("weight_", shape=[128, 3],
            #                           initializer=tf.contrib.layers.xavier_initializer(seed=1))
            weight_ = tf.get_variable("weight_", shape=[hidden_nums, 3],
                                      initializer=tf.glorot_uniform_initializer())
            # weight5 = get_weight([1024, 5], REGULARAZTION_RATE)

            bias_ = tf.Variable(tf.constant(0.001, shape=[3]))
            logits = (tf.matmul(fc2_out, weight_) + bias_)
            # if phase_train == True:
            #     logits = tf.nn.dropout(logits, 0.5)
            # logits = batch_normal(logits, 3, phase_train)
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
    init = tf.global_variables_initializer()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True

    with tf.Session(graph=g1, config=config) as sess:
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
        print("Multi-label process....")
        for epoch in range(Max_steps):
            start_time = time.time()
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
            duration = time.time() - start_time
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
                print("\r{} steps are processing".format(epoch+5), end=" ")
            #     print('Epoch #{}, Training loss = {:.4f}, test loss={:.4f}, '
            #           'correct count = {:.1f}, train accuracy = {:.5f}, vaild accuracy = {:.5f}'.format(*form0))
                # print('acc_per: ', acc_test)
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
            last_batch_time = float(duration)
    print('The last batch costs %fs, the best test accuracy(auc): %.5f, best loss: %.4f, epoch: %d' %(last_batch_time, acc_best, test_losses[int(epochElec)], epochElec))
    return valid_accs, acc_best, test_losses[int(epochElec)], losses, test_losses, train_accs, last_batch_time
    # return acc_best, test_losses[int(epochElec)]