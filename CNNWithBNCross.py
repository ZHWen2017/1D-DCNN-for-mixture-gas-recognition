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
DECAY_STEPS = 20
REGULARAZTION_RATE = 0.0001
MOVING_AVERAGE_DECAY = 0.99
keep_prob = 0.9
length = 40

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
        epsilon = 1e-4

        ema = tf.train.ExponentialMovingAverage(decay=0.7)
        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = tf.cond(ph_train, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        xs_norm = tf.nn.batch_normalization(xs, mean, var, beta, gamma, epsilon)
    return xs_norm
def CNNProcess(hold_i, gas_train, gas_test, Max_steps):
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
                weight1 = tf.Variable(xavier_init([1, 4, 1, 8], 100, 100), trainable=True)
                # weight1 = tf.get_variable(name='weight1', shape=[1, 4, 1, 4], initializer=tf.contrib.layers.xavier_initializer())
                bias1 = tf.Variable(tf.constant(0.0, shape=[8]))
                # bias1 = tf.get_variable(name='bias1', shape=[4], initializer=tf.zeros_initializer())
                # bias1 = avg_class.average(bias1)
                conv1 = tf.nn.conv2d(gas, weight1, [1, 1, 1, 1], padding='VALID') + bias1
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
                                       initializer=tf.glorot_uniform_initializer(seed=0))
            bias_1 = tf.Variable(tf.constant(0.0, shape=[128]))
            # weight3 = avg_class.average(weight3)
            # bias3 = avg_class.average(bias3)
            conv_ = tf.nn.conv2d(Conv3_out, weight_1, [1, 1, 1, 1], padding='VALID') + bias_1
            conv_bn = batch_normal(conv_, 128, phase_train)
            conv_out = tf.nn.relu(conv_bn)
            # print_activation(conv_)
            pool_ = tf.nn.avg_pool(conv_out,[1, 1, 4, 1], [1, 1, 3, 1], 'VALID')
            print_activation(pool_)

        # #     print_activation(conv_)



        # flatten = tf.concat([conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7], axis=1)
        # flatten = tf.cast(flatten, tf.float32)
        # dim = flatten.shape[1].value
        flatten = tf.reshape(pool_, [-1, 10*128])
        dim = flatten.shape[1].value


        with tf.name_scope('fc1') as scope:

            # weight4 = tf.Variable(tf.random_uniform([dim, 1024], -1, 1),trainable=True)
            # weight4 = tf.Variable(xavier_init([dim, 256], 50, 50), trainable=True)
            # weight4 = get_weight([dim, 1024], REGULARAZTION_RATE)
            # weight4 = tf.get_variable("weight4", shape=[dim, 256],
            #                           initializer=tf.contrib.layers.xavier_initializer(seed=1))
            weight4 = tf.get_variable("weight4", shape=[dim, 256],
                                      initializer=tf.glorot_uniform_initializer())
            bias4 = tf.Variable(tf.constant(0.001, shape=[256]))
            fc1 = (tf.matmul(flatten, weight4) + bias4)
            if phase_train == True:
                fc1 = tf.nn.dropout(fc1, 0.25)
            fc1_bn = batch_normal(fc1, 256, phase_train)
            fc1_out = tf.nn.relu(fc1_bn)
            # print_activation(fc1)


        with tf.name_scope('fc2') as scope:
            # weight5 = tf.Variable(tf.random_uniform([1024, 512], -1, 1))
            # weight5 = tf.Variable(xavier_init([256, 128], 50, 100))
            # weight5 = tf.get_variable("weight5", shape=[256, 128],
            #                           initializer=tf.contrib.layers.xavier_initializer(seed=1))
            weight5 = tf.get_variable("weight5", shape=[256, 128],
                                      initializer=tf.glorot_uniform_initializer())
            bias5 = tf.Variable(tf.constant(0.0, shape=[128]), trainable=True)
            fc2 = (tf.matmul(fc1_out, weight5) + bias5)
            if phase_train == True:
                fc2 = tf.nn.dropout(fc2, 0.5)
            fc2_bn = batch_normal(fc2, 128, phase_train)
            # print_activation(fc2)
            fc2_out = tf.nn.relu(fc2_bn)

        with tf.name_scope('output') as scope:
            # weight_ = tf.Variable(tf.random_uniform([512, 3], -1, 1),trainable=True)
            # weight_ = tf.Variable(xavier_init([128, 3], 50, 100), trainable=True)
            # weight_ = tf.get_variable("weight_", shape=[128, 3],
            #                           initializer=tf.contrib.layers.xavier_initializer(seed=1))
            weight_ = tf.get_variable("weight_", shape=[128, 3],
                                      initializer=tf.glorot_uniform_initializer())
            # weight5 = get_weight([1024, 5], REGULARAZTION_RATE)
            bias_ = tf.Variable(tf.constant(0.001, shape=[3]))
            logits = (tf.matmul(fc2_out, weight_) + bias_)

            # print(logits)
            output = tf.nn.sigmoid(logits)
            # predictions = tf.round(output)

        # regularation =  tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)(weight3)
        cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y_holder), 1))
        # cross_entropy = cross_entropy +  regularation
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate=LEARNING_RATE_BASE, global_step=global_step,
                                                   decay_steps=DECAY_STEPS, decay_rate=LEARNING_RATE_DECAY, staircase=True)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)

        # accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, 1),tf.argmax(y_holder, 1)), tf.float32))

        # #metric
        # output_rack = tf.unstack(output, axis=1)
        # label_rack = tf.unstack(y_holder, axis=1)
        # zip0= tf.transpose(tf.concat([output_rack, label_rack], axis=0))
        # auc = []
        # auc_op = []
        # i = 0
        # for output, label in zip0:
        #     value, op = tf.contrib.metrics.streaming_auc(output, label, name='PR')
        #     auc_op.append(op)
        #     auc.append(value)
        #     i+=1
        # mean_auc = tf.reduce_mean(auc)

        saver = tf.train.Saver()
    ckpt_dir = "./ckpt_dir1"
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

            train_out, loss= sess.run([output, cross_entropy], feed_dict=feed_dict)
            test_out, loss_test = sess.run([output, cross_entropy], feed_dict=feed_dict1)
            # y_test = np.array(test_out, dtype=np.int32)
            y_test = np.round(test_out)
            # count_train = classification(y_train, label_batch)
            # count_test = classification(y_test, test_label_batch)

            # train_acc = count_train / len(label_batch)
            # test_acc = count_test / len(test_label_batch)
            # train_auc, _, _ = metric(train_out, label_batch)
            test_count = 0
            for i in range(len(y_test)):
                if (y_test[i] == test_label_batch[i]).all():
                    test_count += 1
            accuracy_test = test_count / len(y_test)
            # test_auc, test_auc_mean = metric(test_out, test_label_batch)
            acc_test, test_acc_mean = metric_acc(y_test, test_label_batch)

            form0 = [epoch+20, loss, loss_test, test_count, accuracy_test]

            if epoch % 20 == 0:
                # print('Epoch #{}, Training loss = {:.4f}, test loss={:.4f}, test_mean_auc = {:.5f}, '
                #       'correct count = {:.1f}, vaild accuracy = {:.5f}'.format(*form0))
                # print('acc_per: ', acc_test)
                # print(train_acc)
                # print('The training set sigmoid output: \n', y_train)

                if accuracy_test >= acc_best:
                    acc_best = accuracy_test
                    saver.save(sess, ckpt_dir+'/BNmodel.ckpt')
                    epochElec = epoch
                    print('Hold %d, The model has updated!' % hold_i)

                losses.append(loss)
                # test_aucs.append(test_auc_mean)
                test_acc_per.append(acc_test)
                valid_accs.append(accuracy_test)
                # train_aucs.append(AUC_train)
                test_losses.append(loss_test)
        # print('Epoch #{}, Training loss = {:.4f}, test loss={:.4f}, test_mean_auc = {:.5f}, '
        #       'correct count = {:.1f}, vaild accuracy = {:.5f}'.format(*form0))
        # saver.save(sess, ckpt_dir + '/model.ckpt')
    test_acc_perBest = test_acc_per[int(epochElec/20)]
    print('\n the best test accuracy(auc): %.5f, best loss: %.4f' %(acc_best, test_losses[int(epochElec/20)]))
    print('test acc per:', test_acc_perBest)
    return acc_best, test_acc_perBest

def gasLoad():
    gas = {}
    f1 = open("../../../gas.txt", "rb")
    gas1 = pickle.load(f1)
    gas['gas'] = gas1['sensor_data']
    gas['label'] = gas1['labels']
    # gas_train['label'] = np.array(gas_train['label'])
    f1.close()

    print(gas['gas'].shape)
    print(gas['label'].shape)
    ##The normalization of training dataset
    for i in range(len(gas['gas'])):
        gas_temp = gas['gas'][i]
        mean_value = np.mean(gas_temp, axis=0)
        std_value = np.std(gas_temp, axis=0)
        # print(mean_value.shape, std_value.shape)
        for j in range(len(mean_value)):
            gas_temp[:, j] = (gas_temp[:, j] - mean_value[j]) / std_value[j]
    # print(gas_train['gas'][3].shape)


    gas['gas'] = np.reshape(gas['gas'], gas['gas'].shape + (1,))

    ##split gas data
    gas_spilt = {}
    gas_spilt['gas'] = np.split(gas['gas'][:630], 10, axis=0)
    gas_spilt['labels'] = np.split(gas['label'][:630], 10, axis=0)
    print(len(gas_spilt['gas'][9]), len(gas_spilt['labels'][9]))
    valid_set = {'gas':[], 'labels':[]}
    train_set = {'gas':[], 'labels':[]}
    for i in range(10):
        valid_set['gas'].append(gas_spilt['gas'][i])
        valid_set['labels'].append(gas_spilt['labels'][i])
        gas_delete = np.delete(gas_spilt['gas'], i, 0)
        labels_delete = np.delete(gas_spilt['labels'], i, 0)
        gas_stack = np.concatenate((gas_delete[0],gas_delete[1],gas_delete[2],gas_delete[3],
                                    gas_delete[4],gas_delete[5], gas_delete[6], gas_delete[7], gas_delete[8]), axis=0)
        labels_stack = np.concatenate((labels_delete[0], labels_delete[1], labels_delete[2],
                                       labels_delete[3], labels_delete[4], labels_delete[5],
                                       labels_delete[6], labels_delete[7], labels_delete[8]), axis=0)
        train_set['gas'].append(gas_stack)
        train_set['labels'].append(labels_stack)
    valid_set['gas'] = np.array(valid_set['gas'])
    valid_set['labels'] = np.array(valid_set['labels'])
    train_set['gas'] = np.array(train_set['gas'])
    train_set['labels'] = np.array(train_set['labels'])
    # ###split gas data
    # gas_spilt = {}
    # gas_spilt['gas'] = np.split(gas['gas'][:630], 5, axis=0)
    # gas_spilt['labels'] = np.split(gas['label'][:630], 5, axis=0)
    # print(len(gas_spilt['gas'][4]), len(gas_spilt['labels'][4]))
    # valid_set = {'gas':[], 'labels':[]}
    # train_set = {'gas':[], 'labels':[]}
    # for i in range(5):
    #     valid_set['gas'].append(gas_spilt['gas'][i])
    #     valid_set['labels'].append(gas_spilt['labels'][i])
    #     gas_delete = np.delete(gas_spilt['gas'], i, 0)
    #     labels_delete = np.delete(gas_spilt['labels'], i, 0)
    #     gas_stack = np.concatenate((gas_delete[0],gas_delete[1],gas_delete[2],gas_delete[3]), axis=0)
    #     labels_stack = np.concatenate((labels_delete[0], labels_delete[1], labels_delete[2],labels_delete[3]), axis=0)
    #     train_set['gas'].append(gas_stack)
    #     train_set['labels'].append(labels_stack)
    # valid_set['gas'] = np.array(valid_set['gas'])
    # valid_set['labels'] = np.array(valid_set['labels'])
    # train_set['gas'] = np.array(train_set['gas'])
    # train_set['labels'] = np.array(train_set['labels'])
    #
    # print(valid_set['gas'].shape, valid_set['labels'].shape, train_set['gas'].shape, train_set['labels'].shape)
    return train_set, valid_set

train_set, valid_set = gasLoad()

acc_valid_bests=[]
acc_pers = []
aucs = []
for hold_i in range(len(train_set['gas'])):
    gas_train={}; gas_valid={}
    gas_train['gas'] = train_set['gas'][hold_i]; gas_train['label'] = train_set['labels'][hold_i]
    gas_valid['gas'] = valid_set['gas'][hold_i]; gas_valid['label'] = valid_set['labels'][hold_i]
    print('\n\n----------------------Hold #{} training and validing-----------------------'.format(hold_i+1))
    acc_valid_best_temp, acc_per  = CNNProcess(hold_i, gas_train, gas_valid, Max_steps)
    acc_valid_bests.append(acc_valid_best_temp)
    acc_pers.append(acc_per)
acc_mean = np.mean(acc_valid_bests)
acc_per_mean = np.mean(acc_pers, axis=0)
print('\nThe mean value of 10 hold cross validation= %.5f' % (acc_mean))
print('The mean value of per acc: ', acc_per_mean)
