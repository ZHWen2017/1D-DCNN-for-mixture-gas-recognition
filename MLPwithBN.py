import tensorflow as tf
import numpy as np
Batch_size = 64
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
DECAY_STEPS = 10

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



def MLP(hold_i, gas_train, gas_test, Maxsteps, length):

    X_holder = tf.placeholder(tf.float32, [None, length])
    y_holder = tf.placeholder(tf.float32, [None, 3])
    phase_train = tf.placeholder(tf.bool, name='phase_train')


    with tf.name_scope('fc1') as scope:

        weight4 = tf.Variable(tf.random_uniform([length, 64], -1, 1),trainable=True)
        # weight4 = tf.Variable(xavier_init([length, 64], 50, 50), trainable=True)
        # weight4 = get_weight([dim, 1024], REGULARAZTION_RATE)
        # weight4 = tf.get_variable("weight4", shape=[dim, 256],
        #                           initializer=tf.contrib.layers.xavier_initializer(seed=1))
        # weight4 = tf.get_variable("weight4", shape=[length, 64],
        #                           initializer=tf.glorot_uniform_initializer())
        bias4 = tf.Variable(tf.constant(0.001, shape=[64]))

        fc1 = (tf.matmul(X_holder, weight4) + bias4)
        # if phase_train == True:
        #     fc1 = tf.nn.dropout(fc1, 0.25)
        # fc1_bn = batch_normal(fc1, 64, phase_train)
        fc1_out = tf.nn.relu(fc1)
        # print_activation(fc1)


    with tf.name_scope('fc2') as scope:
        weight5 = tf.Variable(tf.random_uniform([64, 32], -1, 1))
        # weight5 = tf.Variable(xavier_init([64, 32], 50, 100))
        # weight5 = tf.get_variable("weight5", shape=[256, 128],
        #                           initializer=tf.contrib.layers.xavier_initializer(seed=1))
        # weight5 = tf.get_variable("weight5", shape=[64, 32],
        #                           initializer=tf.glorot_uniform_initializer())
        bias5 = tf.Variable(tf.constant(0.0, shape=[32]), trainable=True)
        fc2 = (tf.matmul(fc1_out, weight5) + bias5)
        # if phase_train == True:
        #     fc2 = tf.nn.dropout(fc2, 0.25)
        # fc2_bn = batch_normal(fc2, 32, phase_train)
        # print_activation(fc2)
        fc2_out = tf.nn.relu(fc2)

    with tf.name_scope('output') as scope:
        weight_ = tf.Variable(tf.random_uniform([32,3],-1, 1),trainable=True)
        # weight_ = tf.Variable(xavier_init([128, 3], 50, 100), trainable=True)
        # weight_ = tf.get_variable("weight_", shape=[128, 3],
        #                           initializer=tf.contrib.layers.xavier_initializer(seed=1))
        # weight_ = tf.get_variable("weight_", shape=[32, 3],
        #                           initializer=tf.glorot_uniform_initializer())
        # weight_ = tf.Variable(xavier_init([32, 3], 50, 50), trainable=True)
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
    cross_entropy = tf.reduce_mean(
        tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y_holder), 1))
    # cross_entropy = cross_entropy +  regularation
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=LEARNING_RATE_BASE, global_step=global_step,
                                               decay_steps=DECAY_STEPS, decay_rate=LEARNING_RATE_DECAY,
                                               staircase=True)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)

    gas_test_batch, label_test_batch = gas_test['gas'], gas_test['label']
    test_dict = {X_holder:gas_test_batch, y_holder:label_test_batch, phase_train:False}

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_losses = []
        test_losses = []
        train_accs = []
        test_accs = []
        acc_best = 0
        for iter in range(Maxsteps):
            gas_batch, label_batch = gen_batch(gas_train['gas'], gas_train['label'], Batch_size)
            train_dict = {X_holder:gas_batch, y_holder:label_batch, phase_train:True}
            y_train, train_loss, _ = sess.run([predictions, cross_entropy, train_step], feed_dict=train_dict)
            y_test, test_loss = sess.run([predictions, cross_entropy], feed_dict=test_dict)
            test_count = 0
            train_count = 0
            for i in range(len(y_test)):
                if (y_test[i] == label_test_batch[i]).all():
                    test_count += 1
            for j in range(len(y_train)):
                if (y_train[j] == label_batch[j]).all():
                    train_count+=1
            accuracy_train = train_count / len(y_train)
            accuracy_test = test_count / len(y_test)
            train_losses.append(train_loss); test_losses.append(test_loss)
            train_accs.append(accuracy_train); test_accs.append(accuracy_test)
            form0 = [hold_i, iter, train_loss, test_loss, accuracy_train, accuracy_test]
            if iter % 20== 0:
                print('FOLD#{}, Iterations #{}, Training loss = {:.4f}, test loss={:.4f}, '
                      'train accuracy = {:.5f}, valid accuracy = {:.5f}'.format(*form0))
            if accuracy_test >= acc_best:
                acc_best = accuracy_test

        return test_accs, acc_best, y_test
