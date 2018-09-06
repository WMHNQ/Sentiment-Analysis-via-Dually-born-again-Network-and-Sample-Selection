# -*- coding: utf-8 -*-
#损失同时加上自己的损失
#SELECT WHICH MODEL YOU WISH TO RUN:
import numpy as np
from lstm_cnn import LSTM_CNN_born_learnweight   #OPTION 0
from cnn_lstm import CNN_LSTM   #OPTION 1
from cnn import CNN_born_learnweight             #OPTION 2 (Model by: Danny Britz)
from lstm import LSTM           #OPTION 3
import time
MODEL_TO_RUN = 0
gate_position = 0 #0表示真实标签位置，1表示信息蒸馏位置，2表示总的位置。
born = False
born_file = 'result/mean_.npy'
# born_file = 'result/LSTMCNN/train_0.npy'
# born_file = 'result/CNN/train_0.npy'
# born_file = 'result/LIN/train_0.npy'
# born_file = 'result/mean_CNN_LSTMCNN.npy'
# born_file = 'result/mean_LIN_LSTMCNN.npy'
# born_file = 'result/mean_LIN_CNN.npy'
# weightloss = np.load('result/trainAStest_LSTMCNN.npy')
weightloss1 = np.load('result/trainAStest_CNN.npy')
weightloss2 = np.load('result/trainAStest_LIN.npy')
weightloss3 = np.load('result/trainAStest_LSTMCNN.npy')
if MODEL_TO_RUN == 0:
    resultfile = 'LSTMCNN_born_learn_gate'+str(time.strftime('%m-%d-%H-%M-%S',time.localtime()))
elif MODEL_TO_RUN == 2:
    resultfile = 'CNN_born_learn_gate'+str(time.strftime('%m-%d-%H-%M-%S',time.localtime()))

f_epoch = 50
bornNtimes = 5
born_epoch = 10
l2_reg = 0

balance_gate = np.array([0.8])    #大于0
import tensorflow as tf
import numpy as np
import os

import datetime
import data_helpers
from tensorflow.contrib import learn

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("data_file", "./data/mr/mr0", "Data source.")
# tf.flags.DEFINE_string("resule_save", "s", "s->/data/mr/mr0, 1->/data/mr/mr0/subset1, 2->/data/mr/mr0/subset2...")

#word embedding
# tf.app.flags.DEFINE_string('embedding_file_path', 'data/twitter_word_embedding_partial_100.txt', 'embedding file')
# tf.app.flags.DEFINE_integer('word_embedding_dim', 100, 'dimension of word embedding')
# tf.app.flags.DEFINE_string('embedding_file_path', 'data/glove.6B.300d.txt', 'embedding file')
tf.app.flags.DEFINE_string('embedding_file_path', 'data/mr_glove.42B.300d.txt', 'embedding file')
tf.app.flags.DEFINE_integer('word_embedding_dim', 300, 'dimension of word embedding')

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", l2_reg, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", f_epoch, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")

def preprocess():
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    x_train_str, y_train = data_helpers.load_data_and_labels(FLAGS.data_file + '/train.txt')
    x_dev_str, y_dev = data_helpers.load_data_and_labels(FLAGS.data_file+'/dev.txt')
    x_test_str, y_test = data_helpers.load_data_and_labels(FLAGS.data_file + '/test.txt')

    #word embedding ---> x[324,1413,1,41,43,0,0,0]  y[0,1]
    #word_id_mapping,such as  apple--->23 ,w2v  23---->[vector]
    word_id_mapping, w2v = data_helpers.load_w2v(FLAGS.embedding_file_path, FLAGS.word_embedding_dim)
    max_document_length = max([len(x.split(" ")) for x in (x_test_str + x_train_str + x_dev_str)])
    x_train, x_train_len = data_helpers.word2id(
        x_train_str,
        word_id_mapping,
        max_document_length
    )
    x_dev, x_dev_len = data_helpers.word2id(
        x_dev_str,
        word_id_mapping,
        max_document_length
    )
    x_test, x_test_len = data_helpers.word2id(
        x_test_str,
        word_id_mapping,
        max_document_length
    )
    if born:
        y_train = np.load(born_file)

    y_train = (np.load('result/LSTMCNN/train_0.npy')+np.load('result/LIN/train_0.npy')+np.load('result/CNN/train_0.npy'))/3
    print("Vocabulary Size: {:d}".format(len(word_id_mapping)))
    print("Train/Dev/test split: {:d}/{:d}/{:d}".format(len(y_train), len(y_dev), len(y_test)))
    return x_train, y_train, x_dev, y_dev, x_test,y_test, x_test_len, w2v

def train(x_train, y_train, x_dev, y_dev, x_test, y_test, word_embedding):
    # Training
    # ==================================================

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            if MODEL_TO_RUN == 0:
                model = LSTM_CNN_born_learnweight(
                    sequence_length=x_train.shape[1],
                    num_classes=y_train.shape[1],
                    word_embedding = word_embedding,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=FLAGS.num_filters,
                    l2_reg_lambda=FLAGS.l2_reg_lambda)
            elif MODEL_TO_RUN == 1:
                model = CNN_LSTM(
                    sequence_length=x_train.shape[1],
                    num_classes=y_train.shape[1],
                    word_embedding = word_embedding,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=FLAGS.num_filters,
                    l2_reg_lambda=FLAGS.l2_reg_lambda)
            elif MODEL_TO_RUN == 2:
                model = CNN_born_learnweight(
                    sequence_length=x_train.shape[1],
                    num_classes=y_train.shape[1],
                    word_embedding = word_embedding,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=FLAGS.num_filters,
                    l2_reg_lambda=FLAGS.l2_reg_lambda)
            elif MODEL_TO_RUN == 3:
                model = LSTM(
                    sequence_length=x_train.shape[1],
                    num_classes=y_train.shape[1],
                    word_embedding = word_embedding,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=FLAGS.num_filters,
                    l2_reg_lambda=FLAGS.l2_reg_lambda)


            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(model.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", ['LSTM_CNN', 'CNN_LSTN', 'CNN', 'LSTM'][MODEL_TO_RUN],'set'+FLAGS.data_file[-1:]))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", model.loss)
            acc_summary = tf.summary.scalar("accuracy1", model.accuracy1)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Test summaries
            test_summary_op = tf.summary.merge([loss_summary, acc_summary])
            test_summary_dir = os.path.join(out_dir, "summaries", "test")
            test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            # vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch1, y_batch2, y_batchw1,y_batchw2,y_batchw3,balance_gate):
                """
                A single training step
                """
                feed_dict = {
                  model.input_x: x_batch,
                  model.input_ypre: y_batch1,
                  model.input_ytrue: y_batch2,
                  model.input_yw1: y_batchw1,
                  model.input_yw2: y_batchw2,
                  model.input_yw3: y_batchw3,
                  model.balance_gate: balance_gate,
                  model.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy1, accuracy2 = sess.run(
                    [train_op, global_step, train_summary_op, model.loss, model.accuracy1, model.accuracy2],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc1 {:g} acc1 {:g}".format(time_str, step, loss, accuracy1,accuracy2))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch1, y_batch2,y_batchw1,y_batchw2,y_batchw3,balance_gate, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  model.input_x: x_batch,
                  model.input_ypre: y_batch1,
                  model.input_ytrue: y_batch2,
                  model.input_yw1: y_batchw1,
                  model.input_yw2: y_batchw2,
                  model.input_yw3: y_batchw3,
                  model.balance_gate: balance_gate,
                  model.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy1,accuracy2, softmax,softmax2 = sess.run(
                    [global_step, dev_summary_op, model.loss, model.accuracy1,model.accuracy2,model.ensemble_softmax,model.softmax2],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc1 {:g} acc1 {:g}".format(time_str, step, loss, accuracy1, accuracy2))
                accuracy = [accuracy1, accuracy2]
                if writer:
                    writer.add_summary(summaries, step)
                return accuracy,softmax,softmax2

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            train_acc, dev_acc, test_acc, train_all_softmax, train_all_softmax2,test_all_softmax = [], [], [], [], [], []
            train_acc1, dev_acc1, test_acc1, train_all_softmax1, test_all_softmax1 = [], [], [], [], []
            max_dev_acc = 0
            for born_num in range(bornNtimes):
                if born_num == 0:
                    batches = data_helpers.batch_iter(
                        list(zip(x_train, y_train,y_train,weightloss1,weightloss2,weightloss3)), FLAGS.batch_size, FLAGS.num_epochs)
                else:
                    # max_ind1 = dev_acc1.index(max(dev_acc1))
                    # y_train_new = train_all_softmax1[max_ind1]
                    batches = data_helpers.batch_iter(
                        list(zip(x_train, train_all_softmax2[-1],y_train,weightloss1,weightloss2,weightloss3)), FLAGS.batch_size, born_epoch)
                train_acc1, dev_acc1, test_acc1, train_all_softmax1, test_all_softmax1 = [], [], [], [], []
                for batch in batches:
                    x_batch, y_batch1, y_batch2, y_batchw1,y_batchw2,y_batchw3 = zip(*batch)
                    train_step(x_batch, y_batch1, y_batch2, y_batchw1,y_batchw2,y_batchw3,balance_gate)
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % FLAGS.evaluate_every == 0:
                        print('\n--------------------------------------born_num'+str(born_num)+'--------------------------------------')
                        print("Evaluation all train:")
                        train_acc_i,train_softmax_i,train_all_softmax2_i = dev_step(x_train, y_train, y_train,y_train,y_train,y_train,balance_gate, writer=dev_summary_writer)
                        train_acc.append(train_acc_i)
                        train_all_softmax.append(train_softmax_i)
                        train_all_softmax2.append(train_all_softmax2_i)
                        train_acc1.append(train_acc_i)
                        train_all_softmax1.append(train_softmax_i)
                        print("")
                    if current_step % FLAGS.evaluate_every == 0:
                        print("\nEvaluation dev:")
                        dev_acc_i, dev_softmax_i,_ = dev_step(x_dev, y_dev, y_dev,y_dev,y_dev,y_dev,balance_gate, writer=dev_summary_writer)
                        dev_acc.append(dev_acc_i)
                        dev_acc1.append(dev_acc_i)
                        print("")
                    if current_step % FLAGS.evaluate_every == 0:
                        print("\nEvaluation Text:")
                        test_acc_i, test_softmax_i,_ = dev_step(x_test, y_test, y_test,y_test,y_test,y_test,balance_gate, writer=test_summary_writer)
                        test_acc.append(test_acc_i)
                        test_all_softmax.append(test_softmax_i)
                        test_acc1.append(test_acc_i)
                        test_all_softmax1.append(test_softmax_i)
                        print('\n--------------------------------------born_num:'+str(born_num)+'--------------------------------------')
                        print("")
                    if current_step % FLAGS.checkpoint_every == 0:
                        if dev_acc_i[0]>max_dev_acc:
                            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                            print("Saved model checkpoint to {}\n".format(path))
                            print('->>>>>>>>>>>>>>>>>>>>>>>')
                            max_dev_acc = dev_acc_i[0]
    return train_acc, dev_acc, test_acc, train_all_softmax, test_all_softmax

def result_make(train_acc, dev_acc, test_acc,train_all_softmax,test_all_softmax, which_set):
    save_file = ['LSTM_CNN', 'CNN_LSTN', 'CNN', 'LSTM'][MODEL_TO_RUN]
    save_path = 'res/'+resultfile+'/'
    print()
    print('save the acc result to file: '+ save_path+which_set+'.txt')
    print('save the softmax value result to file: '+save_path+'train_' + which_set+'.npy'+' and test_'+ which_set+'.npy')

    try:
        f = open (save_path+which_set+'.txt','w')
    except:
        os.mkdir(save_path)
        f = open(save_path + which_set + '.txt', 'w')
    f.write('f_epoch='+str(f_epoch)+'\n')
    f.write('bornNtimes='+str(bornNtimes)+'\n')
    f.write('born_epoch ='+str(born_epoch)+'\n')
    f.write('l2_reg=:'+str(l2_reg)+'\n')
    f.write('dropout_keep_prob=:'+str(dropout_keep_prob)+'\n')
    f.write('learn_rate='+str(learn_rate)+'\n')
    f.write('balance_gate=:'+str(balance_gate)+'\n')
    f.write('best_result: '+str(np.max(np.array(test_acc)))+'\n')
#    f.write(str(np.max(np.array(test_acc)))+'\n')
    f.write('all train acc:\n')
    f.write(str(train_acc))
    f.write("\n")
    f.write('\ndev_acc:\n')
    f.write(str(dev_acc))
    f.write("\n")
    f.write('\ntest_acc:\n')
    f.write(str(test_acc))
    f.write("\n")
    dev_acc_2 = []
    for i in dev_acc:
        a =sum(i)
        dev_acc_2.append(a)

    max_ind = dev_acc_2.index(max(dev_acc_2))

    # max_ind = dev_acc.index(max(dev_acc))
    a = train_acc[max_ind]
    b = dev_acc[max_ind]
    c = test_acc[max_ind]
    f.write("\n")
    f.write("\nWhen batch num =" + str(max_ind+1)+ '*' +str(FLAGS.evaluate_every) + '， dev_acc is the biggest\n')
    f.write("Dev acc = " + str(b) + "\n")
    f.write("Test acc = " + str(c) + "\n")
    f.write("\n")
    f.write("\n")
    f.write("The L2 regularization lambda is: "+ str(FLAGS.l2_reg_lambda)+'\n')
    f.write("0 means all trainset and testset, 1,2,3,4 mean the 4 subsets of trainset")
    np.save(save_path + 'train_' + which_set, train_all_softmax[max_ind])
    np.save(save_path + 'test_' + which_set, test_all_softmax[max_ind])
    f.close()

def main(argv=None):
    #文件处理
    x_train, y_train, x_dev, y_dev, x_test, y_test,x_test_len, word_embedding = preprocess()
    #模型训练
    train_acc, dev_acc, test_acc, train_all_softmax, test_all_softmax = train(x_train, y_train, x_dev, y_dev, x_test, y_test, word_embedding)
    #保存需要的结果
    result_make(train_acc, dev_acc, test_acc,train_all_softmax,test_all_softmax,FLAGS.data_file[-1:])

if __name__ == '__main__':
    main(argv=None)
#    tf.app.run()