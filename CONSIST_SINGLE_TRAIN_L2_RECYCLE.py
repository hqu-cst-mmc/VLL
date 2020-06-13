import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
import random

import model_toupiao
import time

import input_sort
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


flags = tf.app.flags
flags.DEFINE_integer('batch_size', 16, 'Training batch size, default:30')
flags.DEFINE_float('base_lr',0.001, 'base learning rate')
flags.DEFINE_string('model_save_path', '0.25_CONSIST_SINGLE_L2_models', 'path to save model')
flags.DEFINE_integer('display', 1, 'print out the loss every dispaly iterations')
flags.DEFINE_integer('dimension',101, 'print out the loss every dispaly iterations')
flags.DEFINE_integer('max_iter',20, 'max iteration')
flags.DEFINE_integer('cpu_num', 8, 'num of cpu process to read data, default:6')


FLAGS = flags.FLAGS
model_name = "model.ckpt"
old_model = "./original_models/model.ckpt-79001"
gpu_num = 1
gpu_index = 1
momentum = 0.9
num_clip_max = 20
weight_coe = 0.25

def consist_matrix(x,y1):

    new = tf.stack([x])
    y1 = tf.concat([y1,new], 0)

    return y1

def distribution(i,x,y1,y2,labels):

    y1=tf.cond(tf.cast(tf.argmax(x) == labels, tf.bool),
            lambda : consist_matrix(x, y1),
            lambda : y1)
    y2=tf.cond(tf.cast(tf.argmax(x) != labels, tf.bool),
            lambda : consist_matrix(x, y2),
            lambda : y2)
    i = i + 1
    return i, y1,y2

def distance(j,x,y,z):
    x = x + tf.losses.mean_squared_error(z, y)
    # tf.reduce_mean(tf.abs(tf.subtract(z,y)), 0)
    # tf.losses.mean_squared_error(y, z)
    j = j + 1
    return j, x

def difference_loss(consist_loss,difference_softmax,consist_softmax_mean):

    j=tf.constant(1)
    num_difference_softmax = tf.shape(difference_softmax)
    j,consist_loss = tf.while_loop(lambda j,consist_loss:j<num_difference_softmax[0],
                                   lambda j,consist_loss:distance(j,consist_loss,difference_softmax[j],consist_softmax_mean),
                                   [j,consist_loss],
                                   shape_invariants=[j.get_shape(),
                                                     tf.TensorShape(None)])
    return consist_loss

def back():
    x = tf.zeros([FLAGS.dimension], tf.float32)
    # print("1111")
    return x

def com_mean_matrix(x,y):

    # print("2222")
    consist_mean = tf.reduce_sum(x, 0) / (tf.to_float(y)-1)

    return consist_mean

def com_mean_value(consist_loss1,num_difference_softmax):
    constant_value = tf.constant(1)
    x = consist_loss1 / tf.to_float((num_difference_softmax[0] - constant_value))
    return x

def f1(i, x, softmax_loss):
    # softmax_loss= 0.0
    zero_tensor = tf.zeros([1, 101], tf.float32)
    y = x
    for j in range(i + 1, num_clip_max):
        comparatively = tf.cond(tf.cast(x[j] != zero_tensor[0], tf.bool), lambda: f3(y, i, j), lambda: f2())
        # comparatively = tf.sqrt(tf.reduce_sum(tf.squared_difference(x[i], x[j]), 0))
        softmax_loss = softmax_loss + comparatively
    return softmax_loss


def f2():
    return 0.0


def f3(x, i, j):

    comparatively = tf.losses.mean_squared_error(x[i], x[j])

    return comparatively


def f4(x):
    return x

def recycle(softmax_feature_loss1):
    softmax_loss = 0.0
    zero_tensor = tf.zeros([1, 101], tf.float32)

    for i in range(num_clip_max):
        softmax_loss = tf.cond(tf.cast(softmax_feature_loss1[i] != zero_tensor[0], tf.bool),
                               lambda: f1(i, softmax_feature_loss1, softmax_loss), lambda: f4(softmax_loss))

    return softmax_loss

def train():

    num_clip = tf.placeholder(tf.int32)
    img_input = tf.placeholder(tf.float32, shape=(None, 16, 112, 112, 3))
    y_target = tf.placeholder(tf.int64, shape=(None))

    # for gpu_index in range(0, gpu_num):
    #     with tf.device('/gpu:%d' % gpu_index):
    #         y = model_toupiao.C3D(img_input, num_clip, dimensions=FLAGS.dimension,dropout=False,regularizer=True) # not label!
    with tf.device('/gpu:1'):
        y = model_toupiao.C3D(img_input, num_clip, dimensions=FLAGS.dimension,dropout=False,regularizer=True) # not label!
    y = tf.concat(y, 0)
    saver = tf.train.Saver()
    saver_new = tf.train.Saver()

    global_step = tf.Variable(0, trainable=False)

    varlist_weight = []
    varlist_bias = []
    trainable_variables = tf.trainable_variables()
    for var in trainable_variables:
        if 'weight' in var.name:
            varlist_weight.append(var)
        elif 'bias' in var.name:
            varlist_bias.append(var)

    learning_rate = tf.train.exponential_decay(FLAGS.base_lr, global_step, 20000, 0.1,staircase=True)

    opt_weight = tf.train.GradientDescentOptimizer(learning_rate)
    opt_bias = tf.train.GradientDescentOptimizer(learning_rate)

    softmax_feature_loss = tf.nn.softmax(y)
    softmax_loss = 0.0

    num_class = tf.shape(softmax_feature_loss)

    constant_value = tf.constant(1)
    w = tf.constant(0)
    consist_loss = 0.0
    consist_loss = tf.to_float(consist_loss)

    same_softmax = tf.zeros([1, FLAGS.dimension], tf.float32)
    difference_softmax = tf.zeros([1, FLAGS.dimension], tf.float32)

    w, same_softmax, difference_softmax = tf.while_loop(lambda w, same_softmax,difference_softmax: w<num_class[0],
                                                        lambda w,same_softmax, difference_softmax: distribution(w,softmax_feature_loss[w], same_softmax, difference_softmax, y_target[w]),
                                                        [w,same_softmax,difference_softmax],
                                                        shape_invariants=[w.get_shape(),
                                                                          tf.TensorShape([None,None]),
                                                                          tf.TensorShape([None,None])])
    num_diference_softmax = tf.shape(difference_softmax)
    num_same_softmax = tf.shape(same_softmax)
    mk = num_clip_max - num_class[0]
    softmax_feature_loss1 = tf.pad(softmax_feature_loss, [[0, mk], [0, 0]], "CONSTANT")
    dee = num_class[0] + constant_value

    same_softmax_mean_matrix = tf.cond(tf.equal(num_same_softmax[0],constant_value),
                                   lambda:back(),
                                   lambda:com_mean_matrix(same_softmax, num_same_softmax[0])
                                   )

    same_all_recycle = tf.cond(tf.equal(num_same_softmax[0], dee),
                            lambda: recycle(softmax_feature_loss1),
                            lambda: f2())

    consist_loss1 = tf.cond(tf.equal(num_diference_softmax[0], constant_value),
                           lambda : tf.to_float(consist_loss),
                           lambda : difference_loss(consist_loss,difference_softmax,same_softmax_mean_matrix))


    consist_loss2 = tf.cond(tf.equal(num_diference_softmax[0], dee),
                            lambda :recycle(softmax_feature_loss1),
                            lambda :com_mean_value(consist_loss1,num_diference_softmax))

    consist_end = tf.cond(tf.equal(num_same_softmax[0], dee),
                          lambda :same_all_recycle,
                          lambda :consist_loss2)

    cross_mean_loss = tf.reduce_mean(
                  tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_target,logits=y)
                  )

    # cross_mean_loss = tf.reduce_mean(tf.squared_difference(y, y_target))

    weight_decay_loss = tf.add_n(tf.get_collection('weight_decay_loss'))

    loss = (1-weight_coe)*cross_mean_loss + weight_decay_loss + weight_coe*consist_end
    # loss = weight_decay_loss + softmax_loss


    tf.summary.scalar('cross_mean_loss', cross_mean_loss)
    tf.summary.scalar('weight_decay_loss', weight_decay_loss)
    tf.summary.scalar('mean_loss', consist_end)
    tf.summary.scalar('total_loss', loss)

    grad_weight = opt_weight.compute_gradients(loss, varlist_weight)
    grad_bias = opt_bias.compute_gradients(loss, varlist_bias)
    apply_gradient_op_weight = opt_weight.apply_gradients(grad_weight)
    apply_gradient_op_bias = opt_bias.apply_gradients(grad_bias, global_step=global_step)
    train_op = tf.group(apply_gradient_op_weight, apply_gradient_op_bias)

    merged = tf.summary.merge_all()

    rgb_list = 'list/train_01.txt'
    txt = open(rgb_list, "r")
    txt = list(txt)
    txt_length = len(txt)
    sort_list = 'list/sort_train.txt'
    sort_txt = open(sort_list, "r")
    sort_txt = list(sort_txt)
    # sort_length = len(sort_txt)


    video_index = list(range(len(sort_txt)))
    random.shuffle(video_index)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        train_writer = tf.summary.FileWriter('./visual_logs/0.25_train_single_l2', sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)

        var = tf.global_variables()  # 获取所有变量
        print(var)

        var_to_restore = [val for val in var if 'conv_1' in val.name or 'conv_2' in val.name]
        var_to_restore1 = [val for val in var if 'conv_3' in val.name or 'conv_4' in val.name]
        var_to_restore2 = [val for val in var if 'conv_5' in val.name]
        # var_to_restore3 = [val for val in var if 'fc6' in val.name or 'fc7' in val.name]
        # var_to_restore4 = [val for val in var if 'fc8' in val.name]

        # var_to_restore_all = var_to_restore + var_to_restore1 + var_to_restore2 + var_to_restore3 + var_to_restore4
        var_to_restore_all = var_to_restore + var_to_restore1 + var_to_restore2

        print(var_to_restore_all)

        saver = tf.train.Saver(var_to_restore_all)
        # saver = tf.train.Saver()
        saver.restore(sess, old_model)

        var_to_init = [val for val in var if 'fc6' in val.name or 'fc7' in val.name]
        var_to_init1 = [val for val in var if 'fc8' in val.name]
        var_to_init_all = var_to_init + var_to_init1
        # tf.initialize_variables(var_to_init)
        # tf.initialize_variables(var_to_init1)

        sess.run(tf.variables_initializer(var_to_init_all))
        step = 0

        for i in range(FLAGS.max_iter):
            for sort_index in video_index:  # sort_index是sort文件里面的索引， 得到sort_line 是rgb文件的索引
                s_index = 0

                sort_line = sort_txt[sort_index].strip('\n')      #

                line = txt[int(sort_line)].strip('\n').split(" ")
                # line = txt[106332].strip('\n').split(" ")

                original_path = line[0]
                index = int(sort_line)
                # index = 106332
                while index <= txt_length :
                    start_time = time.time()
                    line1 = txt[index].strip('\n').split(" ")

                    new_path = line1[0]
                    if (new_path == original_path):
                        index = index + 1
                        s_index = s_index + 1

                    else:
                        num = s_index
                        if num > num_clip_max:
                            num = num_clip_max
                        original_path = new_path

                        next_start_pos = int(sort_line)

                        train_images, train_labels, next_start_pos, filename, start_filename, valid_len_train = input_sort.read_clip_and_label(
                            rgb_list,
                            num,
                            start_pos=next_start_pos
                            # shuffle=True
                        )


                        duration = time.time() - start_time
                        print('read data time %.3f sec' % (duration))

                        x, summary, loss_value, ce_loss, _, old_weight = sess.run([
                            consist_loss1, merged, loss, cross_mean_loss, train_op, grad_weight], feed_dict={
                            img_input: train_images,
                            y_target: train_labels,
                            num_clip: num
                        })

                        if step % (FLAGS.display) == 0:
                            # print("softmax_feature1:", we[0])
                            print("consist_loss:", x)
                            # print("jieguo:", losss)
                            print("cross_mean_loss:", ce_loss)
                            print("loss:", loss_value)
                            train_writer.add_summary(summary, step)
                        duration = time.time() - start_time
                        print('Step %d: %.3f sec' % (step, duration))


                        if (step) % 10 == 0:
                            saver_new.save(sess, os.path.join(FLAGS.model_save_path, model_name), global_step=global_step)
                        step = step + 1
                        break


def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()
