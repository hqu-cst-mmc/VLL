
import os

import tensorflow as tf

import model_toupiao
import time
import input_data
import numpy as np
import input_data_c3d
import input

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

flags = tf.app.flags
flags.DEFINE_integer('batch_size', 1, 'Training batch size, default:30')
flags.DEFINE_float('base_lr',0.001, 'base learning rate')
flags.DEFINE_string('model_save_path', 'motion_pattern_all_new_global', 'path to save model')
flags.DEFINE_integer('display', 1, 'print out the loss every dispaly iterations')
flags.DEFINE_integer('dimension',101, 'print out the loss every dispaly iterations')
flags.DEFINE_integer('max_iter',80000, 'max iteration')
flags.DEFINE_integer('cpu_num', 8, 'num of cpu process to read data, default:6')

# feature_dir = "./feature"
FLAGS = flags.FLAGS
model_name = "/media/alice/datafile/video_repres_finetune/4096_models_100000/model.ckpt-49001"
gpu_num = 1
momentum = 0.9


def feature_extract():

    num_clip = tf.placeholder(tf.int32)
    img_input = tf.placeholder(tf.float32, shape=(None, 16, 112, 112, 3))
    # y_target = tf.placeholder(tf.float32, shape=(FLAGS.batch_size*gpu_num,FLAGS.dimension))
    y=[]
    for gpu_index in range(0, gpu_num):
        with tf.device('/gpu:%d' % gpu_index):
            y = model_toupiao.C3D(img_input,num_clip, dimensions=FLAGS.dimension,dropout=False,regularizer=True) # not label!
    y = tf.concat(y, 0)
    score_A = tf.nn.softmax(y)
    score_label = tf.argmax(score_A,1)
    # norm_score = tf.reduce_mean(tf.nn.softmax(y), 0)
    # pre_label = np.argmax(np.bincount(score_label))

    saver = tf.train.Saver()

    varlist_weight = []
    varlist_bias = []
    trainable_variables = tf.trainable_variables()
    for var in trainable_variables:
        if 'weight' in var.name:
            varlist_weight.append(var)
        elif 'bias' in var.name:
            varlist_bias.append(var)


    rgb_list = 'list/test_01.list'
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter('./visual_logs/train', sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)
        saver.restore(sess, model_name)

        next_start_pos = 0

        #a = 32827
        # data_number = input.get_data_num(rgb_list)
        # loop = int(data_number / (FLAGS.batch_size * gpu_num))
        txt = open(rgb_list,"r")
        txt = list(txt)
        length = len(txt)
        # txt = list(txt)

        # original_score = np.zeros([1, 101],np.float32)
        # predict_score1 = np.zeros([1, 101],np.float32)
        # num = 0
        index = 0
        s_index =0
        line = txt[index].strip('\n').split(" ")

        original_path = line[0]
        while index < length:

            write_file = open("predict+toupiao_2.txt", "a+")
            line1 = txt[index].strip('\n').split(" ")

            new_path = line1[0]
            if (new_path == original_path):

                index = index + 1

            else:
                num = index-s_index
                s_index = index
                original_path = new_path

                train_images, train_labels, next_start_pos, filename, _, valid_len_train = input_data_c3d.read_clip_and_label(
                    rgb_list,
                    num,
                    start_pos=next_start_pos
                )

                predict_score = score_label.eval(
                    session=sess,
                    feed_dict={img_input: train_images,
                               num_clip: num}
                )

                true_label = train_labels[0],
                top1_predicted_label = np.argmax(np.bincount(predict_score))
                # Write results: true label, class prob for true label, predicted label, class prob for predicted label
                write_file.write('{}, {}\n'.format(
                    true_label[0],
                    top1_predicted_label
                ))

                continue


            #
            # mean_score = np.sum(predict_score1, 0)
            # # len_one_class = len(predict_score1) - 1
            # predict_score1 = np.concatenate((predict_score1, predict_score), 0)
            #

            # predict_score1 = np.zeros([1, 101], np.float32)

            # predict_score1 = []
            # mean_score = []

        write_file.close()
        print("done")




def main(_):
    feature_extract()


if __name__ == '__main__':
    tf.app.run()
