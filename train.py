from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import params
import reader
import models

import tensorflow as tf
from six.moves import xrange

FLAGS = None


def main(_):
    # ===================================== data input =====================================
    tf.logging.set_verbosity(tf.logging.INFO)

    sess = tf.InteractiveSession()

    data_reader = reader.Reader(FLAGS.data_dir)
    for set_index in ['training', 'validation', 'testing']:
        set_size = data_reader.labels[set_index].shape
        print('')
        print(set_index, set_size)

    # ===================================== create model ===================================
    feature_input = tf.placeholder(
        tf.float32, [None, params.LENGTH, 6, 7], name='feature')
    label_input = tf.placeholder(
        tf.int64, [None], name='label')

    logits, keep_prob = models.create_model(
        feature_input,
        FLAGS.model_architecture,
        is_training=True
    )

    # Define loss and optimizer
    # Create the back propagation and training evaluation machinery in the graph.
    with tf.name_scope('cross_entropy'):
        cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=label_input)
    tf.summary.scalar('cross_entropy', cross_entropy_mean)

    with tf.name_scope('train'):
        # optimizer = tf.train.AdamOptimizer(params.LEARNING_RATE)
        # gradients, variables = zip(*optimizer.compute_gradients(cross_entropy_mean))
        # G = gradients
        # gradients, _ = tf.clip_by_global_norm(gradients, 5)
        # train_step = optimizer.apply_gradients(zip(gradients, variables))
        train_step = tf.train.GradientDescentOptimizer(params.LEARNING_RATE).minimize(cross_entropy_mean)

    predicted_indices = tf.argmax(logits, 1)
    correct_prediction = tf.equal(predicted_indices, label_input)
    confusion_matrix = tf.confusion_matrix(label_input, predicted_indices, 4)
    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation_step)

    global_step = tf.train.get_or_create_global_step()
    increment_global_step = tf.assign(global_step, global_step + 1)
    saver = tf.train.Saver(tf.global_variables())

    for var in tf.trainable_variables():
        tf.summary.histogram(var.name, var)

    merged_summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/training', sess.graph)
    validation_writer = tf.summary.FileWriter(FLAGS.log_dir + '/validation')

    tf.global_variables_initializer().run()

    start_step = 1

    if FLAGS.start_checkpoint:
        models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)
        start_step = global_step.eval(session=sess)

    tf.logging.info('Training from step: %d ', start_step)

    # Save graph.pbtxt.
    tf.train.write_graph(sess.graph_def, FLAGS.train_dir,
                         FLAGS.model_architecture + '.pbtxt')

    # =================================== Training loop =====================================
    for training_step in xrange(start_step, params.TRAINING_STEPS):
        training_features, training_labels = data_reader.get_mesh(params.BATCH_SIZE, 0, 'training')

        # print('features', training_features.shape)
        # print('labels', training_labels)
        train_summary, train_accuracy, cross_entropy_value, _, _ = sess.run(
            [merged_summaries, evaluation_step, cross_entropy_mean, train_step, increment_global_step],
            feed_dict={
                feature_input: training_features,
                label_input: training_labels,
                keep_prob: params.KEEP_PROB
            })
        train_writer.add_summary(train_summary, training_step)
        tf.logging.info('Step #%d: accuracy %.2f%%, cross entropy %f' %
                        (training_step, train_accuracy * 100, cross_entropy_value))
        is_last_step = (training_step == params.TRAINING_STEPS)

        # =================================== validation ====================================
        if (training_step % params.EVAL_STEP_INTERVAL) == 0 or is_last_step:
            set_size = int(data_reader.labels['validation'].shape[0])
            tf.logging.info('Validation set size = %d', set_size)
            total_accuracy = 0
            total_conf_matrix = None
            batch_count = 0
            for i in xrange(0, set_size, params.BATCH_SIZE):
                if (set_size - i) < params.BATCH_SIZE:
                    break
                batch_count += 1

                validation_features, validation_labels = data_reader.get_mesh(params.BATCH_SIZE, i, 'validation')

                # Run a validation step and capture training summaries for TensorBoard
                # with the `merged` op.
                validation_summary, validation_accuracy, conf_matrix = sess.run(
                    [merged_summaries, evaluation_step, confusion_matrix],
                    feed_dict={
                        feature_input: validation_features,
                        label_input: validation_labels,
                        keep_prob: 1.0
                    })
                validation_writer.add_summary(validation_summary, training_step)

                total_accuracy += validation_accuracy
                if total_conf_matrix is None:
                    total_conf_matrix = conf_matrix
                else:
                    total_conf_matrix += conf_matrix

            tf.logging.info('Confusion Matrix:\n %s' % total_conf_matrix)
            tf.logging.info('Step %d: Validation accuracy = %.2f%% (N=%d)' %
                            (training_step, total_accuracy / batch_count * 100,
                             params.BATCH_SIZE * batch_count))

        # Save the model checkpoint periodically.
        if training_step % params.SAVE_STEP_INTERVAL == 0 or training_step == params.TRAINING_STEPS:
            checkpoint_path = os.path.join(FLAGS.train_dir,
                                           FLAGS.model_architecture + '.ckpt')
            tf.logging.info('Saving to "%s-%d"', checkpoint_path, training_step)
            saver.save(sess, checkpoint_path, global_step=training_step)

    # ========================================= test ===========================================
    set_size = int(data_reader.labels['testing'].shape[0])
    tf.logging.info('Testing set size = %d', set_size)
    total_accuracy = 0
    total_conf_matrix = None
    batch_count = 0
    for i in xrange(0, set_size, params.BATCH_SIZE):
        if (set_size - i) < params.BATCH_SIZE:
            break
        batch_count += 1

        test_features, test_labels = data_reader.get_mesh(params.BATCH_SIZE, i, 'testing')

        test_accuracy, conf_matrix = sess.run(
            [evaluation_step, confusion_matrix],
            feed_dict={
                feature_input: test_features,
                label_input: test_labels,
                keep_prob: 1.0
            })
        total_accuracy += test_accuracy
        if total_conf_matrix is None:
            total_conf_matrix = conf_matrix
        else:
            total_conf_matrix += conf_matrix
    tf.logging.info('Confusion Matrix:\n %s' % total_conf_matrix)
    tf.logging.info('Final test accuracy = %.2f%% (N=%d)' %
                    (total_accuracy / batch_count * 100,
                     params.BATCH_SIZE * batch_count))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./project_datasets/',
        help='Project datasets.')
    parser.add_argument(
        '--log_dir',
        type=str,
        default='./log',
        help='Where to save summary logs for TensorBoard.')
    parser.add_argument(
        '--train_dir',
        type=str,
        default='./weights',
        help='Directory to write event logs and checkpoint.')
    parser.add_argument(
        '--model_architecture',
        type=str,
        default='cnn_lstm',  # gru, lstm, cnn_gru
        help='What model architecture to use.')
    parser.add_argument(
        '--start_checkpoint',
        type=str,
        default='',
        help='If specified, restore this pretrained model before any training.')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
