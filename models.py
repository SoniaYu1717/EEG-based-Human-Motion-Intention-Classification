from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import params
import tensorflow as tf


def create_model(feature_input, model_architecture, is_training):
    if model_architecture == 'gru':
        return create_gru_model(feature_input, is_training)
    elif model_architecture == 'lstm':
        return create_lstm_model(feature_input, is_training)
    elif model_architecture == 'cnn_gru':
        return create_cnn_gru_model(feature_input, is_training)
    elif model_architecture == 'cnn_lstm':
        return create_cnn_lstm_model(feature_input, is_training)
    else:
        raise Exception('model_architecture argument ' + model_architecture +
                        ' not recognized, should be one of "lstm", "gru", "cnn_gru", "cnn_lstm".')


def load_variables_from_checkpoint(sess, start_checkpoint):
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, start_checkpoint)


def create_gru_model(feature_input, is_training):
    gru_params = {'num_layers': 1,
                  'hidden_size': 128}

    with tf.name_scope('model'):
        if is_training:
            keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            batch_size = params.BATCH_SIZE
        else:
            batch_size = 1

        with tf.variable_scope('GRU'):
            gru_cell = tf.nn.rnn_cell.GRUCell(gru_params['hidden_size'])
            if is_training:
                gru_cell = tf.nn.rnn_cell.DropoutWrapper(gru_cell, output_keep_prob=keep_prob)
            cell = tf.nn.rnn_cell.MultiRNNCell([gru_cell] * gru_params['num_layers'], state_is_tuple=True)

            initial_state = cell.zero_state(batch_size, tf.float32)

            if is_training:
                embeddings = tf.nn.dropout(feature_input, keep_prob)
            (outputs, final_state) = tf.nn.dynamic_rnn(cell, embeddings, initial_state=initial_state)
            output = tf.reshape(outputs[:, -1, :], [-1, gru_params['hidden_size']])
            weights = tf.get_variable("fc_w", [gru_params['hidden_size'], params.LABEL_COUNT], dtype=tf.float32)
            biases = tf.get_variable("fc_b", [params.LABEL_COUNT], dtype=tf.float32)
            logits = tf.add(tf.matmul(output, weights), biases, name="logits")

    if is_training:
        return logits, keep_prob
    else:
        return logits


def create_lstm_model(feature_input, is_training):
    lstm_params = {'num_layers': 1,
                   'hidden_size': 128}

    with tf.name_scope('model'):
        if is_training:
            keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            batch_size = params.BATCH_SIZE
        else:
            batch_size = 1

        with tf.variable_scope('LSTM'):
            lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_params['hidden_size'])
            if is_training:
                lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
            cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * lstm_params['num_layers'], state_is_tuple=True)

            initial_state = cell.zero_state(batch_size, tf.float32)

            if is_training:
                embeddings = tf.nn.dropout(feature_input, keep_prob)
            (outputs, final_state) = tf.nn.dynamic_rnn(cell, embeddings, initial_state=initial_state)
            output = tf.reshape(outputs[:, -1, :], [-1, lstm_params['hidden_size']])
            weights = tf.get_variable("fc_w", [lstm_params['hidden_size'], params.LABEL_COUNT], dtype=tf.float32)
            biases = tf.get_variable("fc_b", [params.LABEL_COUNT], dtype=tf.float32)
            logits = tf.add(tf.matmul(output, weights), biases, name="logits")

    if is_training:
        return logits, keep_prob
    else:
        return logits


def create_cnn_gru_model(feature_input, is_training):
    gru_params = {'num_layers': 1,
                  'hidden_size': 128}

    cnn_gru_params = {
        'init_stddev': 0.01,
        'embedding_size': 128
    }
    first_filter = {
        'width': 8,
        'height': 20,
        'count': 64
    }
    second_filter = {
        'width': 4,
        'height': 10,
        'count': 64
    }

    with tf.name_scope('model'):
        if is_training:
            keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            batch_size = params.BATCH_SIZE
        else:
            batch_size = 1

        with tf.variable_scope('CNN'):
            # batch_size * time_steps * input_size
            # ===> (batch_size * num_frames) * frame_length * input_size
            frames = tf.contrib.signal.frame(feature_input,
                                             frame_length=params.FRAME_SIZE,
                                             frame_step=params.FRAME_STRIDE,
                                             axis=1)
            net = tf.reshape(frames, [-1, params.FRAME_SIZE, params.FEATURE_DIMENSION, 1])
            # first layer
            first_weights = tf.Variable(
                tf.truncated_normal(
                    [first_filter['height'], first_filter['width'], 1, first_filter['count']],
                    stddev=cnn_gru_params['init_stddev']), name='first_w')
            first_bias = tf.Variable(tf.zeros([first_filter['count']]), name='first_b')
            first_conv = tf.nn.conv2d(net, first_weights, [1, 1, 1, 1], 'SAME') + first_bias
            first_relu = tf.nn.relu(first_conv)
            if is_training:
                first_dropout = tf.nn.dropout(first_relu, keep_prob)
            else:
                first_dropout = first_relu
            max_pool = tf.nn.max_pool(first_dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

            # second layer
            second_weights = tf.Variable(
                tf.truncated_normal(
                    [second_filter['height'], second_filter['width'], first_filter['count'], second_filter['count']],
                    stddev=cnn_gru_params['init_stddev']), name='second_w')
            second_bias = tf.Variable(tf.zeros([second_filter['count']]), name='second_b')
            second_conv = tf.nn.conv2d(max_pool, second_weights, [1, 1, 1, 1], 'SAME') + second_bias
            second_relu = tf.nn.relu(second_conv)
            if is_training:
                second_dropout = tf.nn.dropout(second_relu, keep_prob)
            else:
                second_dropout = second_relu
            second_conv_shape = second_dropout.get_shape()
            second_conv_output_width = second_conv_shape[2]
            second_conv_output_height = second_conv_shape[1]
            second_conv_element_count = int(
                second_conv_output_width * second_conv_output_height * second_filter['count'])
            flattened_second_conv = tf.reshape(second_dropout, [-1, second_conv_element_count])

            final_fc_weights = tf.Variable(
                tf.truncated_normal(
                    [second_conv_element_count, cnn_gru_params['embedding_size']],
                    stddev=cnn_gru_params['init_stddev']), name='fc_w')
            final_fc_bias = tf.Variable(tf.zeros([cnn_gru_params['embedding_size']]), name='fc_b')
            final_fc = tf.matmul(flattened_second_conv, final_fc_weights) + final_fc_bias

            embeddings = tf.reshape(final_fc, [batch_size, -1, cnn_gru_params['embedding_size']])

        with tf.variable_scope('GRU'):
            gru_cell = tf.nn.rnn_cell.GRUCell(gru_params['hidden_size'])
            if is_training:
                gru_cell = tf.nn.rnn_cell.DropoutWrapper(gru_cell, output_keep_prob=keep_prob)
            cell = tf.nn.rnn_cell.MultiRNNCell([gru_cell] * gru_params['num_layers'], state_is_tuple=True)

            initial_state = cell.zero_state(batch_size, tf.float32)

            if is_training:
                embeddings = tf.nn.dropout(embeddings, keep_prob)
            (outputs, final_state) = tf.nn.dynamic_rnn(cell, embeddings, initial_state=initial_state)
            output = tf.reshape(outputs[:, -1, :], [-1, gru_params['hidden_size']])
            weights = tf.get_variable("fc_w", [gru_params['hidden_size'], params.LABEL_COUNT], dtype=tf.float32)
            biases = tf.get_variable("fc_b", [params.LABEL_COUNT], dtype=tf.float32)
            logits = tf.add(tf.matmul(output, weights), biases, name="logits")

    if is_training:
        return logits, keep_prob
    else:
        return logits


def create_cnn_lstm_model(feature_input, is_training):
    lstm_params = {'num_layers': 2,
                   'hidden_size': 1024}

    cnn_lstm_params = {
        'init_stddev': 0.01,
        'embedding_size': 1024
    }
    first_filter = {
        'width': 3,
        'height': 3,
        'count': 32
    }
    second_filter = {
        'width': 3,
        'height': 3,
        'count': 64
    }
    third_filter = {
        'width': 3,
        'height': 3,
        'count': 128
    }

    with tf.name_scope('model'):
        if is_training:
            keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            batch_size = params.BATCH_SIZE
        else:
            batch_size = 1

        with tf.variable_scope('CNN'):
            net = tf.reshape(feature_input, [-1, 6, 7, 1])
            # first layer
            first_weights = tf.Variable(
                tf.truncated_normal(
                    [first_filter['height'], first_filter['width'], 1, first_filter['count']],
                    stddev=cnn_lstm_params['init_stddev']), name='first_w')
            first_bias = tf.Variable(tf.zeros([first_filter['count']]), name='first_b')
            first_conv = tf.nn.conv2d(net, first_weights, [1, 1, 1, 1], 'SAME') + first_bias
            first_relu = tf.nn.relu(first_conv)
            if is_training:
                first_dropout = tf.nn.dropout(first_relu, keep_prob)
            else:
                first_dropout = first_relu
            first_max_pool = tf.nn.max_pool(first_dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

            # second layer
            second_weights = tf.Variable(
                tf.truncated_normal(
                    [second_filter['height'], second_filter['width'], first_filter['count'],
                     second_filter['count']],
                    stddev=cnn_lstm_params['init_stddev']), name='second_w')
            second_bias = tf.Variable(tf.zeros([second_filter['count']]), name='second_b')
            second_conv = tf.nn.conv2d(first_max_pool, second_weights, [1, 1, 1, 1], 'SAME') + second_bias
            second_relu = tf.nn.relu(second_conv)
            if is_training:
                second_dropout = tf.nn.dropout(second_relu, keep_prob)
            else:
                second_dropout = second_relu
            second_max_pool = tf.nn.max_pool(second_dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

            # third layer
            third_weights = tf.Variable(
                tf.truncated_normal(
                    [third_filter['height'], third_filter['width'], second_filter['count'],
                     third_filter['count']],
                    stddev=cnn_lstm_params['init_stddev']), name='third_w')
            third_bias = tf.Variable(tf.zeros([third_filter['count']]), name='third_b')
            third_conv = tf.nn.conv2d(second_max_pool, third_weights, [1, 1, 1, 1], 'SAME') + third_bias
            third_relu = tf.nn.relu(third_conv)
            if is_training:
                third_dropout = tf.nn.dropout(third_relu, keep_prob)
            else:
                third_dropout = third_relu

            third_conv_shape = third_dropout.get_shape()
            third_conv_output_width = third_conv_shape[2]
            third_conv_output_height = third_conv_shape[1]
            third_conv_element_count = int(
                third_conv_output_width * third_conv_output_height * third_filter['count'])
            flattened_third_conv = tf.reshape(third_dropout, [-1, third_conv_element_count])

            final_fc_weights = tf.Variable(
                    tf.truncated_normal(
                        [third_conv_element_count, cnn_lstm_params['embedding_size']],
                        stddev=cnn_lstm_params['init_stddev']), name='fc_w')
            final_fc_bias = tf.Variable(tf.zeros([cnn_lstm_params['embedding_size']]), name='fc_b')
            final_fc = tf.matmul(flattened_third_conv, final_fc_weights) + final_fc_bias

            embeddings = tf.reshape(final_fc, [batch_size, -1, cnn_lstm_params['embedding_size']])

        with tf.variable_scope('LSTM'):
            lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_params['hidden_size'])
            if is_training:
                lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
            cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * lstm_params['num_layers'], state_is_tuple=True)

            initial_state = cell.zero_state(batch_size, tf.float32)

            if is_training:
                embeddings = tf.nn.dropout(embeddings, keep_prob)
            (outputs, final_state) = tf.nn.dynamic_rnn(cell, embeddings, initial_state=initial_state)
            output = tf.reshape(outputs[:, -1, :], [-1, lstm_params['hidden_size']])
            weights = tf.get_variable("fc_w", [lstm_params['hidden_size'], params.LABEL_COUNT], dtype=tf.float32)
            biases = tf.get_variable("fc_b", [params.LABEL_COUNT], dtype=tf.float32)
            logits = tf.add(tf.matmul(output, weights), biases, name="logits")

        if is_training:
            return logits, keep_prob
        else:
            return logits
