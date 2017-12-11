import os
import json
import cntk
import argparse
import numpy as np
from cntk.train import Trainer
from resnet import resnet_basic_inc, resnet_basic_stack
from cntk import learning_parameter_schedule, momentum_schedule
from cntk.io import StreamDef, StreamDefs, MinibatchSource, CBFDeserializer
from cntk.layers import Dense, Sequential, Label, LSTM, Recurrence, AttentionModel, Dropout, BatchNormalization, MaxPooling

# Model dimensions
frame_height = 120
frame_width = 120
num_channels = 1
num_classes = 66
hidden_dim = 66
model_name = 'lsa64.model'

# Dataset partition sizes
test_size = 2560
train_size = 640


def cbf_reader(path, is_training, max_samples):
    """
    Returns a MinibatchSource for data at the given path
    :param path: Path to a CBF file
    :param is_training: Set to true if reader is for training set, else false
    :param max_samples: Max no. of samples to read
    """
    deserializer = CBFDeserializer(path, StreamDefs(
        label=StreamDef(field='label', shape=num_classes, is_sparse=True),
        pixels=StreamDef(field='pixels', shape=num_channels * frame_height * frame_width, is_sparse=False)
    ))

    return MinibatchSource(deserializer, randomize=is_training, max_samples=max_samples)


def resnet_model(layer_input):
    layer1 = resnet_basic_stack(layer_input, 1, (3, 3), 6, (1, 1), prefix='conv1')
    layer1 = MaxPooling((3, 3), (2, 2), name='pool1')(layer1)
    layer1 = Dropout(0.3, name='drop1')(layer1)

    layer2 = resnet_basic_inc(layer1, (3, 3), 8, (2, 2), prefix='conv21')
    layer2 = resnet_basic_stack(layer2, 1, (3, 3), 8, (1, 1), prefix='conv22')
    layer2 = Dropout(0.3, name='drop2')(layer2)

    layer3 = resnet_basic_inc(layer2, (3, 3), 10, (2, 2), prefix='conv31')
    layer3 = resnet_basic_stack(layer3, 1, (3, 3), 10, (1, 1), prefix='conv32')
    layer3 = Dropout(0.3, name='drop3')(layer3)

    layer4 = resnet_basic_inc(layer3, (3, 3), 10, (2, 2), prefix='conv41')
    layer4 = resnet_basic_stack(layer4, 1, (3, 3), 10, (1, 1), prefix='conv42')
    layer4 = Dropout(0.3, name='drop4')(layer4)

    return layer4


def create_model():
    with cntk.layers.default_options(enable_self_stabilization=True, go_backwards=False):
        # Encoder: (input*) -> (h0, c0)
        encode = Sequential([
            resnet_model(cntk.placeholder()),
            Dense(hidden_dim, name='fc'),
            Dropout(0.3, name='drop_fc'),
            BatchNormalization(map_rank=1, normalization_time_constant=4096, name='bn_fc'),
            Recurrence(LSTM(hidden_dim, name='lstm_e'), return_full_state=True, name='rec_e'),
            (Label('encoded_h'), Label('encoded_c'))
        ])

        rec_block = LSTM(hidden_dim, name='rec_d')
        attention = AttentionModel(hidden_dim, name='attention')
        drop = Dropout(0.3, name='drop_out')
        output_projection = Dense(num_classes, name='output')

        @cntk.Function
        def decode(history, layer_input):
            encoded_input = encode(layer_input)
            r = history

            @cntk.Function
            def lstm_with_attention(dh, dc, x):
                h_att = attention(encoded_input.outputs[0], dh)
                x = cntk.splice(x, h_att)
                return rec_block(dh, dc, x)

            r = Recurrence(lstm_with_attention)(r)
            r = drop(r)
            r = output_projection(r)
            return r

    return decode


def create_model_train(s2smodel, sentence_start):
    """
    Model used in training (history is known from labels)
    NOTE: the labels must not contain the initial <s>
    """

    @cntk.Function
    def model_train(input_var, labels):
        past_labels = cntk.layers.Delay(initial_state=sentence_start)(labels)
        return s2smodel(past_labels, input_var)

    return model_train


def create_model_greedy(s2smodel, input_sequence, sentence_start):
    @cntk.Function
    @cntk.layers.Signature(input_sequence[cntk.layers.Tensor[num_channels, frame_height, frame_width]])
    def model_greedy(input_var):
        # Subtract previous frame from next frame
        s1 = cntk.sequence.slice(input_var, 1, 20)
        s2 = cntk.sequence.slice(input_var, 0, 19)
        layer_input = s1 - s2

        unfold = cntk.layers.UnfoldFrom(lambda history: s2smodel(history, layer_input) >> cntk.hardmax, length_increase=0.1)
        return unfold(initial_state=sentence_start, dynamic_axes_like=input_var)

    return model_greedy


def create_criterion_function(model, input_sequence, label_sequence):
    @cntk.Function
    @cntk.layers.Signature(input_var=input_sequence[cntk.layers.Tensor[num_channels, frame_height, frame_width]],
                           labels=label_sequence[cntk.layers.Tensor[num_classes]])
    def criterion(input_var, labels):
        # Subtract previous frame from next frame
        s1 = cntk.sequence.slice(input_var, 1, 20)
        s2 = cntk.sequence.slice(input_var, 0, 19)
        layer_input = s1 - s2

        # Remove BOS and EOS tags from label
        label_tail = cntk.sequence.slice(labels, 0, 2, name='label_tail')
        label_head = cntk.sequence.slice(label_tail, 1, 0, name='label_head')

        z = model(layer_input, label_head)
        ce = cntk.cross_entropy_with_softmax(z, label_head)
        errs = cntk.classification_error(z, label_head)
        return ce, errs

    return criterion


def create_sparse_to_dense(vocab_dim, input_sequence):
    """
    Dummy function for printing the input sequence.
    """
    i = cntk.Constant(np.eye(vocab_dim))

    @cntk.Function
    @cntk.layers.Signature(input_sequence[cntk.layers.SparseTensor[vocab_dim]])
    def no_op(input_var):
        return cntk.times(input_var, i)

    return no_op


def format_sequences(sequences, i2w):
    """
    Given a tensor and vocabulary, print the output
    """
    return [' '.join([i2w[np.argmax(w)] for w in s]) for s in sequences]


def test(model_greedy, reader, minibatch_size, sparse_to_dense, i2w, pp):
    while True:
        mb = reader.next_minibatch(minibatch_size)
        if not mb:
            break

        labels = format_sequences(sparse_to_dense(mb[reader.streams.label]), i2w)

        outputs = model_greedy(mb[reader.streams.pixels])
        outputs = format_sequences(outputs, i2w)
        outputs = [output for output in outputs]

        for i in range(len(labels)):
            pp.write('Eval', '{} -> {}'.format(labels[i], outputs[i]))


def main(params):
    # Create output and log directories if they don't exist
    if not os.path.isdir(params['output_folder']):
        os.makedirs(params['output_folder'])

    if not os.path.isdir(params['log_folder']):
        os.makedirs(params['log_folder'])

    with open('./output/label.json', 'r') as jfile:
        label_dict = json.load(jfile)
        label_dict['BOS'] = 64
        label_dict['EOS'] = 65

    i2w = {label_dict[w]: w for w in label_dict}

    start = np.zeros((num_classes,), dtype=np.float32)
    start[64] = 1
    sentence_start = cntk.Constant(start)

    input_axis = cntk.Axis('inputAxis')
    label_axis = cntk.Axis('labelAxis')
    input_sequence = cntk.layers.SequenceOver[input_axis]
    label_sequence = cntk.layers.SequenceOver[label_axis]

    # Create the model and criterion
    model = create_model()
    model_train = create_model_train(model, sentence_start)
    model_greedy = create_model_greedy(model, input_sequence, sentence_start)
    criterion = create_criterion_function(model_train, input_sequence, label_sequence)

    # Create the learner
    reg_weight = 0.005
    mm_schedule = momentum_schedule(0.90)
    lr_schedule = learning_parameter_schedule([(200, 0.1)], minibatch_size=params['minibatch_size'])
    learner = cntk.adam(model_train.parameters, lr_schedule, mm_schedule, l2_regularization_weight=reg_weight)

    # Create writers for logging
    log_file = os.path.join(params['log_folder'], 'log.txt')
    pp_writer = cntk.logging.ProgressPrinter(freq=10, tag='Training', num_epochs=params['max_epochs'], log_to_file=log_file)
    tb_writer = cntk.logging.TensorBoardProgressWriter(freq=10, log_dir=params['log_folder'])

    # Create readers
    train_path = os.path.join(params['input_folder'], 'train{}.cbf'.format(params['prefix']))
    test_path = os.path.join(params['input_folder'], 'test{}.cbf'.format(params['prefix']))

    train_reader = cbf_reader(train_path, is_training=True, max_samples=cntk.io.INFINITELY_REPEAT)
    cv_reader = cbf_reader(test_path, is_training=False, max_samples=cntk.io.INFINITELY_REPEAT)

    # Create trainer and training session
    trainer = Trainer(None, criterion, [learner], [pp_writer, tb_writer])

    sparse_to_dense = create_sparse_to_dense(num_classes, input_sequence)
    if params['restore']:
        trainer.restore_from_checkpoint(model_name)

    try:
        cntk.logging.log_number_of_parameters(model_train)

        for epoch in range(params['max_epochs']):
            sequences = 0
            while sequences < params['epoch_size']:
                # Get next minibatch of training data and train
                mb_train = train_reader.next_minibatch(params['minibatch_size'])
                trainer.train_minibatch({
                    criterion.arguments[0]: mb_train[train_reader.streams.pixels],
                    criterion.arguments[1]: mb_train[train_reader.streams.label]
                })
                sequences += mb_train[train_reader.streams.pixels].num_sequences

            trainer.summarize_training_progress()

            # Validation on the test set
            sequences = 0
            while sequences < params['cv_seqs']:
                mb_cv = cv_reader.next_minibatch(params['minibatch_size'])
                trainer.test_minibatch({
                    criterion.arguments[0]: mb_cv[cv_reader.streams.pixels],
                    criterion.arguments[1]: mb_cv[cv_reader.streams.label]
                })
                sequences += mb_cv[cv_reader.streams.pixels].num_sequences
            trainer.summarize_test_progress()
            tb_writer.flush()

            if (epoch + 1) % 20 == 0:
                test_reader = cbf_reader(test_path, is_training=False, max_samples=cntk.io.FULL_DATA_SWEEP)
                test(model_greedy, test_reader, params['minibatch_size'], sparse_to_dense, i2w, pp_writer)

        test_reader = cbf_reader(test_path, is_training=False, max_samples=cntk.io.FULL_DATA_SWEEP)
        test(model_greedy, test_reader, params['minibatch_size'], sparse_to_dense, i2w, pp_writer)
    finally:
        path = os.path.join(params['output_folder'], 'final_model.dnn')
        model.save(path)
        print('Saved final model to', path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-if', '--input_folder', help='Directory where dataset is located', required=False, default='dataset')
    parser.add_argument('-of', '--output_folder', help='Directory for models and checkpoints', required=False, default='models')
    parser.add_argument('-lf', '--log_folder', help='Directory for log files', required=False, default='logs')
    parser.add_argument('-n', '--num_epochs', help='Total number of epochs to train', type=int, required=False, default=200)
    parser.add_argument('-m', '--minibatch_size', help='Minibatch size in samples', type=int, required=False, default=640)
    parser.add_argument('-e', '--epoch_size', help='Epoch size', type=int, required=False, default=train_size)
    parser.add_argument('-r', '--restore', help='Indicates whether to resume from previous checkpoint', action='store_true')
    parser.add_argument('-c', '--cv_seqs', help='The number of samples to use for cross validation', type=int, required=False, default=test_size)
    parser.add_argument('-p', '--prefix', help='The prefix for the train/test datasets', required=False, default='')

    args = parser.parse_args()
    main({
        'input_folder': args.input_folder,
        'output_folder': args.output_folder,
        'log_folder': args.log_folder,
        'max_epochs': args.num_epochs,
        'minibatch_size': args.minibatch_size,
        'epoch_size': args.epoch_size,
        'restore': args.restore,
        'cv_seqs': args.cv_seqs,
        'prefix': args.prefix
    })
