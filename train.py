import os
import cntk
import argparse
from resnet import resnet_basic_inc, resnet_basic_stack
from cntk import learning_parameter_schedule, momentum_schedule
from cntk.layers import Dense, Sequential, Label, Dropout, MaxPooling
from cntk.io import StreamDef, StreamDefs, MinibatchSource, CBFDeserializer
from cntk.train import Trainer, TestConfig, CrossValidationConfig, training_session, CheckpointConfig

# Model dimensions
frame_height = 120
frame_width = 120
num_channels = 1
sequence_length = 20
num_classes = 64
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
        pixels=StreamDef(field='pixels', shape=frame_height * frame_width * sequence_length, is_sparse=False)
    ))

    return MinibatchSource(deserializer, randomize=is_training, max_samples=max_samples)


def resnet_model(layer_input):
    layer1 = resnet_basic_stack(layer_input, 2, (3, 3), 19, (1, 1), prefix='conv1')
    layer1 = MaxPooling((3, 3), strides=(2, 2), name='pool1')(layer1)
    layer1 = Dropout(0.3, name='drop1')(layer1)

    layer2 = resnet_basic_inc(layer1, (3, 3), 8, (2, 2), prefix='conv21')
    layer2 = resnet_basic_stack(layer2, 1, (3, 3), 8, (1, 1), prefix='conv22')
    layer2 = Dropout(0.3, name='drop2')(layer2)

    layer3 = resnet_basic_inc(layer2, (3, 3), 16, (2, 2), prefix='conv31')
    layer3 = resnet_basic_stack(layer3, 1, (3, 3), 16, (1, 1), prefix='conv32')
    layer3 = Dropout(0.3, name='drop3')(layer3)

    layer4 = resnet_basic_inc(layer3, (3, 3), 32, (2, 2), prefix='conv41')
    layer4 = resnet_basic_stack(layer4, 1, (3, 3), 32, (1, 1), prefix='conv42')
    layer4 = Dropout(0.3, name='drop4')(layer4)

    return layer4


def create_network():
    # Create the input and target variables
    input_var = cntk.input_variable((sequence_length, frame_height, frame_width), name='input_var')
    target_var = cntk.input_variable((num_classes,), is_sparse=True, name='target_var')

    input_head = cntk.slice(input_var, axis=0, begin_index=0, end_index=19)
    input_tail = cntk.slice(input_var, axis=0, begin_index=1, end_index=20)
    diff = input_tail - input_head

    model = Sequential([
        resnet_model(cntk.placeholder()), Label('resnet'),
        Dense(num_classes, name='output')
    ])(diff)

    return {
        'input': input_var,
        'target': target_var,
        'model': model,
        'loss': cntk.cross_entropy_with_softmax(model, target_var),
        'metric': cntk.classification_error(model, target_var)
    }


def main(params):
    # Create output and log directories if they don't exist
    if not os.path.isdir(params['output_folder']):
        os.makedirs(params['output_folder'])

    if not os.path.isdir(params['log_folder']):
        os.makedirs(params['log_folder'])

    # Create the network
    network = create_network()

    # Create readers
    train_reader = cbf_reader(os.path.join(params['input_folder'], 'train{}.cbf'.format(params['prefix'])), is_training=True,
                              max_samples=cntk.io.INFINITELY_REPEAT)
    cv_reader = cbf_reader(os.path.join(params['input_folder'], 'test{}.cbf'.format(params['prefix'])), is_training=True,
                           max_samples=params['cv_seqs'])
    test_reader = cbf_reader(os.path.join(params['input_folder'], 'test{}.cbf'.format(params['prefix'])), is_training=False,
                             max_samples=cntk.io.FULL_DATA_SWEEP)

    input_map = {
        network['input']: train_reader.streams.pixels,
        network['target']: train_reader.streams.label
    }

    # Create learner
    mm_schedule = momentum_schedule(0.90)
    lr_schedule = learning_parameter_schedule([(700, 0.001), (300, 0.0001)], minibatch_size=params['minibatch_size'])
    learner = cntk.adam(network['model'].parameters, lr_schedule, mm_schedule, l2_regularization_weight=0.0005,
                        epoch_size=params['epoch_size'], minibatch_size=params['minibatch_size'])

    # Use TensorBoard for visual logging
    log_file = os.path.join(params['log_folder'], 'log.txt')
    pp_writer = cntk.logging.ProgressPrinter(freq=10, tag='Training', num_epochs=params['max_epochs'], log_to_file=log_file)
    tb_writer = cntk.logging.TensorBoardProgressWriter(freq=10, log_dir=params['log_folder'], model=network['model'])

    # Create trainer and training session
    trainer = Trainer(network['model'], (network['loss'], network['metric']), [learner], [pp_writer, tb_writer])
    test_config = TestConfig(minibatch_source=test_reader, minibatch_size=params['minibatch_size'], model_inputs_to_streams=input_map)
    cv_config = CrossValidationConfig(minibatch_source=cv_reader, frequency=params['epoch_size'], minibatch_size=params['minibatch_size'],
                                      model_inputs_to_streams=input_map)
    checkpoint_config = CheckpointConfig(os.path.join(params['output_folder'], model_name), frequency=params['epoch_size'],
                                         restore=params['restore'])

    session = training_session(trainer=trainer,
                               mb_source=train_reader,
                               mb_size=params['minibatch_size'],
                               model_inputs_to_streams=input_map,
                               max_samples=params['epoch_size'] * params['max_epochs'],
                               progress_frequency=params['epoch_size'],
                               checkpoint_config=checkpoint_config,
                               cv_config=cv_config,
                               test_config=test_config)

    try:
        cntk.logging.log_number_of_parameters(network['model'])
        session.train()
    finally:
        path = os.path.join(params['output_folder'], 'final_model.dnn')
        network['model'].save(path)
        print('Saved final model to', path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-if', '--input_folder', help='Directory where dataset is located', required=False, default='dataset')
    parser.add_argument('-of', '--output_folder', help='Directory for models and checkpoints', required=False, default='models')
    parser.add_argument('-lf', '--log_folder', help='Directory for log files', required=False, default='logs')
    parser.add_argument('-n', '--num_epochs', help='Total number of epochs to train', type=int, required=False, default=1000)
    parser.add_argument('-m', '--minibatch_size', help='Minibatch size in samples', type=int, required=False, default=32)
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
