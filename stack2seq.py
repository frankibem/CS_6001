import os
import sys
import logging
import argparse
import subprocess
from multiprocessing import Process, JoinableQueue
from cntk.io import StreamDef, StreamDefs, MinibatchSource, CBFDeserializer, FULL_DATA_SWEEP

frame_height = 120
frame_width = 120
sequence_length = 20
num_channels = 1
num_classes = 64
num_subjects = 10


def get_logger(prefix):
    """
    Returns a logger to stdout whose messages have the prefix prepended
    :param prefix: The prefix to use
    """
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(fmt='{}: %(message)s'.format(prefix)))
    logger = logging.getLogger('write_ctf')
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    return logger


def cbf_reader(path):
    """
    Returns a MinibatchSource for data at the given path
    :param path: Path to a CBF file
    """
    deserializer = CBFDeserializer(path, StreamDefs(
        label=StreamDef(field='label', shape=num_classes, is_sparse=True),
        pixels=StreamDef(field='pixels', shape=frame_height * frame_width * sequence_length, is_sparse=False)
    ))

    return MinibatchSource(deserializer, randomize=False, max_samples=FULL_DATA_SWEEP)


def ctf_worker(task_queue, input_folder, output_folder, prefix, cbf_queue):
    logger = get_logger(prefix)
    while True:
        prefix = task_queue.get()
        if prefix is None:
            task_queue.task_done()
            break

        input_path = os.path.join(input_folder, '{}.cbf'.format(prefix))
        output_path = os.path.join(output_folder, '{}.ctf'.format(prefix))
        reader = cbf_reader(input_path)

        logger.info('Processing {}'.format(input_path))
        with open(output_path, 'w') as ofile:
            seq_id = 0
            while True:
                mb = reader.next_minibatch(1)
                if not mb:
                    break

                frames = mb[reader.streams.pixels].asarray().reshape((sequence_length, frame_height, frame_width))
                label = mb[reader.streams.label].asarray().argmax()

                for i in range(len(frames)):
                    pixel_str = ' '.join(frames[i].flatten().astype(str))
                    if i == 0:
                        ofile.write('{} |label {}:1 |pixels {}\n'.format(seq_id, 64, pixel_str))
                    elif i == 1:
                        ofile.write('{} |label {}:1 |pixels {}\n'.format(seq_id, label, pixel_str))
                    elif i == 2:
                        ofile.write('{} |label {}:1 |pixels {}\n'.format(seq_id, 65, pixel_str))
                    else:
                        ofile.write('{} |pixels {}\n'.format(seq_id, pixel_str))

                # Go to next sequence
                seq_id += 1

        task_queue.task_done()
        logger.info('Created {}'.format(output_path))

        # Convert to CBF format
        cbf_queue.put(prefix)


def cbf_worker(task_queue, output_folder, prefix):
    logger = get_logger(prefix)
    while True:
        prefix = task_queue.get()
        if prefix is None:
            task_queue.task_done()
            break

        input_path = os.path.join(output_folder, '{}.ctf'.format(prefix))
        output_path = os.path.join(output_folder, '{}.cbf'.format(prefix))

        subprocess.call(
            ['python', 'ctf2bin.py', '--input', input_path, '--output', output_path, '--header', 'dataset\headers\sequential.config', '--chunk_size',
             str(32 << 20)])
        logger.info('Created {}.cbf'.format(prefix))

        # Delete the CTF file
        os.remove(input_path)
        logger.info('Deleted {}'.format(input_path))


def close_and_wait(queue, workers):
    """
    Closes the given queue and waits for all workers to terminate
    """
    # None signals process termination
    for i in range(len(workers)):
        queue.put(None)

    for worker in workers:
        worker.join()


def main(num_procs, input_folder, output_folder):
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Start the CTF and CBF workers
    ctf_workers = []
    cbf_workers = []
    ctf_queue = JoinableQueue()
    cbf_queue = JoinableQueue()

    for i in range(num_procs):
        worker = Process(target=ctf_worker, args=(ctf_queue, input_folder, output_folder, 'ctf-{}'.format(i), cbf_queue))
        worker.start()
        ctf_workers.append(worker)

        worker = Process(target=cbf_worker, args=(cbf_queue, output_folder, 'cbf-{}'.format(i)))
        worker.start()
        cbf_workers.append(worker)

    # Add work items to the CTF queue
    ctf_queue.put('train')
    ctf_queue.put('test')
    for i in range(num_subjects):
        ctf_queue.put('train{}'.format(i))
        ctf_queue.put('test{}'.format(i))

    close_and_wait(ctf_queue, ctf_workers)
    close_and_wait(cbf_queue, cbf_workers)
    print('Conversion completed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-if', '--input_folder', help='The folder containing cbf files for the stacked dataset', required=True)
    parser.add_argument('-of', '--output_folder', help='The folder to store the created cbf files for the sequential dataset', required=True)
    parser.add_argument('-np', '--num_procs', help='The number of processes to use for conversion', type=int, default=4)

    args = parser.parse_args()
    main(args.num_procs, args.input_folder, args.output_folder)
