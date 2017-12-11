import os
import sys
import cv2
import logging
import argparse
import subprocess
import numpy as np
from multiprocessing import Process, JoinableQueue

label_count = 64
user_count = 10
sample_count = 5
sequence_length = 20


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


def read_frames(label, user, sample, input_folder):
    """
    Reads the frames for the given sample
    """
    frames = []
    for i in range(sequence_length):
        path = os.path.join(input_folder, '{}_{}_{}_{}.png'.format(label, user, sample, i))
        frames.append(cv2.imread(path))

    return frames


def process_frames(frames):
    """
    Crops the frames in the given list to 400 x 440, converts them to grayscale,
    resizes them to 120 x 120 and then normalizes them. Pads with the last
    frame if less than 'sequence_length'
    :param frames: The list of frames to process
    :return: The processed frames
    """
    processed = []

    for frame in frames:
        gray = cv2.cvtColor(frame[:, 200:], cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (120, 120), interpolation=cv2.INTER_AREA)
        norm = (gray - gray.mean()) / (gray.std() + 1e-8)
        processed.append(norm)

    return np.stack(processed).astype(np.float32)


def train_test_split(user_id):
    """
    Returns a train/test. If user is none, split will pay no regard to signer.
    Otherwise, all samples belonging to user will be in the test set and all
    other samples will be in the training set.
    """
    training_set = []
    test_set = []

    if user_id is None:
        # Signer-independent dataset
        test_id = 0
        train_id = 0
        for label in range(label_count):
            for user in range(user_count):
                samples = np.arange(sample_count)
                np.random.shuffle(samples)

                test_set.append((label, user, samples[0], test_id))
                test_id += 1

                for sample in samples[1:]:
                    training_set.append((label, user, sample, train_id))
                    train_id += 1
    else:
        # Signer-dependent dataset
        seq_id = 0
        for label in range(label_count):
            for user in range(user_count):
                for sample in range(sample_count):
                    sample = label, user, sample, seq_id
                    seq_id += 1

                    if user == user_id:
                        test_set.append(sample)
                    else:
                        training_set.append(sample)

    return training_set, test_set


def train_test_queue(training_set, test_set, num_procs):
    train_queue = JoinableQueue()
    for entry in training_set:
        train_queue.put(entry)

    test_queue = JoinableQueue()
    for entry in test_set:
        test_queue.put(entry)

    # None will signal process end
    for i in range(num_procs):
        train_queue.put(None)
        test_queue.put(None)

    return train_queue, test_queue


def ctf_worker(task_queue, input_folder, output_folder, file_name, status):
    """
    Writes samples from the task queue to the CNTK CTF Format
    :param task_queue: The task queue with samples to process
    :param input_folder: The input folder where raw videos are stored
    :param output_folder: The output folder where the CTF files will be written
    :param file_name: The name of the file to write to
    :param status: One of 'Train' or 'Test' for logging.
    """
    logger = get_logger(status)
    with open(os.path.join(output_folder, file_name), 'w') as ofile:
        while True:
            sample = task_queue.get()
            if sample is None:
                task_queue.task_done()
                break

            label, user, sample, seq_id = sample
            frames = read_frames(label, user, sample, input_folder)
            frames = process_frames(frames)

            pixel_str = ' '.join(frames.flatten().astype(str))
            ofile.write('|label {}:1 |pixels {}\n'.format(label, pixel_str))

            task_queue.task_done()
            logger.info('Processed ({}, {}, {})'.format(label, user, sample))


def merge_worker(task_queue, num_procs, output_dir, cbf_queue):
    """
    Merge the CTF files with prefix, 'prefix', into a single CTF file
    :param task_queue: The queue of tasks to process
    :param num_procs: The number of processes used to write the CTF files. Sames as the number of CTF files to read from
    :param output_dir: The directory to write the output to
    :param cbf_queue: The queue for items to convert from CTF to CBF
    """

    while True:
        prefix = task_queue.get()
        if prefix is None:
            task_queue.task_done()
            break

        logger = get_logger(prefix)
        ctf_files = [os.path.join(output_dir, '{}_{}.ctf'.format(prefix, i)) for i in range(num_procs)]

        with open(os.path.join(output_dir, '{}.ctf'.format(prefix)), 'w') as output_file:
            for file in ctf_files:
                with open(file, 'r') as input_file:
                    for line in input_file:
                        output_file.write(line)
                logger.info('Processed {}'.format(file))

        # Delete the individual CTF files
        for file in ctf_files:
            logger.info('Deleting {}'.format(file))
            os.remove(file)

        logger.info('{}: Merge Complete.'.format(prefix))
        cbf_queue.put(prefix)


def cbf_worker(task_queue, output_folder):
    while True:
        prefix = task_queue.get()
        if prefix is None:
            task_queue.task_done()
            break

        logger = get_logger(prefix)
        input_path = os.path.join(output_folder, '{}.ctf'.format(prefix))
        output_path = os.path.join(output_folder, '{}.cbf'.format(prefix))

        subprocess.call(
            ['python', 'ctf2bin.py', '--input', input_path, '--output', output_path, '--header', 'dataset\headers\stack.config', '--chunk_size',
             '131072'])
        logger.info('Created {}.cbf'.format(prefix))

        # Delete the CTF file
        os.remove(input_path)
        logger.info('Deleted {}'.format(input_path))


def write_ctf(train_queue, test_queue, train_prefix, test_prefix, num_procs, input_folder, output_folder, merge_queue):
    # Write subsets of training data across multiple processes
    for i in range(num_procs):
        worker = Process(target=ctf_worker,
                         args=(train_queue, input_folder, output_folder, '{}_{}.ctf'.format(train_prefix, i), train_prefix))
        worker.start()

    train_queue.join()
    merge_queue.put(train_prefix)
    print('{}: Writing training set completed.\n'.format(train_prefix))

    # Write subsets of test data across multiple processes
    for i in range(num_procs):
        worker = Process(target=ctf_worker,
                         args=(test_queue, input_folder, output_folder, '{}_{}.ctf'.format(test_prefix, i), test_prefix))
        worker.start()

    test_queue.join()
    merge_queue.put(test_prefix)
    print('{}: Writing test set completed.\n'.format(test_prefix))


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
    # Create output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Start the processes for merging and ctf-cbf conversion
    merge_workers = []
    merge_queue = JoinableQueue()

    cbf_workers = []
    cbf_queue = JoinableQueue()

    for i in range(num_procs):
        worker = Process(target=merge_worker, args=(merge_queue, num_procs, output_folder, cbf_queue))
        worker.start()
        merge_workers.append(worker)

        worker = Process(target=cbf_worker, args=(cbf_queue, output_folder))
        worker.start()
        cbf_workers.append(worker)

    # Subject-dependent dataset
    train_set, test_set = train_test_split(user_id=None)
    train_queue, test_queue = train_test_queue(train_set, test_set, num_procs)
    write_ctf(train_queue, test_queue, 'train', 'test', num_procs, input_folder, output_folder, merge_queue)

    # Subject-independent datasets
    for i in range(user_count):
        train_set, test_set = train_test_split(user_id=i)
        train_queue, test_queue = train_test_queue(train_set, test_set, num_procs)
        write_ctf(train_queue, test_queue, 'train{}'.format(i), 'test{}'.format(i), num_procs, input_folder, output_folder, merge_queue)

    close_and_wait(merge_queue, merge_workers)
    close_and_wait(cbf_queue, cbf_workers)
    print('Creating dataset(s) completed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-np', '--num_procs', type=int, help='The number of worker processes to use', default=4)
    parser.add_argument('-if', '--input_folder', type=str, help='Input folder containing video files', required=True)
    parser.add_argument('-of', '--output_folder', type=str, help='Folder to store all outputs', required=True)

    args = parser.parse_args()
    main(args.num_procs, args.input_folder, args.output_folder)
