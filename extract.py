import os
import sys
import math
import imageio
import logging
import argparse
import numpy as np
from multiprocessing import Process, JoinableQueue

label_count = 64
user_count = 10
sample_count = 5


def get_logger():
    handler = logging.StreamHandler(stream=sys.stdout)
    logger = logging.getLogger('write_ctf')
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    return logger


def get_indices(frame_count, length):
    """
    Returns a list of length 'reasonably' distributed indices
    :param frame_count: The number of frames in a video
    :param length: The number of frames to extract
    :return: A list of indices
    """
    assert frame_count >= length, 'frame_count ({}) must be >= length ({})'.format(frame_count, length)

    width = frame_count / length
    start = np.random.randint(0, width, 1)[0]

    width = math.floor(width)

    indices = [start]
    for i in range(1, length):
        indices.append(indices[-1] + width)

    return indices


def read_and_write_frames(label, user, sample, sequence_length, input_folder, output_folder):
    """
    Reads frames for the given video and write to disk
    """
    path = os.path.join(input_folder, '{:0>3}_{:0>3}_{:0>3}.mp4'.format(label + 1, user + 1, sample + 1))

    frames = []
    reader = imageio.get_reader(path)
    length = reader.get_length()

    if length < sequence_length:
        indices = range(length)
    else:
        indices = get_indices(length, sequence_length)

    for i in indices:
        frames.append(reader.get_data(i))
    reader.close()

    # Pad with copies of the last frame if length < sequence_length
    padding = sequence_length - length
    if padding > 0:
        last = frames[-1]
        for i in range(padding):
            frames.append(np.copy(last))

    # Write frames to output folder
    for i in range(sequence_length):
        path = os.path.join(output_folder, '{}_{}_{}_{}.png'.format(label, user, sample, i))
        imageio.imwrite(path, frames[i])


def worker_method(queue, sequence_length, input_folder, output_folder):
    logger = get_logger()
    while True:
        item = queue.get()
        if item is None:
            queue.task_done()
            break

        read_and_write_frames(item[0], item[1], item[2], sequence_length, input_folder, output_folder)
        logger.info('Processed {:0>3}_{:0>3}_{:0>3}.mp4'.format(item[0] + 1, item[1] + 1, item[2] + 1))
        queue.task_done()


def main(sequence_length, num_procs, input_folder, output_folder):
    # Create output directory if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    queue = JoinableQueue()
    for label in range(label_count):
        for user in range(user_count):
            for sample in range(sample_count):
                queue.put((label, user, sample))

    # None will signal process stop
    for i in range(num_procs):
        queue.put(None)

    workers = []
    for i in range(num_procs):
        worker = Process(target=worker_method, args=(queue, sequence_length, input_folder, output_folder))
        workers.append(worker)
        worker.start()

    queue.join()
    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-np', '--num_procs', type=int, help='The number of worker processes to use', default=4)
    parser.add_argument('-if', '--input_folder', type=str, help='Input folder containing video files', required=True)
    parser.add_argument('-of', '--output_folder', type=str, help='Folder to store all outputs', required=True)
    parser.add_argument('-s', '--sequence_length', type=int, help='The number of frames to extract per video')

    args = parser.parse_args()
    main(args.sequence_length, args.num_procs, args.input_folder, args.output_folder)
