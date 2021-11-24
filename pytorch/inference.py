import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import librosa
import matplotlib.pyplot as plt
import torch

from utilities import create_folder, get_filename
from models import Transfer_Cnn14
from models import Cnn14_DecisionLevelMax_Transfer
from pytorch_utils import move_data_to_device
import config
from evaluate import Evaluator

import numpy as np
import argparse
import time
import logging
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import (sample_rate, classes_num, mel_bins, fmin, fmax, window_size,
                    hop_size, window, pad_mode, center, ref, amin, top_db)
from losses import get_loss_func
from pytorch_utils import move_data_to_device, do_mixup
from utilities import (create_folder, get_filename, create_logging, StatisticsContainer, Mixup)
from data_generator import GtzanDataset, TrainSampler, EvaluateSampler, collate_fn
from models import (Transfer_Cnn14, Cnn14_DecisionLevelMax_Transfer)
from evaluate import Evaluator

import pandas as pd


def audio_tagging(args):
    """Inference audio tagging result of an audio clip.
    """

    # Arugments & parameters
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    audio_path = args.audio_path
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    classes_num = config.classes_num
    labels = config.labels

    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size,
                  hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax,
                  classes_num=classes_num)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Parallel
    if 'cuda' in str(device):
        model.to(device)
        print('GPU number: {}'.format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    else:
        print('Using CPU.')

    # Load audio ##주목
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

    waveform = waveform[None, :]  # (1, audio_length)
    waveform = move_data_to_device(waveform, device)

    # Forward
    with torch.no_grad():
        model.eval()
        batch_output_dict = model(waveform, None)
        print('batch_output_dict\n', batch_output_dict)

    if model_type == 'Transfer_Cnn14':
        clipwise_output = np.exp(batch_output_dict['clipwise_output'].data.cpu().numpy()[0])
    elif model_type == 'Cnn14_DecisionLevelMax_Transfer':
        clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()[0]


    """(classes_num,)"""

    print('clipwise_output:\n', clipwise_output)

    sorted_indexes = np.argsort(clipwise_output)[::-1]

    # Print audio tagging top probabilities
    print(np.array(labels))
    max_index = clipwise_output.argmax()
    print(max_index)
    for k in range(5):
        print('{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]],
                                  clipwise_output[sorted_indexes[k]]))

    # Print embedding
    if 'embedding' in batch_output_dict.keys():
        embedding = batch_output_dict['embedding'].data.cpu().numpy()[0]
        print('embedding: {}'.format(embedding.shape))

    detection_info = [[audio_path, np.array(labels)[max_index],
                      str(clipwise_output[4]), str(clipwise_output[1]),
                      str(clipwise_output[3]), str(clipwise_output[2]),
                      str(clipwise_output[0])]]
    print(detection_info)
    # ['negative', 'growling', 'whining', 'howling', 'barking'

    column_list = ["file", "prediction", "class_negative", "class_growling", "class_whining", "class_howling",
                   "class_barking"]
    barking_df = pd.DataFrame(detection_info, columns=column_list)
    barking_df.to_csv(audio_path[:-4]+'.csv')
    print(audio_path[:-4]+'.csv')

    return clipwise_output, labels


def sound_event_detection(args):
    """Inference sound event detection result of an audio clip.
    """

    # Arugments & parameters
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    audio_path = args.audio_path
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    classes_num = config.classes_num
    labels = config.labels
    frames_per_second = sample_rate // hop_size
    print('frames_per_second', frames_per_second)

    # Paths
    fig_path = os.path.join('results', '{}.png'.format(get_filename(audio_path)))
    create_folder(os.path.dirname(fig_path))

    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size,
                  hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax,
                  classes_num=classes_num, )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if 'cuda' in str(device):
        model.to(device)

    # Load audio
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

    waveform = waveform[None, :]  # (1, audio_length)
    waveform = move_data_to_device(waveform, device)

    # Forward
    with torch.no_grad():
        model.eval()
        batch_output_dict = model(waveform, None)

    framewise_output = batch_output_dict['framewise_output'].data.cpu().numpy()[0]
    """(time_steps, classes_num)"""
    print('framewise_output:', framewise_output[0].argmax())

    print('Sound event detection result (time_steps x classes_num): {}'.format(
        framewise_output.shape))

    sorted_indexes = np.argsort(np.max(framewise_output, axis=0))[::-1]

    top_k = 5  # Show top results classnum보다 작아야함
    top_result_mat = framewise_output[:, sorted_indexes[0: top_k]]
    top_1_result_mat = top_result_mat.copy()

    for idx, frame in enumerate(top_1_result_mat):
        max_index = frame.argmax()
        frame[:] = [0]*len(frame)
        frame[max_index] = 1

    print('top result mat', top_result_mat)
    print('frame number', len(top_result_mat))
    for idx, frame in enumerate(top_result_mat):
        if idx % (frames_per_second // 2) == 0:
            if frame[frame.argmax()] > 0.03: # threshold
                print('frame_label', idx / frames_per_second,
                      np.array(labels)[sorted_indexes[0: top_k]][frame.argmax()])
            else:
                print('frame_label', 'None')
    print(np.array(labels)[sorted_indexes[0: top_k]].shape)
    """(time_steps, top_k)"""

    # Plot result    
    stft = librosa.core.stft(y=waveform[0].data.cpu().numpy(), n_fft=window_size,
                             hop_length=hop_size, window='hann', center=True)
    frames_num = stft.shape[-1]

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 5))
    axs[0].matshow(np.log(np.abs(stft)), origin='lower', aspect='auto', cmap='jet')
    axs[0].set_ylabel('Frequency bins')
    axs[0].set_title('Log spectrogram')
    axs[1].matshow(top_result_mat.T, origin='upper', aspect='auto', cmap='jet', vmin=0, vmax=1)
    axs[1].xaxis.set_ticks(np.arange(0, frames_num, frames_per_second))
    axs[1].xaxis.set_ticklabels(np.arange(0, frames_num / frames_per_second))
    axs[1].yaxis.set_ticks(np.arange(0, top_k))
    axs[1].yaxis.set_ticklabels(np.array(labels)[sorted_indexes[0: top_k]])
    axs[1].yaxis.grid(color='k', linestyle='solid', linewidth=0.3, alpha=0.3)
    axs[1].set_xlabel('Seconds')
    axs[1].xaxis.set_ticks_position('bottom')
    axs[2].matshow(top_1_result_mat.T, origin='upper', aspect='auto', cmap='jet', vmin=0, vmax=1)
    axs[2].yaxis.set_ticks(np.arange(0, top_k))
    axs[2].yaxis.set_ticklabels(np.array(labels)[sorted_indexes[0: top_k]])

    plt.tight_layout()
    plt.savefig(fig_path)
    print('Save sound event detection visualization to {}'.format(fig_path))

    return framewise_output, labels


if __name__ == '__main__':
    import time

    start = time.time()

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_at = subparsers.add_parser('audio_tagging')
    parser_at.add_argument('--sample_rate', type=int, default=32000)
    parser_at.add_argument('--window_size', type=int, default=1024)
    parser_at.add_argument('--hop_size', type=int, default=320)
    parser_at.add_argument('--mel_bins', type=int, default=64)
    parser_at.add_argument('--fmin', type=int, default=50)
    parser_at.add_argument('--fmax', type=int, default=14000)
    parser_at.add_argument('--model_type', type=str, required=True)
    parser_at.add_argument('--checkpoint_path', type=str, required=True)
    parser_at.add_argument('--audio_path', type=str, required=True)
    parser_at.add_argument('--cuda', action='store_true', default=False)

    parser_sed = subparsers.add_parser('sound_event_detection')
    parser_sed.add_argument('--sample_rate', type=int, default=32000)
    parser_sed.add_argument('--window_size', type=int, default=1024)
    parser_sed.add_argument('--hop_size', type=int, default=320)
    parser_sed.add_argument('--mel_bins', type=int, default=64)
    parser_sed.add_argument('--fmin', type=int, default=50)
    parser_sed.add_argument('--fmax', type=int, default=14000)
    parser_sed.add_argument('--model_type', type=str, required=True)
    parser_sed.add_argument('--checkpoint_path', type=str, required=True)
    parser_sed.add_argument('--audio_path', type=str, required=True)
    parser_sed.add_argument('--cuda', action='store_true', default=False)

    args = parser.parse_args()

    if args.mode == 'audio_tagging':
        audio_tagging(args)

    elif args.mode == 'sound_event_detection':
        sound_event_detection(args)

    else:
        raise Exception('Error argument!')

    print("time :", time.time() - start)
