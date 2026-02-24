import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split, Dataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import joblib
import matplotlib.pyplot as plt
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import imageio
import tempfile
import os

from Data import Field
from SignedDistanceFunction import SignedDistanceFunction
from NNModel import IntervalNeuralNetwork

def elevation_gif(utme, utmn, elevation, elevation_count=100, utm_count=100, path='nn', save='elevation'):
    warnings.filterwarnings('ignore')

    encoder = joblib.load(f'{path}/strat.enc')

    config = joblib.load(f'{path}/config.txt')

    model = IntervalNeuralNetwork(config['output_size'])
    state_dict = torch.load(f'{path}/sdf.pth')
    model.load_state_dict(state_dict)
    model.eval()

    utme_scaler = joblib.load(f'{path}/utme.scl')
    utmn_scaler = joblib.load(f'{path}/utmn.scl')
    elevation_scaler = joblib.load(f'{path}/elevation.scl')

    utme_size = (utme[1] - utme[0])/utm_count
    utmn_size = (utmn[1] - utmn[0])/utm_count

    elevations = np.linspace(elevation[0], elevation[1], elevation_count)

    n_classes = config['output_size']
    vmin, vmax = 0, n_classes - 1
    all_classes = np.arange(n_classes)
    all_labels = encoder.inverse_transform(all_classes)

    temp_dir = tempfile.mkdtemp()
    frame_paths = []

    for step, elevation in enumerate(elevations):
        data = np.full([utm_count, utm_count], -1)

        for idx in range(0, utm_count):

            x = utme_scaler.transform([[utme[0] + utme_size * idx]])[0][0]
            z = elevation_scaler.transform([[elevation]])[0][0]

            for jdx in range(0, utm_count):
                y = utmn_scaler.transform([[utmn[0] + utmn_size * jdx]])[0][0]

                X = torch.tensor([z, x, y]).float().unsqueeze(0)

                with torch.no_grad():
                    output = model(X)

                data[jdx, idx] = int(torch.argmax(output))

        fig, ax = plt.subplots()
        im = ax.imshow(data, cmap='jet', origin='lower', aspect='auto', extent=(utme[1], utme[0], utmn[1], utmn[0]),
                       vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(im, ax=ax)

        cbar.set_ticks(all_classes)
        cbar.set_ticklabels(all_labels)
        cbar.set_label('Formation')

        ax.set_xlabel('UTME')
        ax.set_ylabel('UTMN')
        ax.set_title(f'Elevation: {elevation:.1f}m')

        frame_path = os.path.join(temp_dir, f'frame_{step:04d}.png')
        plt.savefig(frame_path, dpi=100)
        plt.close()
        frame_paths.append(frame_path)

        print(f'Rendered frame {step + 1}/{elevation_count} at elevation {elevation:.1f}m')

    frames = [imageio.imread(p) for p in frame_paths]
    imageio.mimsave(f'{save}.gif', frames, fps=1)

    for p in frame_paths:
        os.remove(p)
    os.rmdir(temp_dir)


def elevation_cross_section(utme, utmn, elevation, utm_count=100, path='nn', save='elevation_cross'):
    warnings.filterwarnings('ignore')

    encoder = joblib.load(f'{path}/strat.enc')

    config = joblib.load(f'{path}/config.txt')

    model = IntervalNeuralNetwork(config['output_size'])
    state_dict = torch.load(f'{path}/sdf.pth')
    model.load_state_dict(state_dict)
    model.eval()

    utme_scaler = joblib.load(f'{path}/utme.scl')
    utmn_scaler = joblib.load(f'{path}/utmn.scl')
    elevation_scaler = joblib.load(f'{path}/elevation.scl')

    utme = (min(utme), max(utme))
    utmn = (min(utmn), max(utmn))

    utme_size = (utme[1] - utme[0])/utm_count
    utmn_size = (utmn[1] - utmn[0])/utm_count

    data = np.full([utm_count, utm_count], -1)

    for idx in range(0, utm_count):

        x = utme_scaler.transform([[utme[0] + utme_size * idx]])[0][0]
        z = elevation_scaler.transform([[elevation]])[0][0]

        for jdx in range(0, utm_count):

            y = utmn_scaler.transform([[utmn[0] + utmn_size * jdx]])[0][0]

            X = torch.tensor([z, x, y]).float().unsqueeze(0)

            with torch.no_grad():
                output = model(X)

            data[jdx, idx] = int(torch.argmax(output))

    fig, ax = plt.subplots()

    im = ax.imshow(data, cmap='jet', origin='lower', aspect='auto', extent=(utme[1], utme[0], utmn[1], utmn[0]))
    cbar = plt.colorbar(im, ax=ax)

    classes = np.unique(data[data >= 0])
    cbar.set_ticks(classes)

    classes = encoder.inverse_transform(classes)
    cbar.set_ticklabels(classes)
    cbar.set_label('Formation')

    ax.set_xlabel('UTME')
    ax.set_ylabel('UTMN')

    plt.savefig(f'{save}.png')
    plt.close()

def utme_cross_section(utme, utmn, elevation, utm_count=100, path='nn', save='utme_cross'):
    warnings.filterwarnings('ignore')

    encoder = joblib.load(f'{path}/strat.enc')

    config = joblib.load(f'{path}/config.txt')

    model = IntervalNeuralNetwork(config['output_size'])
    state_dict = torch.load(f'{path}/sdf.pth')
    model.load_state_dict(state_dict)
    model.eval()

    utme_scaler = joblib.load(f'{path}/utme.scl')
    utmn_scaler = joblib.load(f'{path}/utmn.scl')
    elevation_scaler = joblib.load(f'{path}/elevation.scl')

    utme = (min(utme), max(utme))
    elevation = (min(elevation), max(elevation))

    utme_size = (utme[1] - utme[0])/utm_count

    data = np.full([(elevation[1] - elevation[0]), utm_count], -1)

    for idx in range(0, utm_count):

        x = utme_scaler.transform([[utme[0] + utme_size * idx]])[0][0]
        y = utmn_scaler.transform([[utmn]])[0][0]

        for jdx in range(0, elevation[1] - elevation[0]):

            z = elevation_scaler.transform([[jdx + elevation[0]]])[0][0]

            X = torch.tensor([z, x, y]).float().unsqueeze(0)

            with torch.no_grad():
                output = model(X)

            data[jdx, idx] = int(torch.argmax(output))

    fig, ax = plt.subplots()

    im = ax.imshow(data, cmap='jet', origin='lower', aspect='auto', extent=(utme[0], utme[1], elevation[0], elevation[1]))
    cbar = plt.colorbar(im, ax=ax)

    classes = np.unique(data[data >= 0])
    cbar.set_ticks(classes)

    classes = encoder.inverse_transform(classes)
    cbar.set_ticklabels(classes)
    cbar.set_label('Formation')

    ax.set_xlabel('UTME')
    ax.set_ylabel('Elevation')

    plt.savefig(f'{save}.png')
    plt.close()