import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

output_plot_dir = 'audio_plots'
if not os.path.exists(output_plot_dir):
    os.makedirs(output_plot_dir)

def load_audio(file_path):
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    return audio_data, sample_rate

def plot_waveform(audio_data, sample_rate, output_dir):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio_data, sr=sample_rate)
    plt.title('Waveform')
    plt.savefig(os.path.join(output_dir, 'waveform.png'))
    plt.close()

def plot_spectrogram(audio_data, sample_rate, output_dir):
    plt.figure(figsize=(10, 4))
    spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
    librosa.display.specshow(spectrogram, sr=sample_rate, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.savefig(os.path.join(output_dir, 'spectrogram.png'))
    plt.close()

def plot_mfcc(audio_data, sample_rate, output_dir):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.savefig(os.path.join(output_dir, 'mfcc.png'))
    plt.close()
    return mfccs

def plot_chroma(audio_data, sample_rate, output_dir):
    chromagram = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(chromagram, sr=sample_rate, x_axis='time', y_axis='chroma', cmap='coolwarm')
    plt.colorbar()
    plt.title('Chroma Features')
    plt.savefig(os.path.join(output_dir, 'chroma.png'))
    plt.close()
    return chromagram

def plot_spectral_contrast(audio_data, sample_rate, output_dir):
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectral_contrast, sr=sample_rate, x_axis='time')
    plt.colorbar()
    plt.title('Spectral Contrast')
    plt.savefig(os.path.join(output_dir, 'spectral_contrast.png'))
    plt.close()
    return spectral_contrast

def plot_zero_crossing_rate(audio_data, sample_rate, output_dir):
    zero_crossings = librosa.feature.zero_crossing_rate(audio_data)
    plt.figure(figsize=(10, 4))
    plt.plot(zero_crossings[0])
    plt.title('Zero-Crossing Rate')
    plt.xlabel('Frame')
    plt.ylabel('Rate')
    plt.savefig(os.path.join(output_dir, 'zero_crossing_rate.png'))
    plt.close()
    return zero_crossings

def plot_spectral_rolloff(audio_data, sample_rate, output_dir):
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)
    plt.figure(figsize=(10, 4))
    plt.semilogy(spectral_rolloff.T)
    plt.title('Spectral Roll-off')
    plt.xlabel('Frame')
    plt.ylabel('Hz')
    plt.savefig(os.path.join(output_dir, 'spectral_rolloff.png'))
    plt.close()
    return spectral_rolloff

def save_features_to_csv(features, output_file):
    df = pd.DataFrame(features)
    df.to_csv(output_file, index=False)

def main(audio_file_path):
    audio_data, sample_rate = load_audio(audio_file_path)
    
    plot_waveform(audio_data, sample_rate, output_plot_dir)
    plot_spectrogram(audio_data, sample_rate, output_plot_dir)
    mfccs = plot_mfcc(audio_data, sample_rate, output_plot_dir)
    chroma = plot_chroma(audio_data, sample_rate, output_plot_dir)
    spectral_contrast = plot_spectral_contrast(audio_data, sample_rate, output_plot_dir)
    zero_crossings = plot_zero_crossing_rate(audio_data, sample_rate, output_plot_dir)
    spectral_rolloff = plot_spectral_rolloff(audio_data, sample_rate, output_plot_dir)
    
    features = {
        'MFCC': mfccs.mean(axis=1),
        'Chroma': chroma.mean(axis=1),
        'Spectral Contrast': spectral_contrast.mean(axis=1),
        'Zero Crossing Rate': zero_crossings.mean(axis=1),
        'Spectral Rolloff': spectral_rolloff.mean(axis=1)
    }
    
    features_df = pd.DataFrame.from_dict(features, orient='index').T

    features_df.to_csv('audio_features.csv', index=False)
# Example usage
audio_file_path = 'audio_data_sample/0.wav'
main(audio_file_path)
