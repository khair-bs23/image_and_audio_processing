from fileinput import filename
import os
from turtle import title
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class AudioProcessing:
    def __init__(self, output_plot_dir):
        self.output_plot_dir = output_plot_dir 
        if not os.path.exists(output_plot_dir):
            os.makedirs(output_plot_dir)

    def load_audio(self, file_path):
        audio_data, sample_rate = librosa.load(file_path, sr=None)
        return audio_data, sample_rate
    
    def librosa_plot(self, plot_name, plot_type_data, sample_rate, file_name, title, output_dir,  x_axis, y_axis=None, cmap='viridis'):
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(plot_type_data, sr=sample_rate, x_axis = x_axis, y_axis=y_axis, cmap=cmap)
        plt.colorbar()
        plt.title(title)
        plt.savefig(os.path.join(output_dir, f'{file_name}_{plot_name}'))
        plt.close()

    def plot_waveform(self, audio_data, sample_rate, output_dir, file_name):
        plt.figure(figsize=(10, 4))
        librosa.display.waveshow(audio_data, sr=sample_rate)
        plt.title(f'Waveform of {file_name}')
        plt.savefig(os.path.join(output_dir, f'{file_name}_waveform.png'))
        plt.close()

    def extract_spectrogram(self, audio_data, sample_rate, output_dir, file_name):
        spectrogram_data = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
        self.librosa_plot('spectogram', spectrogram_data, sample_rate, file_name, f'Spectrogram of {file_name}', output_dir, x_axis = 'time', y_axis='log')
 

    def extract_mfcc(self, audio_data, sample_rate, output_dir, file_name):
        mfcc_data = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=10)
        self.librosa_plot('mfcc', mfcc_data, sample_rate, file_name, f'MFCC of {file_name}', output_dir, x_axis = 'time')
        return mfcc_data

    def extract_chroma(self, audio_data, sample_rate, output_dir, file_name):
        chromagram_data = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
        self.librosa_plot('chromagram', chromagram_data, sample_rate, file_name, f'Chroma Features of {file_name}', output_dir, x_axis='time', y_axis='chroma', cmap='coolwarm')
        return chromagram_data

    def extract_spectral_contrast(self, audio_data, sample_rate, output_dir, file_name):
        spectral_contrast_data = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate)
        self.librosa_plot('spectral_contrast', spectral_contrast_data, sample_rate, file_name, f'Spectral Contrast of {file_name}', output_dir, x_axis='time')
        return spectral_contrast_data

    def extract_zero_crossing_rate(self, audio_data, sample_rate, output_dir, file_name):
        zero_crossings = librosa.feature.zero_crossing_rate(audio_data)
        plt.figure(figsize=(10, 4))
        plt.plot(zero_crossings[0])
        plt.title(f'Zero-Crossing Rate of {file_name}')
        plt.xlabel('Frame')
        plt.ylabel('Rate')
        plt.savefig(os.path.join(output_dir, f'{file_name}_zero_crossing_rate.png'))
        plt.close()
        return zero_crossings

    def extract_spectral_rolloff(self, audio_data, sample_rate, output_dir, file_name):
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)
        plt.figure(figsize=(10, 4))
        plt.semilogy(spectral_rolloff.T)
        plt.title(f'Spectral Roll-off of {file_name}')
        plt.xlabel('Frame')
        plt.ylabel('Hz')
        plt.savefig(os.path.join(output_dir, f'{file_name}_spectral_rolloff.png'))
        plt.close()
        return spectral_rolloff

    def feature_extraction(self, audio_file_path):
        file_name = os.path.basename(audio_file_path).split('.')[0]
        audio_data, sample_rate = self.load_audio(audio_file_path)

        file_output_dir = os.path.join(self.output_plot_dir, file_name)
        if not os.path.exists(file_output_dir):
            os.makedirs(file_output_dir)

        self.plot_waveform(audio_data, sample_rate, file_output_dir, file_name)
        self.extract_spectrogram(audio_data, sample_rate, file_output_dir, file_name)
        mfccs = self.extract_mfcc(audio_data, sample_rate, file_output_dir, file_name)
        chroma = self.extract_chroma(audio_data, sample_rate, file_output_dir, file_name)
        spectral_contrast = self.extract_spectral_contrast(audio_data, sample_rate, file_output_dir, file_name)
        zero_crossings = self.extract_zero_crossing_rate(audio_data, sample_rate, file_output_dir, file_name)
        spectral_rolloff = self.extract_spectral_rolloff(audio_data, sample_rate, file_output_dir, file_name)

        features = {
            'file_name': file_name
        }
        
        for i in range(mfccs.shape[0]):
            features[f'MFCC_{i+1:02}'] = mfccs[i].mean()
        
        for i in range(chroma.shape[0]):
            features[f'Chroma_{i+1:02}'] = chroma[i].mean()
        
        for i in range(spectral_contrast.shape[0]):
            features[f'Spectral_Contrast_{i+1:02}'] = spectral_contrast[i].mean()
        
        features['Zero_Crossing_Rate'] = zero_crossings.mean()
        
        features['Spectral_Rolloff'] = spectral_rolloff.mean()
        
        return features

    def process_audio_files(self, audio_files_dir):
        all_features = []
        
        for file_name in os.listdir(audio_files_dir):
            if file_name.endswith('.wav'):
                audio_file_path = os.path.join(audio_files_dir, file_name)
                features = self.feature_extraction(audio_file_path)
                all_features.append(features)
        
        return all_features

    def save_features_to_csv(self, all_features, output_file):
        df = pd.DataFrame(all_features)
        df.to_csv(output_file, index=False)


if __name__=='__main__':
    audio_files_dir = 'audio_data_sample'
    audio_processor = AudioProcessing('audio_plots')
    all_features = audio_processor.process_audio_files(audio_files_dir)
    audio_processor.save_features_to_csv(all_features, 'audio_features.csv')

