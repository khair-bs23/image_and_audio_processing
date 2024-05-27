# Image Processing

**Objective:** Load, Process and Save the Processed Images

**Requirements:**
  1. Load and Display Images
  2. Perform the following operations:
     * Convert the image to grayscale.
     * Resize the image in suitable size for using computer vision models.
     * Do processing like blurring, noise adding/denoising/augmentation.
     * Detect edges
     * Perform histogram equalization
     * Apply global and adaptive thresholding on the grayscale image.
     * Save each processed image to disk in a specified directory.
  3. Save the processed image in a directory named cv_output


**Technology Used:**
  1. Python
  2. Visual Studio Code


**Sample Input**

<img src="https://github.com/khair-bs23/image_and_audio_processing/assets/167753101/72d15e29-dbda-4e81-aac3-06ab3f9d6613" alt="Alolan Dugtrio" width="300" height="300">
<img src="https://github.com/khair-bs23/image_and_audio_processing/assets/167753101/ba77e52b-b755-46ca-a681-7cfa8e705770" alt="Alolan Dugtrio" width="300" height="300">

**Sample Processed Output**

![processed_image_1](https://github.com/khair-bs23/image_and_audio_processing/assets/167753101/89c903dc-fcd6-41c6-b355-09e8363a66b2)
![processed_image_1](https://github.com/khair-bs23/image_and_audio_processing/assets/167753101/20f296e7-49b0-43b2-a373-4d3ede23b94e) 

# Audio Processing 

**Objective:** Load audio files, extract features and create a csv of those features

**Requirements:**
  1. Load Audio files from the directory
  2. Perform the following operations and plot each features
     * Plot the waveform of the audio signal.
     * Compute and display the spectrogram.
     * Extract and visualize the following features:
         * MFCC (Mel-Frequency Cepstral Coefficients)
         * Chroma features
         * Spectral contrast
         * Zero-crossing rate
         * Spectral roll-off
     * Save all extracted features to a CSV file.

**Technology Used:**
1. Python
2. Visual Studio Code

**Sample Input** 
Audio Files in .wav or .mp4

**Sample Output**

1. Plots
   
  ![0_chromagram](https://github.com/khair-bs23/image_and_audio_processing/assets/167753101/f58e802a-3f3b-4251-8980-ececfae26c3b)
  ![0_mfcc](https://github.com/khair-bs23/image_and_audio_processing/assets/167753101/3cb42367-6a93-444f-b35b-59515bcef88d)
  ![0_spectogram](https://github.com/khair-bs23/image_and_audio_processing/assets/167753101/9b687e61-7254-4470-bdce-9a79abdd4bdd)
  ![0_spectral_contrast](https://github.com/khair-bs23/image_and_audio_processing/assets/167753101/c17a6a46-27da-4382-88dd-3cadef16a7f2)
  ![0_spectral_rolloff](https://github.com/khair-bs23/image_and_audio_processing/assets/167753101/8f1a78bf-fbcb-4429-8fc0-62ded8d315c1)
  ![0_waveform](https://github.com/khair-bs23/image_and_audio_processing/assets/167753101/e66afe72-f133-4d7d-93ad-5dfae733ddbe)
  ![0_zero_crossing_rate](https://github.com/khair-bs23/image_and_audio_processing/assets/167753101/b2d4f1c9-2aa8-4dd5-91a4-5e246dc62a28)

2. CSV
   
  ![Screenshot 2024-05-27 185608](https://github.com/khair-bs23/image_and_audio_processing/assets/167753101/754a2807-45b9-4f29-8c96-9c954b50df92)



  
