# STAGE 3 - Audio Synchronization and Complex Style Transfer

This directory contains a collection of scripts designed to synchronize the BPM and duration of two songs, "NCS.wav" and "SY5.wav," followed by transforming them into frequency domain image representations using Discrete Cosine Transform (DCT). The image derived from "SY5.wav" serves as the source image, while the one created from "NCS.wav" serves as the style image for applying the style transfer algorithm. The goal is to make the source image resemble the style image, then convert the processed image back into audio using IDCT. The final script focuses on enhancing the reconstructed audio by applying bandpass filters.

## Execution Order

1. Run the script `1_Preprocessing_modify_BPM`.
2. Upon successful completion, execute the script `2_Preprocessing_audio_to_image`.
3. After that, run the script `3_Processing_style_transfer`.
4. Using the outcome of the previous step, execute the script `4_Processing_image_to_audio`.
5. Finally, execute the script `5_Postprocessing_bandpass_filters`.

## Script Functions

### 1_Preprocessing_modify_BPM:

- Reads "SY5.wav" and "NCS.wav" from the INPUT folder.
- Adjusts BPM and duration to synchronize both audio files.
- Saves the adjusted audios as "ORIGINAL_PROCESED.wav" and "STYLE_PROCESED.wav" in OUTPUT_1.

### 2_Preprocessing_audio_to_image:

- Reads audio files from OUTPUT_1.
- Transforms audios into frequency domain images using DCT.
- Generates additional components for accurate reconstruction.
- Stores processed data in OUTPUT_2/ORIGIN and OUTPUT_2/STYLE.

### 3_Processing_style_transfer:

- Utilizes the PNG image stored in OUTPUT_2/ORIGIN.
- Applies style transfer algorithm to resemble the image in OUTPUT_2/STYLE.
- Saves the resulting RGB image in OUTPUT_3.

### 4_Processing_image_to_audio:

- Takes content from OUTPUT_3.
- Reconstructs various audio signals in WAV format, experimenting with filters on the OUTPUT_3 image for improvement.
- Applies IDCT to convert processed images back to audio.
- Stores results in OUTPUT_4.

### 5_Postprocessing_bandpass_filters:

- Uses the reconstructed audio from OUTPUT_4.
- Applies bandpass filters to enhance stored audio signals.
- Saves the refined audio outputs in OUTPUT_5.

## Notes

- These scripts collectively synchronize audio, transform it into the frequency domain, apply style transfer, reconstruct audio, and enhance its quality through bandpass filtering. Follow the provided execution order for accurate outcomes.

- In case you want to work with the ".py" scripts, first make sure you have the necessary permissions to execute them.
