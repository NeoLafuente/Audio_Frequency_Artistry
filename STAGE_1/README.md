# STAGE 1 - Audio Processing and Frequency Domain Transformation

This repository contains a set of scripts designed to process audio signals and transform them into frequency domain representations using Discrete Cosine Transform (DCT). The scripts then perform an inverse transformation (IDCT) to reconstruct the audio signal, followed by applying bandpass filters to enhance the reconstructed audio.

## Execution Order

1. Run the script `1_Preprocessing_audio_to_image`.
2. After successful completion, execute the script `2_Processing_image_to_audio`.
3. Finally, execute the script `3_Postprocessing_bandpass_filter`.

## Script Functions

### 1_Preprocessing_audio_to_image:

- Reads "NCS.wav" from the INPUT folder.
- Computes DCT to create a frequency domain representation.
- Generates supplementary data for proper reconstruction.
- Stores processed data in OUTPUT_1.

### 2_Processing_image_to_audio:

- Takes data from OUTPUT_1.
- Performs IDCT to reconstruct audio.
- Saves reconstructed audio in OUTPUT_2.

### 3_Postprocessing_bandpass_filter:

- Utilizes reconstructed audio from OUTPUT_2.
- Applies bandpass filters to try to enhance audio.
- Saves improved audio in OUTPUT_3.

## Notes

- These scripts collectively aim to transform audio into the frequency domain, reconstruct it, and then enhance the audio quality through bandpass filtering. Make sure to adhere to the specified execution order for accurate results.

- In case you want to work with the ".py" scripts, first make sure you have the necessary permissions to execute them.
