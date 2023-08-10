# STAGE 2 - Audio Style Transfer and Enhancement

This folder contains a collection of scripts designed to process audio signals and convert them into frequency domain representations using Discrete Cosine Transform (DCT). Subsequently, the scripts apply a style transfer algorithm to transform this frequency image into a representation resembling a famous painting. The following steps involve converting the resultant image back into audio through IDCT. The final script then focuses on enhancing the reconstructed audio by applying bandpass filters.

## Execution Order

1. Run the script `1_Preprocessing_audio_to_image`.
2. Upon completion, execute the script `2_Processing_style_transfer`.
3. With the output of the previous step, run the script `3_Processing_image_to_audio`.
4. Finally, execute the script `4_Postprocessing_bandpass_filter`.

## Script Functions

### 1_Preprocessing_audio_to_image:

- Reads "NCS.wav" from the INPUT folder.
- Computes DCT to create a frequency domain representation.
- Generates supplementary data for proper reconstruction.
- Stores processed data in OUTPUT_1.

### 2_Processing_style_transfer:

- Utilizes the PNG image stored in OUTPUT_1.
- Applies a style transfer algorithm to resemble the painting "van_gogh.png" from the INPUT folder.
- Saves the resulting RGB image in OUTPUT_2.

### 3_Processing_image_to_audio:

- Takes the contents of OUTPUT_2.
- Reconstructs various audio signals in WAV format, attempting filter-based improvements on the OUTPUT_2 image.
- Applies IDCT to convert processed images back to audio.
- Stores the results in OUTPUT_3.

### 4_Postprocessing_bandpass_filter:

- Uses the reconstructed audio from OUTPUT_3.
- Applies bandpass filters in an effort to enhance the stored audio signals.
- Saves the refined audio outputs in OUTPUT_4.

## Notes

- These scripts collectively aim to transform audio into the frequency domain, apply style transfer, reconstruct audio, and enhance its quality through bandpass filtering. Follow the specified execution order for accurate results.

- In case you want to work with the ".py" scripts, first make sure you have the necessary permissions to execute them.
