# Audio_Frequency_Artistry
Explore the synergy of audio processing and artistry in AudioFrequencyArtistry. Transform audio into visuals through stages, unlocking new dimensions of creativity. Copyright Melchor Lafuente Duque

## General description

This repository comprises an audio processing pipeline organized into three distinct stages: **STAGE_1**, **STAGE_2**, and **STAGE_3**. Each stage contains a set of scripts that transform audio signals into frequency domain representations using Discrete Cosine Transform (DCT), apply style transfer algorithms, reconstruct audio, and enhance audio quality through bandpass filtering.

## STAGE_1 - Audio to Frequency Domain and Reconstruction

The scripts within this stage focus on transforming audio signals into frequency domain representations using DCT. These representations are then utilized to reconstruct the audio signal and apply bandpass filters to enhance its quality.

### Execution Order for STAGE_1 scripts:

1. `1_Preprocessing_audio_to_image`
2. `2_Processing_image_to_audio`
3. `3_Postprocessing_bandpass_filter`

## STAGE_2 - Audio Style Transfer and Enhancement

The scripts within this stage focus on style transfer algorithms, audio reconstruction, and quality improvement. The goal is to create audio signals that resemble famous paintings through style transfer.

### Execution Order for STAGE_2 scripts:

1. `1_Preprocessing_audio_to_image`
2. `2_Processing_style_transfer`
3. `3_Processing_image_to_audio`
4. `4_Postprocessing_bandpass_filter`

## STAGE_3 - Audio Synchronization and Complex Style Transfer

In this stage, the scripts synchronize two audio files, apply style transfer with complex source and style images, and reconstruct audio with enhanced quality.

### Execution Order for STAGE_3 scripts:

1. `1_Preprocessing_modify_BPM`
2. `2_Preprocessing_audio_to_image`
3. `3_Processing_style_transfer`
4. `4_Processing_image_to_audio`
5. `5_Postprocessing_bandpass_filters`
