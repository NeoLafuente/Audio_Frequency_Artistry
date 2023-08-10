#!/usr/bin/env python
# coding: utf-8

# ##### Copyright 2023 Melchor Lafuente Duque

# # DOWNLOAD THE INPUTS

# In[1]:


import requests
import os

# Define the file URLs and their corresponding destination folders
files = {
    "NCS.wav": "https://drive.google.com/uc?id=1SuT00AKBbrg5diGNXler4dAR2ytDgbgc",
    "SY5.wav": "https://drive.google.com/uc?id=1r5nqpH7aRT1w2jCFJYpn4qKSEm3dUHOc",
    "van_gogh.jpg": "https://drive.google.com/uc?id=1GAgLXQniZIO6spngWcU4OpmVFr-X4pd1"
}

# Define the destination folders for each file
destination_folders = {
    "NCS.wav": ["STAGE_1/INPUT", "STAGE_2/INPUT", "STAGE_3/INPUT"],
    "SY5.wav": ["STAGE_3/INPUT"],
    "van_gogh.jpg": ["STAGE_2/INPUT"]
}

# Create the destination folders if they don't exist
for folders in destination_folders.values():
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

# Download and save the files
for file_name, file_url in files.items():
    response = requests.get(file_url)
    if response.status_code == 200:
        for folder in destination_folders.get(file_name, []):
            file_path = os.path.join(folder, file_name)
            with open(file_path, "wb") as file:
                file.write(response.content)
            print(f"File '{file_name}' downloaded and saved to '{folder}'")
    else:
        print(f"Error downloading '{file_name}'")

print("Download and save completed.")


# # ADITIONAL HELP

# As the files .pickle of the STAGE_2 and STAGE_3 are to heavy and complex to create, you can just download them and add them to their correspondig folders executing the next code:

# In[3]:


# Define the file URLs and their corresponding destination folders
files = {
    "final_image_sliced.pickle": [
        ("https://drive.google.com/uc?id=1txddUIlF7syH02P0RM8xYuKw45JB-K5c", "STAGE_2/OUTPUT_2"),
        ("https://drive.google.com/uc?id=1FYdT3EeTSAeRxpyd5Ea3YlgN5C0jxrqr", "STAGE_3/OUTPUT_3")
    ]
}

# Download and save the files
for file_name, destinations in files.items():
    for file_url, folder in destinations:
        response = requests.get(file_url)
        if response.status_code == 200:
            file_path = os.path.join(folder, file_name)
            with open(file_path, "wb") as file:
                file.write(response.content)
            print(f"File '{file_name}' downloaded and saved to '{folder}'")
        else:
            print(f"Error downloading '{file_name}'")

print("Download and save completed.")

