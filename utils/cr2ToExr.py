import rawpy
import imageio
import os

def convert_cr2_to_exr(cr2_file, exr_file):
    # Read the raw CR2 file
    with rawpy.imread(cr2_file) as raw:
        # Extract the raw image data
        rgb = raw.postprocess()
    
    # Save as OpenEXR format
    imageio.imsave(exr_file, rgb)

def batch_convert_cr2_to_exr(cr2_dir, exr_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(exr_dir):
        os.makedirs(exr_dir)
    
    # Loop through all .cr2 files in the directory
    for filename in os.listdir(cr2_dir):
        if filename.endswith('.cr2'):
            cr2_file = os.path.join(cr2_dir, filename)
            exr_file = os.path.join(exr_dir, os.path.splitext(filename)[0] + '.exr')
            convert_cr2_to_exr(cr2_file, exr_file)

# Example usage:
cr2_directory = 'path/to/cr2/files'
exr_directory = 'path/to/save/exr/files'
batch_convert_cr2_to_exr(cr2_directory, exr_directory)
