# Import the FaceDB library
from facedb import FaceDB

# Create a FaceDB instance
db = FaceDB(
    path="facedata")

import os
import glob

# Specify the folder path
folder_path = "/Users/rahmad/work/others/facerecs/imglist/"

def add_all_images(folder_path):
    # Get the list of all image files in the specified folder
    image_files = glob.glob(os.path.join(folder_path, '*.*'))

    for image_file in image_files:
        # Extract the image name without the file extension
        image_name = os.path.splitext(os.path.basename(image_file))[0]
        print(image_name)

        # Add the image to the database
        face_id = db.add(name=image_name, img=image_file, check_similar=False)


def add_image():
    # Add a new face to the database
    # add all image in imglist folder
    face_id = db.add(name="trump", img="imglist/trump.jpeg", check_similar=False)

def search_image(name):
    # Recognize a face
    result = db.recognize(img="imglist/bob_7.png", include="name")

    print(result["name"])

    # # Check if the recognized face is similar to the one in the database
    # if result and result["name"] == name:
    #     print("Recognized as :", name)
    # else:
    #     print("Unknown face")

search_image("biden")

# Add all images from the folder to the database
# add_all_images(folder_path)