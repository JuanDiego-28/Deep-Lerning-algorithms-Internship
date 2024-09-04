
This repository contains a set of Python scripts designed for image processing and deep learning tasks, specifically focused on .svs images.
Scripts:

	# 1 #div.py : Cuts a small area from the .svs image it receives and stores it as png or jpeg  

	# 2 #model1.py: contains the differnt NN that Dmitri developed 

	# 3 #tiles_gen.py: This script will crop into smaller tiles the input image. Then it will filter the data from the background 

	# 4 #train_test_val.py: Separate into validation, test and train sets the tiles you get from tiles_gen.py 

	# 5 #C_MODEL.py: import one of the UNET from model1.py and train it using the sets you get previously 

Usage:

    Install Dependencies: Ensure you have the necessary libraries installed (e.g., OpenCV, TensorFlow, Keras).

    Prepare Input Data: Have your .svs images ready in a designated directory.

    Run Scripts: Execute the scripts in the following order:
    	div.py: get small area just if need  
        tiles_gen.py: Generate tiles from your input images.
        train_test_val.py: Separate the generated tiles into training, validation, and testing sets.
        C_MODEL.py: Train a U-Net model using the prepared datasets.


