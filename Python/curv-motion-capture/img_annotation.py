# LIBRARY IMPORT

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab

# SETTINGS

#   Main Directory and Data type
DATADIR = "C:/Users/bdour/Documents/Data/Curv/Samples"
trial_type = 'running-side-view'

#   List of subjects numbers
subject_list = ['1', '2', '4', '5', '7', '9', '10']

#   List of landmarks to annotate

landmarks = ['Nose','Left eye', 'Right eye', 'Left ear','Right ear','Left shoulder','Right shoulder','Left elbow','Right elbow','Left wrist','Right wrist','Left hip', 'Right hip', 'Left knee','Right knee','Left ankle', 'Right ankle', 'Left heel', 'Right heel', 'Left big toe','Right big toe','Left pinky toe','Right pinky toe']

#   Figure settings
ft_size = 12

# ANNOTATION LOOPS

for subject in subject_list:
    # Define path for subject folder
    path = os.path.join(DATADIR, trial_type + '\\subject_' + subject)
    # List items in subject folder
    img = os.listdir(path)
    # Initialize array
    annotations = np.zeros([len(landmarks), 0])
    for i in range(0, len(img)):
        # Read image
        im = plt.imread(os.path.join(path, img[i]),'jpg')
        # Generate figure
        fig = pylab.figure(1)
        fig.canvas.set_window_title('(' + img[i] + '): Image manual annotation')
        fig.suptitle('Please carefully annotate the ' + str(len(landmarks)) + ' landmarks (in the specified order)\n\nNote: RIGHT CLICK to cancel last selection', fontsize=ft_size)
        ax = fig.add_subplot(111)
        ax.imshow(im)
        ax.axis('image')
        ax.axis('on')
        # plot the figure maximizes
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')                  # for 'TkAgg' backend
        # Function to allow landmark selection
        x = fig.ginput(len(landmarks), timeout=60)
        # Save landmarks coordinates in a numpy array
        annotations = np.concatenate((annotations, x), axis=1)
    plt.close(fig)

    # DATA EXPORT

    # Definie data frame structure
    f_data = pd.MultiIndex.from_product([img, ['X', 'Y']])
    # Generate data frame
    df_data = pd.DataFrame(annotations, columns=f_data)
    # Define export path
    export_path = path + '.csv'
    df_data.to_csv(export_path, index=False)
