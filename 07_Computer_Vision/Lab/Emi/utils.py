import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import cv2

def get_classes_distribution(y_data, LABELS):
    # Get the count for each label
    y = np.bincount(y_data)
    ii = np.nonzero(y)[0]
    label_counts = zip(ii, y[ii])
    # Get total number of samples
    total_samples = len(y_data)
    # Count the number of items in each class
    for label, count in label_counts:
        class_name = LABELS[label]
        percent = (count / total_samples) * 100
        print("{:<15s}:  {} or {:.2f}%".format(class_name, count, percent))
        
    return label_counts


def plot_label_per_class(y_data, LABELS):
    classes = sorted(np.unique(y_data))
    f, ax = plt.subplots(1,1, figsize=(12, 4))
    g = sns.countplot(y_data, order=classes)
    g.set_title("Number of labels for each class")
    for p, label in zip(g.patches, classes):
        g.annotate(LABELS[label], (p.get_x(), p.get_height() + 0.2))
    
    plt.show()


def sample_images_data(x_data, y_data, LABELS):
    # An empty list to collect some samples
    sample_images = []
    sample_labels = []
    # Iterate over the keys of the labels dictionary defined in the above cell
    for k in LABELS.keys():
        # Get four samples for each category
        samples = np.where(y_data == k)[0][:4]
        # Append the samples to the samples list
        for s in samples:
            img = x_data[s]
            sample_images.append(img)
            sample_labels.append(y_data[s])

    print("Total number of sample images to plot: ", len(sample_images))
    return sample_images, sample_labels


def plot_sample_images(data_sample_images, data_sample_labels, LABELS, cmap="gray"):
    # Plot the sample images now
    f, ax = plt.subplots(5, 8, figsize=(16, 10))

    for i, img in enumerate(data_sample_images):
        ax[i//8, i%8].imshow(img, cmap=cmap)
        ax[i//8, i%8].axis('off')
        ax[i//8, i%8].set_title(LABELS[data_sample_labels[i]])
    plt.show()