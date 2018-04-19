import cv2
import os
import random
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np

# random example images
labels = []
images = []
root = 'poker/poker'
for i in os.listdir(root):
    name = i.split('.')
    print(i)
    labels.append(name[0])
    path = os.path.join(root,i)
    images.append(cv2.imread(path))
    dir_path = 'Aug/'+name[0]
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.
seq = iaa.Sequential(
    [
        sometimes([
            iaa.CropAndPad(
            percent=(0.2, 0.5),
            pad_mode='constant',
            pad_cval=0,
            ),
            iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
            rotate=(-45, 45), # rotate by -45 to +45 degrees
            shear=(-16, 16), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=0, # if mode is constant, use a cval between 0 and 255
            mode='constant' # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            ),
        ]),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong

        iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                ]),

        iaa.SomeOf((0, 3),
            [
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                # search either for all edges or for directed edges,
                # blend the result with the original image using a blobby mask
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast

            ],
            random_order=True
        )
    ],
    random_order=True
)

for iters in range(0,2000):
    ia.seed(random.randint(0,65535))
    images_aug = seq.augment_images(images)
    for i in range(0,len(images_aug)):
        img_path = 'Aug/'+labels[i]+'/'+str(iters)+'.png'
        print(img_path)
        cv2.imwrite(img_path,images_aug[i])