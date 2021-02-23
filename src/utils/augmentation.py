
import numpy as np
import imgaug as ia
from PIL import Image
from imgaug import augmenters as iaa


IMAGE_AUG_CONFIGS = {'iaa.Sometimes': 0.5,                                    # Augment only p percent of all images with one or more augmenters.
                     'iaa.SomeOf': (0, 5),                                    # List augmenter that applies only some of its children to inputs.
                     'iaa.Crop': {'percent': (0, 0.05)},                      # Crop images, i.e. remove columns/rows of pixels at the sides of images.
                     'iaa.Affine': {'scale': {"x": (0.995, 1.01),             # Apply PIL-like affine transformations to images.
                                              "y": (0.995, 1.01)},
                                    'translate_percent': {"x": (-0.3, 0.3),
                                                          "y": (-0.3, 0.3)},
                                    'rotate': (-45, 45),
                                    'shear': (-45, 45),
                                    },

                     'iaa.contrast.LinearContrast': {'alpha': (0.5, 1.5),        # Change the contrast of images.
                                                     'per_channel': 0.5},

                     'iaa.Grayscale': (0.0, 1.0),      # Augmenter to convert images to their grayscale versions.
                     'iaa.GaussianBlur': (1, 1.2),     # Augmenter to blur images using gaussian kernels.
                     'iaa.AverageBlur': (1, 3),        # Blur an image by computing simple means over neighbourhoods.
                     'iaa.MedianBlur': (1, 3),         # Blur an image by computing median values over neighbourhoods.
                     'iaa.EdgeDetect': (0, 0.7),             # Augmenter that detects all edges in images, marks them in a black
                                                       # and white image and then overlays the result with the original image.
                     'iaa.DirectedEdgeDetect': {'alpha': (0, 0.7),
                                                'direction': (0.0, 1.0),
                                                },     # Augmenter that detects edges that have certain directions and
                                                       # marks them in a black and white image and then overlays the
                                                       # result with the original image.
                     'iaa.Sharpen': {'alpha': (0, 0.7),
                                     'lightness': (0.75, 1.5),
                                     },                        # Augmenter that sharpens images and overlays the result with the original image.

                     'iaa.AdditiveGaussianNoise': {'loc': 0,
                                                   'scale': (0.0, 0.005*255),
                                                   'per_channel': 0.001
                                                   },                            # Add noise sampled from gaussian distributions elementwise to images.

                     'iaa.Dropout': {'p': (0.001, 0.01), 'per_channel': 0.5},    # Augmenter that sets a certain fraction of pixels in images to zero.
                     'iaa.CoarseDropout': {'p': 0.1, 'size_percent': 0.5},       # Augmenter that sets rectangular areas within images to zero.
                     'iaa.Add': {'value': (-10, 10), 'per_channel': 0.5},        # Add a value to all pixels in an image.
                     'iaa.Multiply':  {'mul': (0.5, 1.5), 'per_channel': 0.5},   # Multiply all pixels in an image with a specific value, thereby
                                                                                 # making the image darker or brighter.
                     'iaa.ElasticTransformation': {'alpha': (0.1, 0.2),
                                                   'sigma': 0.05},               # Transform images by moving pixels locally around using displacement fields.
                     'iaa.PiecewiseAffine': (0.001, 0.005),                      # Apply affine transformations that differ between local neighbourhoods.
                     'iaa.AddToHueAndSaturation': {'value': (-10, 10),
                                                   'per_channel': True},         # Increases or decreases hue and saturation by random values.
                     'iaa.Fliplr': 0.5                                           # Flip/mirror input images horizontally.
                     }


class ImgAugTransform:

    def __init__(self, with_aug, seed=42):
        super(ImgAugTransform, self).__init__()
        self.with_aug = with_aug
        ia.seed(seed)

    def aug(self, img):
        sometimes = lambda aug: iaa.Sometimes(IMAGE_AUG_CONFIGS['iaa.Sometimes'], aug)
        seq = iaa.Sequential(
            [
                 sometimes(iaa.Crop(percent=IMAGE_AUG_CONFIGS['iaa.Crop']['percent'])),
                 iaa.Affine(
                     scale=IMAGE_AUG_CONFIGS['iaa.Affine']['scale'],
                     translate_percent=IMAGE_AUG_CONFIGS['iaa.Affine']['translate_percent'],
                     rotate=IMAGE_AUG_CONFIGS['iaa.Affine']['rotate'],
                     shear=IMAGE_AUG_CONFIGS['iaa.Affine']['shear'],
                 ),
                  iaa.SomeOf(IMAGE_AUG_CONFIGS['iaa.SomeOf'],
                  [
                          sometimes(iaa.OneOf([
                                  iaa.OneOf([
                                      iaa.GaussianBlur(sigma=IMAGE_AUG_CONFIGS['iaa.GaussianBlur']),
                                      iaa.AverageBlur(k=IMAGE_AUG_CONFIGS['iaa.AverageBlur']),
                                      iaa.MedianBlur(k=IMAGE_AUG_CONFIGS['iaa.MedianBlur'])
                                  ]),
                                  iaa.contrast.LinearContrast(**IMAGE_AUG_CONFIGS['iaa.contrast.LinearContrast']),
                                  iaa.Grayscale(alpha=IMAGE_AUG_CONFIGS['iaa.Grayscale']),
                                  iaa.OneOf([
                                      iaa.EdgeDetect(alpha=IMAGE_AUG_CONFIGS['iaa.EdgeDetect']),
                                      iaa.DirectedEdgeDetect(
                                          alpha=IMAGE_AUG_CONFIGS['iaa.DirectedEdgeDetect']['alpha'],
                                          direction=IMAGE_AUG_CONFIGS['iaa.DirectedEdgeDetect']['direction'],
                                      ),
                                  ]),
                          ])),
                          sometimes(iaa.Sharpen(alpha=IMAGE_AUG_CONFIGS['iaa.Sharpen']['alpha'],
                                                 lightness=IMAGE_AUG_CONFIGS['iaa.Sharpen']['lightness'])),
                          sometimes(iaa.AdditiveGaussianNoise(loc=IMAGE_AUG_CONFIGS['iaa.AdditiveGaussianNoise']['loc'],
                                                              scale=IMAGE_AUG_CONFIGS['iaa.AdditiveGaussianNoise']['scale'],
                                                              per_channel=IMAGE_AUG_CONFIGS['iaa.AdditiveGaussianNoise']['per_channel'])
                                    ),
                          sometimes(iaa.Dropout(p=IMAGE_AUG_CONFIGS['iaa.Dropout']['p'],
                                                per_channel=IMAGE_AUG_CONFIGS['iaa.Dropout']['per_channel'])),
                          sometimes(iaa.CoarseDropout(p=IMAGE_AUG_CONFIGS['iaa.CoarseDropout']['p'],
                                                      size_percent=IMAGE_AUG_CONFIGS['iaa.CoarseDropout']['size_percent'])),
                          sometimes(iaa.Add(value=IMAGE_AUG_CONFIGS['iaa.Add']['value'],
                                            per_channel=IMAGE_AUG_CONFIGS['iaa.Add']['per_channel'])),
                          sometimes(iaa.Multiply(mul=IMAGE_AUG_CONFIGS['iaa.Multiply']['mul'],
                                                 per_channel=IMAGE_AUG_CONFIGS['iaa.Multiply']['per_channel'])),
                          sometimes(iaa.ElasticTransformation(alpha=IMAGE_AUG_CONFIGS['iaa.ElasticTransformation']['alpha'],
                                                              sigma=IMAGE_AUG_CONFIGS['iaa.ElasticTransformation']['sigma'])
                                    ),
                          sometimes(iaa.PiecewiseAffine(scale=IMAGE_AUG_CONFIGS['iaa.PiecewiseAffine'])),
                          sometimes(iaa.AddToHueAndSaturation(value=IMAGE_AUG_CONFIGS['iaa.AddToHueAndSaturation']['value'],
                                                              per_channel=IMAGE_AUG_CONFIGS['iaa.AddToHueAndSaturation']['per_channel'])),
                          sometimes(iaa.Fliplr(p=IMAGE_AUG_CONFIGS['iaa.Fliplr']))

                      ],
                      random_order=True
                  )
             ],
            random_order=True
        )
        return seq.augment_image(img)

    def __call__(self, img):
        img = np.asarray(img)
        if self.with_aug:
            img = self.aug(img)
        img = Image.fromarray(img)
        return img
