import numpy as np
from keras.preprocessing import image

from keras.applications.vgg19 import preprocess_input


# general
def load_image(img_path):
    '''
    Takes an image path and returns pre-processed image.
    '''
    # image loading
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)
    # 3d to 4d tensor
    x = np.expand_dims(x, axis=0)
    # perform preprocessing (for ex: mean subtract, rgb to bgr, etc.)
    x = preprocess_input(x)
    return x

def load_image_inception(img_path):
    # image loading
    img = image.load_img(img_path, target_size=(299,299))
    x = image.img_to_array(img)
    # 3d to 4d tensor
    x = np.expand_dims(x, axis=0)
    # preprocessing for inception v3 input
    x /= 255.
    x -= 0.5
    x *= 2.
    return x



# pre-trained vgg model configs

# for use with ImageDataGenerator flow. we dynamically change RGB ordering to what vgg was trained on which is BGR
def override_keras_directory_iterator_next():
    """Overrides .next method of DirectoryIterator in Keras
      to reorder color channels for images from RGB to BGR"""
    from keras.preprocessing.image import DirectoryIterator

    original_next = DirectoryIterator.next

    # do not allow to override one more time
    if 'custom_next' in str(original_next):
        return

    def custom_next(self):
        batch_x, batch_y = original_next(self)

        batch_x = batch_x[:, ::-1, :, :]
        return batch_x, batch_y

    DirectoryIterator.next = custom_next

# for use with ImageDataGenerator flow. this is the mean for the vgg model. we can use featurewise_center=True now
def apply_vgg_mean(image_data_generator):
    """Subtracts the VGG dataset mean"""
    image_data_generator.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape((3, 1, 1))