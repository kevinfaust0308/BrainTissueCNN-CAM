from keras.layers.core import Lambda
from keras.models import Sequential
import keras.backend as K
import tensorflow as tf
import numpy as np
import cv2

from pretrained_model_custom_image_preprocessing import load_image_inception, load_image


def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))


def target_category_loss_output_shape(input_shape):
    return input_shape


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


#####################################################################################

def grad_cam(input_model, model_kind, nb_classes, image_path, layer_name, category_index=-1):
    # preprocess input depending on model
    if model_kind == 'inception':
        image = load_image_inception(image_path)
        img_shape = (299, 299)
    else:
        image = load_image(image_path)
        img_shape = (224, 224)

    # predict image
    pred = input_model.predict(image)
    # will just return cam of most confident predicted class if no class inx specified
    if category_index == -1:
        # get class index with the highest prediction
        category_index = np.argmax(pred)

    # start of pre-written code
    model = Sequential()
    model.add(input_model)

    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    model.add(Lambda(target_layer,
                     output_shape=target_category_loss_output_shape))

    loss = K.sum(model.layers[-1].output)
    
    conv_output = None
    for l in model.layers[0].layers:
        if l.name == layer_name:
            conv_output = l.output
    if conv_output is None:
        raise Exception("Invalid layer name provided")
    
    
    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.layers[0].input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.ones(output.shape[0: 2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, img_shape)
    cam = np.maximum(cam, 0)
    
    heatmap_cam = cam / np.max(cam)

    # Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255 * heatmap_cam), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    
    # overlay cam, heatmap cam
    return np.uint8(cam), heatmap_cam
