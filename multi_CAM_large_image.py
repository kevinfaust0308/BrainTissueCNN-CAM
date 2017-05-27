import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.backend import function
from PIL import Image

##############################################################
# Main functions (at bottom):

# 1. overlay_single_layered_cam_large_image
# 2. overlay_multi_layered_cam_large_image
##############################################################

# colormaps (https://matplotlib.org/examples/color/colormaps_reference.html)
# for multi layered cam, these are some possible colors
COLORMAPS = ['Reds', 'Blues', 'Greens', 'Purples', 'Oranges', 'Greys', 'jet']


def get_convolved_image_and_pred(model, img, conv_layer):
    '''
    Returns convolved image after the specified convolutional layer as well as the predicted image class
    '''

    img = np.expand_dims(img, axis=0)

    # variables for easy access to needed layers
    input_layer = model.layers[0]
    final_layer = model.layers[-1]

    # a function that takes in the input layer and outputs prediction of image
    get_output = function([input_layer.input], [conv_layer.output, final_layer.output])

    # run function
    conv_img, pred = get_output([img])

    return conv_img[0], pred[0]


def get_conv_layer(model, conv_name):
    '''
    Returns convolutional layer within model which has the specified convolutional layer name
    '''

    final_conv_layer = None

    # loop until we find
    for l in model.layers:
        if l.name == conv_name:
            final_conv_layer = l

    # if didnt find the layer, raise error
    if not final_conv_layer:
        raise Exception('Layer name does not exist within model')

    return final_conv_layer


def get_class_indexes(pred, class_idx, show_top_x_classes):
    '''
    Returns the indices within the prediction array which represent the classes of interest (the classes we want to generate cam's for)
    '''

    # return prediction of specified class
    return np.asarray([class_idx]) if class_idx is not None else np.argpartition(pred, -show_top_x_classes)[
                                                                 -show_top_x_classes:].flatten()


def generate_cam(class_weights, conv_img):
    '''
    Generates basic cam with dimensions the size of the feature maps given by conv_outputs
    '''

    # Vectorized code

    # multiply length/width feature map dimensions
    width_times_len = conv_img.shape[0] * conv_img.shape[1]
    # our output shape will be the same as our convolved image/feature map dimensions (w/o # of maps)
    output_shape = conv_img.shape[:2]
    # reshape into 2d
    temp = conv_img.reshape((width_times_len, conv_img.shape[2]))
    # multiply all our convolved images with its corresponding weights
    # reshape to class activation map
    return np.matmul(temp, class_weights).reshape(output_shape)


def generate_single_cam_overlay(class_weights, conv_img, colormap, image_width_height, overlay_alpha,
                                remove_white_pixels):
    '''
    Generates cam overlay over entire image
    '''

    # generate base cam
    cam = generate_cam(class_weights, conv_img)

    # resize cam to dimensions (width, height)
    cam = cv2.resize(cam, image_width_height)

    # apply colormap
    cam = apply_color_map_on_BW(cam, colormap)

    # apply transparency
    cam = apply_cam_transparency(cam, overlay_alpha, remove_white_pixels)

    # return cam
    return cam


def apply_color_map_on_BW(bw, colormap):
    '''
    Applies a RGB colormap on a BW image
    '''

    # change cam to rgb.
    cmap = plt.get_cmap(colormap)

    # applying color map makes all the pixels be between 0 and 1
    color = cmap(bw)

    # make it to ranges between 0-255
    color = (color * 255).astype(np.uint8)

    # return color-mapped cam
    return np.delete(color, 3, 2)


def apply_cam_transparency(cam, overlay_alpha, remove_white_pixels):
    '''
    Applies an overlay alpha layer to a RGB cam and optionally makes whitish pixels transparent
    '''

    # original cam width and height
    orig_height = cam.shape[0]
    orig_width = cam.shape[1]

    # make alpha and set all to the preset overlay_alpha
    alpha = np.empty(orig_height * orig_width)
    alpha.fill((1 - overlay_alpha) * 255)

    # make unimportant heatmap areas be transparent to avoid overlay color dilution (if set to true)
    # reshape for looping
    cam = cam.reshape((orig_width * orig_height, cam.shape[2]))
    if remove_white_pixels:
        alpha[np.sum(cam, axis=1) > 0.90 * 255 * 3] = 0

    # reshape back to normal and return
    cam = cam.reshape((orig_width, orig_height, cam.shape[1]))
    alpha = alpha.reshape((orig_width, orig_height))
    return np.dstack((cam, alpha))


def get_image_with_cam(class_indices, class_weights, conv_img, original_img, overlay_alpha,
                       remove_white_pixels):
    '''
    Returns original image with cam overlay's applied
    '''

    # dimensions of original image
    image_width_height = original_img.shape[1], original_img.shape[0]

    # set the class activation map to be the original image which we will build up on top of
    # change to PIL image
    original_img = Image.fromarray((original_img * 255).astype(np.uint8))

    # loop through each class index and build its cam
    # overlay each cam over the original image
    for class_idx in class_indices:
        # get the class weights of the current class index (WEIGHTSxNUM_CLASSES)
        curr_class_weights = class_weights[:, class_idx]

        # get the color map to use
        colormap = COLORMAPS[class_idx]

        # generate a cam in rgba form in same size as image
        cam = generate_single_cam_overlay(curr_class_weights, conv_img, colormap, image_width_height, overlay_alpha,
                                          remove_white_pixels)

        # change cam to PIL image
        cam = Image.fromarray(cam.astype(np.uint8))

        # add cam overlay ontop of original image
        original_img.paste(cam, (0, 0), cam)

    # change PIL image to numpy array
    original_img = np.asarray(list(original_img.getdata()))
    # reshape and return
    return original_img.reshape(image_width_height + (3,)).astype(np.uint8)


def get_final_cam_overlay_and_pred(model, image, classes, conv_layer, overlay_alpha, show_top_x_classes, class_idx):
    '''
    Returns single cam overlay (if class_idx is not None) or multi cam overlay (if show_top_x_classes is not None)
    '''

    # image
    original_img = image / 255.

    # get convolved image after the specified conv_layer and the predictions
    conv_img, pred = get_convolved_image_and_pred(model, original_img, conv_layer)

    # Get the input weights to the final layer
    class_weights = model.layers[-1].get_weights()[0]

    # dont remove low confidence pixels (our colormap has white as weak)
    remove_white_pixels = False

    if show_top_x_classes is not None:
        # since we are stacking cams, we want to remove low confidence pixels
        remove_white_pixels = True

        # determine the indices of the classes we will get a cam for
        class_indices = get_class_indexes(pred, None, show_top_x_classes)

    elif class_idx is not None:
        # determine which class we will get a cam for
        class_indices = get_class_indexes(pred, class_idx, None)

    # get single overlayed cam over image, keeping or throwing away low confidence pixels
    cam = get_image_with_cam(class_indices, class_weights, conv_img, original_img, overlay_alpha, remove_white_pixels)

    # our more easier to read prediction
    new_pred = {}
    for j in class_indices:
        # get label for this class and add as a key and add the prediction score
        new_pred[classes[j]] = pred[j]

    # return cam and our class predictions
    return cam, new_pred


def get_single_layered_cam(model, image, classes, conv_layer, class_idx, overlay_alpha=0.3):
    '''   
    Returns the class activation map of a image class

    Class activation map is an unsupervised way of doing object localization with accuracy near par with supervised methods

    Parameters
    -----------
    model : '~keras.models'
        Model to generate prediction and CAM off of

    image : ndarray
        Matrix representation 
        
    classes: list of strings
        Names of all the classes the model was trained on

    conv_layer : '~keras.layers'
        Convolutional layer which we will generate the CAM

    class_name: str
        Name of class to generate cam on

    overlay_alpha: float, optional, default: 0.5, values: [0,1]
        Transparency of the cam overlay on top of the original image. 'overlay_alpha' is ignored if 'overlay' is False

    Returns
    --------
    cam : array_like
    pred: array_like
    '''

    return get_final_cam_overlay_and_pred(model, image, classes, conv_layer, overlay_alpha, None, class_idx)


def get_multi_layered_cam(model, image, classes, conv_layer, overlay_alpha=0.3, show_top_x_classes=3):
    '''   
    Returns the image-to-predict overlayed with class activation maps

    Class activation map is an unsupervised way of doing object localization with accuracy near par with supervised methods

    Parameters
    -----------
    model : '~keras.models'
        Model to generate prediction and CAM off of

    image : ndarray
        Matrix representation 

    classes: list of strings
        Names of all the classes the model was trained on

    conv_layer : '~keras.layers'
        Convolutional layer which we will generate the CAM

    overlay_alpha: float, optional, default: 0.3, values: [0,1]
        Transparency of the cam overlay on top of the original image

    show_top_x_classes: int, optional, default: None, values: [0,5]
        Overlays cam's of x classes that have the highest probabilities

    Returns
    --------
    cam : array_like
    pred: dict
    '''

    return get_final_cam_overlay_and_pred(model, image, classes, conv_layer, overlay_alpha, show_top_x_classes, None)


def overlay_prediction_on_image(image, pred, text_color):
    '''
    Takes a CAM overlayed image and writes predictions over it inplace

    Parameters
    -----------
    image : ndarray
        Matrix representation 

    pred : dict of list
        Has label as key and list of accuracy + additional information in list format

    text_color : tuple of int
        RGB tuple

    Returns
    --------
    None
    '''

    my_list = []

    # loop through all classes and its predictions
    for label in pred:
        p = '{:.2f}'.format(pred[label] * 100)  # make pred accuracy 2 decimals
        text = '{}: {}%'.format(label, p)
        my_list.append(text)

    # starting top offset
    top_offset = 30

    # loop through all those pred labels and add to image
    for label in my_list:
        cv2.putText(
            image,  # put text on this image
            label,  # text
            (10, top_offset),  # left offset, top offset
            cv2.FONT_HERSHEY_SIMPLEX,  # font family
            0.6,  # scale of text size
            text_color,  # color
            1,  # line width
            cv2.LINE_AA  # line type
        )

        # move next text down for spacing
        top_offset += 30


def get_final_cam_overlay_and_pred_large_image(model, cnn_trained_image_size, classes, image, conv_name, overlay_alpha,
                                               show_top_x_classes, class_idx, overlay_predictions,
                                               overlay_text_color):
    # check for invalid params
    if not (0 <= overlay_alpha <= 1):
        raise Exception("Invalid overlay_alpha given")

    if (image.shape[0] < cnn_trained_image_size) or (image.shape[1] < cnn_trained_image_size):
        raise Exception("Image too small")

    if len(classes) > len(COLORMAPS):
        raise Exception("Only {} classes are currently supported".format(str(len(COLORMAPS))))

    # show user what color each class on the cam will be. currently restricted to having 7 classes (one for each colormap)
    for i, _class in enumerate(classes):
        print('{} --> {}'.format(_class, COLORMAPS[i][:-1]))

    # shape of image
    height, width, chn = image.shape

    # check whether image has dimensions of multiples of cnn_trained_image_size
    if width % cnn_trained_image_size != 0:
        # 'crop' leftover from right
        width -= width % cnn_trained_image_size
    if height % cnn_trained_image_size != 0:
        # 'crop' leftover from bottom
        height -= height % cnn_trained_image_size

    # get convolutional layer
    conv_layer = get_conv_layer(model, conv_name)

    # new image placeholder (will hold our recreated image)
    new_image = np.zeros((height, width, chn))

    # this is a slow process so add progress bar
    print('Analyzing')

    # go through all the sub-image sections in the horizontal
    for i in list(range(0, (width - cnn_trained_image_size) + 1, cnn_trained_image_size)):
        # go through all the sub-image sections in the vertical
        for j in list(range(0, (height - cnn_trained_image_size) + 1, cnn_trained_image_size)):

            # progress bar
            print('.', end='')

            # get current sub-image
            sub_img = image[j:j + cnn_trained_image_size, i:i + cnn_trained_image_size, :]

            # determine whether single or multi cam
            if show_top_x_classes is not None:
                # get stacked cam of top x classes
                overlay_cam, pred = get_multi_layered_cam(model, sub_img, classes, conv_layer,
                                                          overlay_alpha=overlay_alpha,
                                                          show_top_x_classes=show_top_x_classes)

            elif class_idx is not None:
                # single layer cam of specific class
                overlay_cam, pred = get_single_layered_cam(model, sub_img, classes, conv_layer,
                                                           overlay_alpha=overlay_alpha,
                                                           class_idx=class_idx)

            # show on subimage the class and predictions
            if overlay_predictions:
                overlay_prediction_on_image(overlay_cam, pred, overlay_text_color)
            else:
                # TODO: potentially do something else? averaging the prediction wasnt very useful
                pass

            # put sub-image into correct spot of matrix (recreating image)
            new_image[j:j + cnn_trained_image_size, i:i + cnn_trained_image_size, :] = overlay_cam

    # progress bar
    print('\nComplete')

    # return recreated image
    return new_image.astype(np.uint8)


def overlay_single_layered_cam_large_image(model, cnn_trained_image_size, classes, image, conv_name, class_name,
                                           overlay_alpha=0.5,
                                           overlay_predictions=False,
                                           overlay_text_color=(0, 0, 0)):
    ''' 
    Takes in a larger image than what the CNN model was trained on and returns a single-CAM overlay (based on the chosen class). 
    If image dimensions are not multiples of the CNN's trained image size, rightmost and/or bottommost parts of the image are ignored.
    Returns total average prediction if 'overlay_predictions' is False. Otherwise, shows individual sub-image predictions
    as overlay

    Currently supports as many classes as colormap values (7)

    Class activation map is an unsupervised way of doing object localization with accuracy near par with supervised methods

    Parameters
    -----------
    model : '~keras.models'
        Model to generate prediction and CAM off of

    cnn_trained_image_size: int
        Image size trained on the CNN

    classes: list of str

    image : ndarray
        Matrix representation. Must have dimensions that are multiples of 'cnn_trained_image_size'

    conv_name : str
        Name of the convolutional layer which we will generate the CAM

    class_idx: str
        Overlay cam of this specific class over the original image

    overlay_alpha : float, optional, default: 0.5, values: [0,1]
        Transparency of the cam overlay on top of the original image

    overlay_predictions : bool, optional, default: False
        Overlay prediction of each sub-image

    overlay_text_color : tuple of int
        RGB color of the overlay prediction text. Ignored if 'overlay_predictions' is False

    Returns
    --------
    new_image : array_like
    my_dict : dict
    '''

    if class_name and class_name not in classes:
        raise Exception("Class not found")

    # class index of interest
    class_idx = classes.index(class_name)

    return get_final_cam_overlay_and_pred_large_image(model, cnn_trained_image_size, classes, image, conv_name,
                                                      overlay_alpha=overlay_alpha,
                                                      class_idx=class_idx, show_top_x_classes=None,
                                                      overlay_predictions=overlay_predictions,
                                                      overlay_text_color=overlay_text_color)


def overlay_multi_layered_cam_large_image(model, cnn_trained_image_size, classes, image, conv_name, overlay_alpha=0.5,
                                          show_top_x_classes=3, overlay_predictions=False,
                                          overlay_text_color=(0, 0, 0)):
    ''' 
    Takes in a larger image than what the CNN model was trained on and returns a multi-CAM overlay. If image dimensions
    are not multiples of the CNN's trained image size, rightmost and/or bottommost parts of the image are ignored.
    Returns total average prediction if 'overlay_predictions' is False. Otherwise, shows individual sub-image predictions
    as overlay
    
    Each overlay on each subimage will have at most its top 5 predicted classes shown.
    Currently supports as many classes as colormap values (7)

    Class activation map is an unsupervised way of doing object localization with accuracy near par with supervised methods

    Parameters
    -----------
    model : '~keras.models'
        Model to generate prediction and CAM off of

    cnn_trained_image_size: int
        Image size trained on the CNN

    classes: list of str

    image : ndarray
        Matrix representation. Must have dimensions that are multiples of 'cnn_trained_image_size'

    conv_name : str
        Name of the convolutional layer which we will generate the CAM

    overlay_alpha : float, optional, default: 0.5, values: [0,1]
        Transparency of the cam overlay on top of the original image
        
    show_top_x_classes: int, optional, default: None, values: [0,5]
        Overlays cam's of x classes that have the highest probabilities

    overlay_predictions : bool, optional, default: False
        Overlay prediction of each sub-image

    overlay_text_color : tuple of int
        RGB color of the overlay prediction text. Ignored if 'overlay_predictions' is False

    Returns
    --------
    new_image : array_like
    my_dict : dict
    '''

    if not (0 <= show_top_x_classes <= 5):
        raise Exception("Can only show between 0 and 5 classes")

    # check if we have less classes than the number of classes desired to show
    if len(classes) < show_top_x_classes:
        # show all the classes
        show_top_x_classes = len(classes)

    return get_final_cam_overlay_and_pred_large_image(model, cnn_trained_image_size, classes, image, conv_name,
                                                      overlay_alpha=overlay_alpha,
                                                      class_idx=None, show_top_x_classes=show_top_x_classes,
                                                      overlay_predictions=overlay_predictions,
                                                      overlay_text_color=overlay_text_color)
