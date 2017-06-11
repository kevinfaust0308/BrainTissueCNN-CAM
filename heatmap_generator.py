import numpy as np
import matplotlib.pyplot as plt
import cv2
import keras.backend as K
import time

##############################################################
# Main functions (at bottom):

# 1. overlay_single_layered_cam_large_image
# 2. overlay_multi_layered_cam_large_image
##############################################################

# colormaps (https://matplotlib.org/examples/color/colormaps_reference.html)
# for multi layered cam, these are some possible colors
COLORMAPS = ['Blues', 'Greens', 'Reds', 'Purples', 'Oranges', 'Greys']
SINGLE_CAM_OVERLAY_COLORMAP = 'YlOrRd'


def blend_transparent(face_img, overlay_t_img):
    # code from: https://stackoverflow.com/questions/36921496/how-to-join-png-with-alpha-transparency-in-a-frame-in-realtime/37198079#37198079

    # Split out the transparency mask from the colour info
    overlay_img = overlay_t_img[:, :, :3]  # Grab the RBG planes
    overlay_mask = overlay_t_img[:, :, 3:]  # And the alpha plane

    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2RGB)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2RGB)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))


def get_convolved_image_and_pred(model, img, conv_layer):
    '''
    Returns convolved image after the specified convolutional layer as well as the predicted image class
    '''

    img = np.expand_dims(img, axis=0)

    # variables for easy access to needed layers
    input_layer = model.layers[0]
    final_layer = model.layers[-1]

    # a function that takes in the input layer and outputs prediction of image
    get_output = K.function([input_layer.input], [conv_layer.output, final_layer.output])

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


def generate_cam_for_overlay(class_weights, conv_img, colormap, image_width_height, overlay_alpha,
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

    # applying color map is a two step process:
    # first, image is normalized to 0-1 and then this number in the 0-1 range is
    # mapped to a color using an instance of a subclass of Colormap
    # NB: it is a RGBA image
    color = cmap(bw)

    # make it to ranges between 0-255
    return color * 255


def apply_cam_transparency(cam, overlay_alpha, remove_white_pixels):
    '''
    Adjusts alpha layer of a RGBA cam and optionally makes whitish pixels transparent
    '''

    # original cam width and height
    orig_height = cam.shape[0]
    orig_width = cam.shape[1]

    # split image from alpha mask
    rgb_img = cam[:, :, :3]

    # set alpha matrix to the preset overlay_alpha
    cam[:, :, 3] *= 1 - overlay_alpha

    # reshape alpha matrix into a vector
    alpha = cam[:, :, 3].reshape(orig_width * orig_height)

    # make unimportant heatmap areas be transparent to avoid overlay color dilution (if set to true)
    # reshape rgb image
    rgb_img = rgb_img.reshape((orig_width * orig_height, rgb_img.shape[2]))
    if remove_white_pixels:
        # change in place
        alpha[np.sum(rgb_img, axis=1) > 0.9 * 255 * 3] = 0

    return cam


def get_image_with_cam(class_indices, class_weights, conv_img, original_img, overlay_alpha,
                       remove_white_pixels):
    '''
    Returns original image with cam overlay's applied
    '''

    # dimensions of original image
    image_width_height = original_img.shape[1], original_img.shape[0]

    original_img = (original_img * 255).astype(np.uint8)

    # loop through each class index and build its cam
    # overlay each cam over the original image
    for class_idx in class_indices:
        # get the class weights of the current class index (WEIGHTSxNUM_CLASSES)
        curr_class_weights = class_weights[:, class_idx]

        # if we are doing single cam, we just use the 'jet' colormap. allows us to use infinite classes for single cam
        if len(class_indices) == 1:
            colormap = SINGLE_CAM_OVERLAY_COLORMAP
        # otherwise the colormap we use is based on the class index
        else:
            # get the color map to use
            colormap = COLORMAPS[class_idx]

        # generate a cam in rgba form in same size as image
        cam = generate_cam_for_overlay(curr_class_weights, conv_img, colormap, image_width_height, overlay_alpha,
                                       remove_white_pixels).astype(np.uint8)

        # put rgba cam on top of original image
        original_img = blend_transparent(original_img, cam)

    return original_img


def get_final_cam_overlay_and_pred(model, image, conv_layer, overlay_alpha, show_top_x_classes, class_idx):
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

    # return cam and our class predictions
    return cam, pred


def get_single_layered_cam(model, image, conv_layer, class_idx, overlay_alpha):
    '''
    Returns the class activation map of a image class
    '''

    return get_final_cam_overlay_and_pred(model, image, conv_layer, overlay_alpha, None, class_idx)


def get_multi_layered_cam(model, image, conv_layer, overlay_alpha, show_top_x_classes):
    '''
    Returns the image-to-predict overlayed with multiple class activation maps
    '''

    return get_final_cam_overlay_and_pred(model, image, conv_layer, overlay_alpha, show_top_x_classes, None)


def get_final_cam_overlay_and_pred_large_image(model, cnn_trained_image_size, num_classes, image, conv_name,
                                               overlay_alpha,
                                               show_top_x_classes, class_idx):
    # check for invalid params
    if not (0 <= overlay_alpha <= 1):
        raise Exception("Invalid overlay_alpha given")

    if (image.shape[0] < cnn_trained_image_size) or (image.shape[1] < cnn_trained_image_size):
        raise Exception("Image too small")

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
    new_image = np.zeros((height, width, chn), dtype=np.uint8)

    # average prediction counter
    tot_pred = np.zeros(num_classes)
    counter = 0

    l = len(range(0, (width - cnn_trained_image_size) + 1, cnn_trained_image_size))
    w = len(range(0, (height - cnn_trained_image_size) + 1, cnn_trained_image_size))
    x = 0

    # go through all the sub-image sections in the horizontal
    for i in range(0, (width - cnn_trained_image_size) + 1, cnn_trained_image_size):

        t = time.clock()

        # go through all the sub-image sections in the vertical
        for j in range(0, (height - cnn_trained_image_size) + 1, cnn_trained_image_size):

            # get current sub-image
            sub_img = image[j:j + cnn_trained_image_size, i:i + cnn_trained_image_size, :]

            # determine whether single or multi cam
            if show_top_x_classes is not None:
                # get stacked cam of top x classes
                overlay_cam, pred = get_multi_layered_cam(model, sub_img, conv_layer,
                                                          overlay_alpha=overlay_alpha,
                                                          show_top_x_classes=show_top_x_classes)

            elif class_idx is not None:
                # single layer cam of specific class
                overlay_cam, pred = get_single_layered_cam(model, sub_img, conv_layer,
                                                           overlay_alpha=overlay_alpha,
                                                           class_idx=class_idx)

            tot_pred += pred
            counter += 1

            # put sub-image into correct spot of matrix (recreating image)
            new_image[j:j + cnn_trained_image_size, i:i + cnn_trained_image_size, :] = overlay_cam.astype(np.uint8)

        x += 1
        print("{:0.2f}% ({}/{} tiles) in {:0.2f}s".format(x * 100.0 / l, x * w, l * w, time.clock() - t))

    # return recreated image
    return new_image, tot_pred / counter


def englishify_pred(pred, classes, is_multi):
    '''
    Takes raw array predictions and makes it into a readable string
    '''

    result = ''

    # loop through each class and combine class with pred
    for i in range(len(classes)):
        result += '{}: {:.2f}%'.format(classes[i], pred[i])

        # if multi-layer, add colormap color
        if is_multi:
            result += ' ({})'.format(COLORMAPS[i][:-1])
        result += '\n'

    return result


def overlay_single_layered_cam_large_image(model, cnn_trained_image_size, classes, image, conv_name, class_name,
                                           overlay_alpha=0.5):
    '''
    Takes in an image of any size (>= cnn_trained_image_size) and returns a heatmap overlay based on the chosen class.
    Rightmost and/or bottommost parts of the image are ignored if image dimensions are not
    multiples of the CNN's trained image size. Additionally returns a predicted percent breakdown
    of each class' presence within the image
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
    class_name: str
        Name of class to generate cam on
    overlay_alpha : float, optional, default: 0.5, values: [0,1]
        Transparency of the cam overlay on top of the original image
    Returns
    --------
    new_image : array_like
    pred : str
    '''

    if class_name and class_name not in classes:
        raise Exception("Class not found")

    # class index of interest
    class_idx = classes.index(class_name)

    heatmap, raw_pred = get_final_cam_overlay_and_pred_large_image(model, cnn_trained_image_size, len(classes), image,
                                                                   conv_name,
                                                                   overlay_alpha=overlay_alpha,
                                                                   class_idx=class_idx, show_top_x_classes=None)

    # make readable before return
    return heatmap, englishify_pred(raw_pred, classes, is_multi=False)


def overlay_multi_layered_cam_large_image(model, cnn_trained_image_size, classes, image, conv_name, overlay_alpha=0.5,
                                          show_top_x_classes=3):
    '''
    Takes in an image of any size (>= cnn_trained_image_size) and returns a heatmap overlay of multiple classes.
    Rightmost and/or bottommost parts of the image are ignored if image dimensions are not
    multiples of the CNN's trained image size. Additionally returns a predicted percent breakdown
    of each class' presence within the image
    N.B. Currently supports as many classes as colormap values
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
    show_top_x_classes: int, optional, default: 3
        Overlays cam's of x classes that have the highest probabilities
    Returns
    --------
    new_image : array_like
    pred : str
    '''

    if len(classes) > len(COLORMAPS):
        raise Exception("Only {} classes are currently supported".format(str(len(COLORMAPS))))

    # check if we have less classes than the number of classes desired to show
    if len(classes) < show_top_x_classes:
        # show all the classes
        show_top_x_classes = len(classes)

    heatmap, raw_pred = get_final_cam_overlay_and_pred_large_image(model, cnn_trained_image_size, len(classes), image,
                                                                   conv_name,
                                                                   overlay_alpha=overlay_alpha,
                                                                   class_idx=None,
                                                                   show_top_x_classes=show_top_x_classes)

    # make readable before return
    return heatmap, englishify_pred(raw_pred, classes, is_multi=True)
