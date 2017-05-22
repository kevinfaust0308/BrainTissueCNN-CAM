import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.backend import function
from PIL import Image


##############################################################
# Main functions (at bottom):

# 1. overlay_cam_large_image
# 2. overlay_multi_stacked_cam_large_image
##############################################################

# colormaps (https://matplotlib.org/examples/color/colormaps_reference.html)
# for multi layered cam, these are some possible colors
COLORMAPS = ['Reds', 'Blues', 'Greens', 'Purples', 'Oranges', 'Greys', 'jet']


def get_pred_and_conv_outputs(model, img, conv_name):
    '''
    Returns the predictions of the image evaluated by the model as well as the feature maps
    from the final convolutional layer
    '''
    
    img = np.expand_dims(img, axis=0)

    # variables for easy access to needed layers
    input_layer = model.layers[0]
    final_layer = model.layers[-1]

    # get feature maps from this conv layer
    final_conv_layer = None
    for l in model.layers:
        if l.name == conv_name:
            final_conv_layer = l
    # if didnt find the layer, raise error
    if not final_conv_layer:
        raise Exception('Layer name does not exist within model')

    # a function that takes in the input layer and outputs
    # 1. the feature maps after the last convolutional layer 2. prediction of image
    get_output = function([input_layer.input], [final_conv_layer.output, final_layer.output])

    # call function on our image
    conv_outputs, pred = get_output([img])
    conv_outputs = conv_outputs[0]  # 4d tensor to 3d tensor
    pred = pred[0]  # 2d array unnecessary. make 1d to make look better

    return pred, conv_outputs


def get_class_indexes(pred, class_idx, show_top_x_classes, threshold):
    '''
    Returns the indices within the prediction array which represent the classes of interest (the classes we want to generate cam's for)
    '''

    # check if we want a specific class index
    if class_idx is not None:

        # will just return cam of most confident predicted class
        if class_idx == -1:
            # get class index with the highest prediction
            class_idx = np.argmax(pred)

        # make into single element array
        class_indexes = np.asarray([class_idx])


    # check whether show_top_x_classes is true
    elif show_top_x_classes:
        pred_copy = pred.copy()
        class_indexes = []

        # loop as long as we have predictions
        while len(pred_copy) > 0 and show_top_x_classes > 0:
            # get max prediction index within original prediction array (untouched. has non-messed up indices) and add
            # to class_indexes
            class_indexes.append(np.where(pred == max(pred_copy)))
            # remove that max prediction value
            pred_copy = np.delete(pred_copy, np.argmax(pred_copy))

            # decrease counter
            show_top_x_classes -= 1

        # change from list to array
        class_indexes = np.asarray(class_indexes).flatten()

    # otherwise just get the class_indexes that have predictions greater than the threshold
    else:

        # get the indexes of the classes whose predicted probability is greater than the desired threshold
        class_indexes = np.argwhere(pred > threshold)
        # flatten to 1d array of indices
        class_indexes = class_indexes.flatten()

    return class_indexes


def generate_cam(class_weights, conv_outputs):
    '''
    Generates basic cam with dimensions the size of the feature maps given by conv_outputs
    '''

    ########## VECTORIZED CODE FOR GETTING CAM ##########

    # multiply length/width feature map dimensions
    width_times_len = conv_outputs.shape[0] * conv_outputs.shape[1]
    # our output shape will be the same as our feature map dimensions (w/o # of maps)
    output_shape = conv_outputs.shape[:2]
    # reshape into 2d
    temp = conv_outputs.reshape((width_times_len, conv_outputs.shape[2]))
    # multiply all our feature maps with its corresponding weights
    # reshape to class activation map
    cam = np.matmul(temp, class_weights).reshape(output_shape)

    ######################################################

    # .
    # .
    # .

    ########## NON-VECTORIZED CODE ##########

    #     # loop through all our class weights
    #     for i in range(len(class_weights)):

    #         # get ith class weight and multiply it with ith feature map
    #         # matrix-add that to our cam
    #         cam += class_weights[i] * conv_outputs[:, :, i]

    #########################################

    return cam


def generate_single_cam_overlay(class_weights, conv_outputs, colormap, image_width_height, overlay_alpha,
                                remove_white_pixels):
    '''
    Generates cam overlay over entire image
    '''

    # generate base cam
    cam = generate_cam(class_weights, conv_outputs)

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

    # make cam have alpha transparency (4 channel)
    # solid alpha for now
    cam = np.dstack((cam, np.ones((orig_width, orig_height))))

    # make unimportant heatmap areas be transparent to avoid overlay color dilution (if set to true)
    # reshape for looping
    cam = cam.reshape((orig_width * orig_height, cam.shape[2]))
    # loop through each pixel
    for pixel in cam:
        # if we want to remove whitish pixels, check if current pixel averages to be whitish (values close to 255)
        # then make it a transparent black pixel
        if remove_white_pixels and np.mean(pixel[:3]) > 0.90 * 255:
            pixel[:4] = 0
        # otherwise, just make the alpha equal to overlay_alpha
        else:
            pixel[3] = (1 - overlay_alpha) * 255

    # reshape back to normal and return
    return cam.reshape((orig_width, orig_height, cam.shape[1]))












#############

# slight modification for function overlay_multi_stacked_cam_large_image
# each class index will have an associated colormap in order to preserve coloring for a certain class across multiple subimages
# requires classes == colormaps

#############
def get_image_with_cam(class_indices, class_weights, conv_outputs, colormaps, original_img, overlay_alpha,
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

        # if we are only doing a single overlay, we'll only have a single element in our colormaps
        if len(colormaps) == 1:
            colormap = colormaps[0]
        else:
            # get the color map to use
            colormap = colormaps[class_idx]

        # generate a cam in rgba form in same size as image
        cam = generate_single_cam_overlay(curr_class_weights, conv_outputs, colormap, image_width_height, overlay_alpha,
                                          remove_white_pixels)

        # change cam to PIL image
        cam = Image.fromarray(cam.astype(np.uint8))

        # add cam overlay ontop of original image
        original_img.paste(cam, (0, 0), cam)

    # change PIL image to numpy array
    original_img = np.asarray(list(original_img.getdata()))
    # reshape and return
    return original_img.reshape(image_width_height + (3,)).astype(np.uint8)



############

# original implementation. colormap will be assigned to order of decreasing class probabilities

############
# def get_image_with_cam(class_indices, class_weights, conv_outputs, colormaps, original_img, overlay_alpha,
#                        remove_white_pixels):
#     '''
#     Returns original image with cam overlay's applied
#     '''

#     # dimensions of original image
#     image_width_height = original_img.shape[1], original_img.shape[0]

#     # set the class activation map to be the original image which we will build up on top of
#     # change to PIL image
#     original_img = Image.fromarray((original_img * 255).astype(np.uint8))

#     # loop through each class index and build its cam
#     # overlay each cam over the original image
#     for cmap_idx, class_idx in enumerate(class_indices):
#         # get the class weights of the current class index (WEIGHTSxNUM_CLASSES)
#         curr_class_weights = class_weights[:, class_idx]

#         # get the color map to use
#         colormap = colormaps[cmap_idx]

#         # generate a cam in rgba form in same size as image
#         cam = generate_single_cam_overlay(curr_class_weights, conv_outputs, colormap, image_width_height, overlay_alpha,
#                                           remove_white_pixels)

#         # change cam to PIL image
#         cam = Image.fromarray(cam.astype(np.uint8))

#         # add cam overlay ontop of original image
#         original_img.paste(cam, (0, 0), cam)

#     # change PIL image to numpy array
#     original_img = np.asarray(list(original_img.getdata()))
#     # reshape and return
#     return original_img.reshape(image_width_height + (3,)).astype(np.uint8)


def get_cam(model, image, conv_name, class_idx=-1, overlay=False, overlay_alpha=0.5,
                           cmap=None):
    '''   
    Returns the class activation map of a image class, optionally as an overlay over the original image
    
    Class activation map is an unsupervised way of doing object localization with accuracy near par with supervised methods
    
    Parameters
    -----------
    model : '~keras.models'
        Model to generate prediction and CAM off of

    image : ndarray
        Matrix representation 
        
    conv_name : str
        Name of the convolutional layer which we will generate the CAM
        
    class_idx: int, optional, default: -1
        If -1, defaults to using index of class with highest probability of representing the given image
        
    overlay: boolean, optional, default: False
        If False, only cam is generated
        
    overlay_alpha: float, optional, default: 0.5, values: [0,1]
        Transparency of the cam overlay on top of the original image. 'overlay_alpha' is ignored if 'overlay' is False
        
    cmap : `~matplotlib.colors.Colormap`, optional, default: None
        If None, defaults to 'jet' if 'overlay' is True otherwise cam is left as B/W

    Returns
    --------
    cam : array_like
    pred: array_like
    '''

    # check for invalid params
    if not (0 <= overlay_alpha <= 1):
        raise Exception("Invalid overlay_alpha given")

    # image
    original_img = image/255.

    # get predictions and final convolutional layer feature maps
    pred, conv_outputs = get_pred_and_conv_outputs(model, original_img, conv_name)

    # determine which class we will get a cam for
    class_indices = get_class_indexes(pred, class_idx, None, None)

    # Get the input weights to the final layer
    class_weights = model.layers[-1].get_weights()[0]

    # check if we want cam as overlay or just the cam itself
    if overlay:
        # if no color specified, default to 'jet'
        if not cmap:
            cmap = 'jet'
        # get single overlayed cam over image using the passed in color map and keeping all pixels from the cam shown
        cam = get_image_with_cam(class_indices, class_weights, conv_outputs, [cmap], original_img, overlay_alpha, False)
    else:
        # since we just want a single raw cam, we can get the weights from the class of interest directly
        w = class_weights[:, class_indices[0]]
        cam = generate_cam(w, conv_outputs)
        # add color map if passed in
        if cmap:
            cam = apply_color_map_on_BW(cam, cmap)

    # return cam and our class predictions
    return cam, pred


def get_multi_layered_cam(model, image, classes, conv_name, overlay_alpha=0.3, threshold=0.3,
                          show_top_x_classes=None, pretty_top_predictions=True):  
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
        
    conv_name : str
        Name of the convolutional layer which we will generate the CAM
        
    overlay_alpha: float, optional, default: 0.3, values: [0,1]
        Transparency of the cam overlay on top of the original image
        
    threshold: float, optional, default: 0.3, values: [0.2,1]
        Only class probabilities greater that this threshold will have their cam's shown as an overlay. 
        
    show_top_x_classes: int, optional, default: None, values: [0,5]
        if None, fallsback to threshold. Otherwise only overlays cam's of x classes that have the highest probabilities. 'threshold' is ignored if 'show_top_x_classes' is False
        
    pretty_top_predictions: boolean, optional, default: True
        Changes raw image prediction scores to have class labels and what color each class represents on the overlay

    Returns
    --------
    cam : array_like
    pred: array_like
    '''

    # model was trained on these classes
    CLASSES = classes

    # check for invalid params
    if not (0 <= overlay_alpha <= 1):
        raise Exception("Invalid overlay_alpha given")

    if not (0.2 <= threshold <= 1):
        raise Exception("Invalid threshold given")

    if show_top_x_classes and (not (0 <= show_top_x_classes <= 5)):
        raise Exception("Invalid show_top_x_classes given")

    # check if we have less classes than the number of classes desired to show
    if show_top_x_classes and len(CLASSES) < show_top_x_classes:
        # show all the classes
        show_top_x_classes = len(CLASSES)

    # image
    original_img = image/255.

    # get predictions and final convolutional layer feature maps
    pred, conv_outputs = get_pred_and_conv_outputs(model, original_img, conv_name)

    # determine which classes we will get a cam for
    class_indices = get_class_indexes(pred, None, show_top_x_classes, threshold)

    # Get the input weights to the final layer
    class_weights = model.layers[-1].get_weights()[0]

    # apply cam overlays on top of image (and make whitish areas of cam overlay transparent)
    cam = get_image_with_cam(class_indices, class_weights, conv_outputs, COLORMAPS, original_img, overlay_alpha, True)

    # check if we should englishify pred
    if pretty_top_predictions:
        new_pred = {}
        
        
        
        ##########################
        # changed for overlay_multi_stacked_cam_large_image
        ##########################
        
        for j in class_indices:
            # get label for this class and add as a key and add the prediction score and cmap legend
            new_pred[CLASSES[j]] = [pred[j], COLORMAPS[j]]
        pred = new_pred
          
        # for i, j in enumerate(class_indices):
#             # get label for this class and add as a key and add the prediction score and cmap legend
#             new_pred[CLASSES[j]] = [pred[j], COLORMAPS[i]]
#         pred = new_pred



    # return cam and our class predictions
    return cam, pred




def overlay_prediction_on_image(image, pred, color): 
    '''
    Takes a CAM overlayed image and writes predictions over it inplace
    
    Parameters
    -----------
    image : ndarray
        Matrix representation 
        
    pred : dict of list
        Has label as key and list of accuracy + additional information in list format
        
    color : str
        RGB tuple
    
    Returns
    --------
    None
    '''
        
    # change pred to list
    my_list = []
    for obj in pred:
        
        listy = pred[obj].copy()

        # make pred accuracy 2 decimals
        p = '{:.2f}'.format(listy.pop(0)*100)
        
        # check if we only have accuracy or if we have other info such as color legend
        if len(listy) > 0:
            text = '{}: {}% {}'.format(obj, p, listy)
        else:
            text = '{}: {}%'.format(obj, p)
        
        # our new list will have object name, acc
        my_list.append(text)
    
    
    # text color
    text_color = color
    # starting top offset
    top_offset = 30
    
    # loop through all those pred labels and add to image
    for label in my_list:

        cv2.putText(
            image, # put text on this image
            "{}".format(label), # text
            (10,top_offset),# left offset, top offset
            cv2.FONT_HERSHEY_SIMPLEX, # font family
            0.6, # scale of text size
            text_color, # color
            1, # line width
            cv2.LINE_AA # line type
        )

        # move next text down for spacing
        top_offset += 30
        
        




def get_multi_stacked_cam(model, image, classes, conv_name, overlay_alpha=0.3, threshold=0.3,
                          show_top_x_classes=None, pretty_top_predictions=True):  
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
        
    conv_name : str
        Name of the convolutional layer which we will generate the CAM
        
    overlay_alpha: float, optional, default: 0.3, values: [0,1]
        Transparency of the cam overlay on top of the original image
        
    threshold: float, optional, default: 0.3, values: [0.2,1]
        Only class probabilities greater that this threshold will have their cam's shown as an overlay. 
        
    show_top_x_classes: int, optional, default: None, values: [0,5]
        if None, fallsback to threshold. Otherwise only overlays cam's of x classes that have the highest probabilities. 'threshold' is ignored if 'show_top_x_classes' is False
        
    pretty_top_predictions: boolean, optional, default: True
        Changes raw image prediction scores to have class labels and what color each class represents on the overlay

    Returns
    --------
    cam : array_like
    pred: array_like
    '''

    # model was trained on these classes
    CLASSES = classes

    # check for invalid params
    if not (0 <= overlay_alpha <= 1):
        raise Exception("Invalid overlay_alpha given")

    if not (0.2 <= threshold <= 1):
        raise Exception("Invalid threshold given")

    if show_top_x_classes and (not (0 <= show_top_x_classes <= 5)):
        raise Exception("Invalid show_top_x_classes given")

    # check if we have less classes than the number of classes desired to show
    if show_top_x_classes and len(CLASSES) < show_top_x_classes:
        # show all the classes
        show_top_x_classes = len(CLASSES)

    # image
    original_img = image/255.

    # get predictions and final convolutional layer feature maps
    pred, conv_outputs = get_pred_and_conv_outputs(model, original_img, conv_name)

    # determine which classes we will get a cam for
    class_indices = get_class_indexes(pred, None, show_top_x_classes, threshold)

    # Get the input weights to the final layer
    class_weights = model.layers[-1].get_weights()[0]

    # apply cam overlays on top of image (and make whitish areas of cam overlay transparent)
    cam = get_image_with_cam(class_indices, class_weights, conv_outputs, COLORMAPS, original_img, overlay_alpha, True)

    # check if we should englishify pred
    if pretty_top_predictions:
        new_pred = {}
        
        ##########################
        # changed for overlay_multi_stacked_cam_large_image
        ##########################
        
        for j in class_indices:
            # get label for this class and add as a key and add the prediction score and cmap legend
            new_pred[CLASSES[j]] = [pred[j], COLORMAPS[j]]
        pred = new_pred
          
        # for i, j in enumerate(class_indices):
#             # get label for this class and add as a key and add the prediction score and cmap legend
#             new_pred[CLASSES[j]] = [pred[j], COLORMAPS[i]]
#         pred = new_pred

    # return cam and our class predictions
    return cam, pred








def overlay_cam_large_image(model, cnn_trained_image_size, classes, image, conv_name, class_idx, overlay_alpha=0.5,
                           cmap=None):
    ''' 
    Takes in a larger image in multiples of the image sizes trained on the CNN and returns a CAM overlay. Returns average 
    prediction of all sub-images
    
    Class activation map is an unsupervised way of doing object localization with accuracy near par with supervised methods
    
    Parameters
    -----------
    model : '~keras.models'
        Model to generate prediction and CAM off of
        
    cnn_trained_image_size: str
        Image size trained on the CNN
        
    classes: list of str

    image : ndarray
        Matrix representation. Must have dimensions that are multiples of 'cnn_trained_image_size'
        
    conv_name : str
        Name of the convolutional layer which we will generate the CAM
        
    class_idx: int
        Generate CAM of this image category
        
    overlay_alpha: float, optional, default: 0.3, values: [0,1]
        Transparency of the cam overlay on top of the original image
          
    cmap : `~matplotlib.colors.Colormap`, optional, default: None
        If None, defaults to 'jet' 
    
    Returns
    --------
    new_image : array_like
    my_dict : dict
    '''
     
    # shape of image
    height, width, chn = image.shape
    
    # new image placeholder
    new_image = np.zeros((height+1, width+1, chn))
    
    # we will keep track of the average prediction
    avg_pred = np.asarray([0.0] * len(classes))
    counter = 0
    
    # this is a slow process so add progress bar
    print('Analyzing', end='')
    
    # go through all the sub-image sections in the horizontal
    for i in list(range(0,(width-cnn_trained_image_size)+1,cnn_trained_image_size)):
        # go through all the sub-image sections in the vertical
        for j in list(range(0,(height-cnn_trained_image_size)+1,cnn_trained_image_size)):
            # get current sub-image
            sub_img = image[j:j+cnn_trained_image_size,i:i+cnn_trained_image_size,:]
            
            # get cam of desired class
            overlay_cam, pred = get_cam(model, sub_img, conv_name, class_idx, overlay=True, overlay_alpha=overlay_alpha, cmap=cmap)
            
            # add pred to total
            avg_pred += pred
            # add counter in order to divide avg_pred at end to get average
            counter += 1
            
            # put sub-image into correct spot of matrix (recreating image)
            new_image[j:j+cnn_trained_image_size,i:i+cnn_trained_image_size,:] = overlay_cam
            
            # progress bar
            print('.', end='')
            
    # progress bar
    print('Complete')
   
    # calculate average pred
    avg_pred = avg_pred / counter
    
    my_dict = {}
    for i in range(len(classes)):
        my_dict[classes[i]] = avg_pred[i]
           
    # return recreated image and average predictions
    return new_image.astype(np.uint8), my_dict




def overlay_multi_layered_cam_large_image(model, cnn_trained_image_size, classes, image, conv_name, overlay_alpha=0.5, overlay_predictions=False, overlay_text_color=(0,0,0)):
    ''' 
    Takes in a larger image in multiples of the image sizes trained on the CNN and returns a multi CAM overlay. Each
    overlay on each subimage will have only its top 3 predicted classes shown.
    
    Returns total average prediction if 'overlay_predictions' is False. Otherwise, shows individual sub-image predictions
    as overlay
    
    Class activation map is an unsupervised way of doing object localization with accuracy near par with supervised methods
    
    Parameters
    -----------
    model : '~keras.models'
        Model to generate prediction and CAM off of
        
    cnn_trained_image_size: str
        Image size trained on the CNN
        
    classes: list of str

    image : ndarray
        Matrix representation. Must have dimensions that are multiples of 'cnn_trained_image_size'
        
    conv_name : str
        Name of the convolutional layer which we will generate the CAM
                
    overlay_alpha : float, optional, default: 0.3, values: [0,1]
        Transparency of the cam overlay on top of the original image
          
    overlay_predictions : bool, optional, default: False
        Overlay prediction of each sub-image
          
    overlay_text_color : str
        RGB color of the overlay prediction text. Ignored if 'overlay_predictions' is False
    
    Returns
    --------
    new_image : array_like
    my_dict : dict
    '''
    
    
    # show user what color each class on the cam will be. currently restricted to having 5 classes (one for each colormap)
    for i, _class in enumerate(classes):
        print('{} --> {}'.format(_class, COLORMAPS[i][:-1]))
              
    #####################################################################################################################
     
    # shape of image
    height, width, chn = image.shape
    
    # new image placeholder
    new_image = np.zeros((height+1, width+1, chn))
    
    # we will keep track of the average prediction
    avg_pred = np.asarray([0.0] * len(classes))
    counter = 0
    
    # show top 3 classes for each subimage
    IMAGES_TO_SHOW = 3
    
    # this is a slow process so add progress bar
    print('Analyzing', end='')
    
    # go through all the sub-image sections in the horizontal
    for i in list(range(0,(width-cnn_trained_image_size)+1,cnn_trained_image_size)):
        # go through all the sub-image sections in the vertical
        for j in list(range(0,(height-cnn_trained_image_size)+1,cnn_trained_image_size)):
            # get current sub-image
            sub_img = image[j:j+cnn_trained_image_size,i:i+cnn_trained_image_size,:]  
            
            # show on subimage the top 3 predictions 
            if overlay_predictions:
                # get cam of desired class (and prettyify predictions for overlay purposes)
                overlay_cam, pred = get_multi_layered_cam(model, sub_img, classes, conv_name, overlay_alpha=overlay_alpha, threshold=0.3,
                          show_top_x_classes=IMAGES_TO_SHOW, pretty_top_predictions=True)
            
                overlay_prediction_on_image(overlay_cam, pred, overlay_text_color)
                
            else:  
                # get cam of desired class
                overlay_cam, pred = get_multi_layered_cam(model, sub_img, classes, conv_name, overlay_alpha=overlay_alpha, threshold=0.3,
                          show_top_x_classes=IMAGES_TO_SHOW, pretty_top_predictions=False)
                
                # add pred to total
                avg_pred += pred
                
                # add counter in order to divide avg_pred at end to get average
                counter += 1
              
            # put sub-image into correct spot of matrix (recreating image)
            new_image[j:j+cnn_trained_image_size,i:i+cnn_trained_image_size,:] = overlay_cam
            
            # progress bar
            print('.', end='')
            
    # progress bar
    print('Complete')
    
    # only if we are not overlay prediction do we need avg pred
    my_dict = {}
    if not overlay_predictions:
        # calculate average pred
        avg_pred = avg_pred / counter

        for i in range(len(classes)):
            my_dict[classes[i]] = avg_pred[i]
               
    # return recreated image    
    return new_image.astype(np.uint8), my_dict
