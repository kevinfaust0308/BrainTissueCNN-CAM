from keras.models import load_model
import time
import cv2

from heatmap_generator import overlay_multi_layered_cam_large_image, overlay_single_layered_cam_large_image

##################################################### CONFIGS #########################################################

# file io
INPUT_IMAGE_PATH = 'input.jpg'
OUTPUT_IMAGE_PATH = 'output_heatmap.jpg'

# model settings
MODEL_PATH = 'VGG19_256.h5'
TRAINED_IMG_SIZE = 256  # size of each tile (size our cnn was trained on)
CONV_BLOCK = 'block5_pool'  # name of the final convolutional layer (for vgg) (can view layers using model.summary())
CLASSES = ['Blank', 'Gray Mat.', 'Lesion', 'White Mat.']  # classes model was trained on. ordering matters

# design
OVERLAY_PRED = True  # get heatmap with predictions written over each tile
OVERLAY_TEXT_COLOR = (0, 255, 80)  # overlay text color
ALPHA = 0.3  # heatmap transparency

# single layer configs
HEATPMAP_CLASS = 'Gray Mat.'  # get heatmap of specified class

# multi layer configs
SHOW_TOP_X_CLASSES = len(CLASSES)  # show colors of all classes

################################################# END OF CONFIGS ######################################################

# load model
model = load_model(MODEL_PATH)


def generate_heatmap(img_path, save_path, multi=True):
    # image to do heatmap on
    im = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    if multi:
        cam, avg_pred = overlay_multi_layered_cam_large_image(model, TRAINED_IMG_SIZE, CLASSES, im, CONV_BLOCK,
                                                              show_top_x_classes=SHOW_TOP_X_CLASSES,
                                                              overlay_alpha=ALPHA, overlay_predictions=OVERLAY_PRED,
                                                              overlay_text_color=OVERLAY_TEXT_COLOR)
    else:
        cam, avg_pred = overlay_single_layered_cam_large_image(model, TRAINED_IMG_SIZE, CLASSES, im, CONV_BLOCK,
                                                               class_name=HEATPMAP_CLASS, overlay_alpha=ALPHA,
                                                               overlay_predictions=OVERLAY_PRED,
                                                               overlay_text_color=OVERLAY_TEXT_COLOR)

    # save heatmap generated
    cv2.imwrite(save_path, cv2.cvtColor(cam, cv2.COLOR_RGB2BGR))

    return avg_pred


if __name__ == '__main__':
    start_time = time.clock()
    # heatmap, avg_pred = generate_heatmap_single_layer('small_tiled_tissue.jpg', 'single_class_heatmap_GRAYMATTER.jpg')
    avg_pred = generate_heatmap(INPUT_IMAGE_PATH, OUTPUT_IMAGE_PATH)
    print('\n' + avg_pred + '\n')
    print(time.clock() - start_time, "seconds")
