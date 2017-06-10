from keras.models import load_model
import time
import cv2

from multi_CAM_large_image import overlay_multi_layered_cam_large_image, overlay_single_layered_cam_large_image

# load model
model = load_model('VGG19_trained.h5')

### configurations
overlay_pred = True  # get heatmap with predictions written over each tile
trained_img_size = 304  # size of each tile (size our cnn was trained on)
conv_block = 'block5_conv4'  # name of the final convolutional layer (for vgg) (can view layers using model.summary())
text_color = (0, 255, 80)  # overlay text color
classes = ['Blank', 'Gray Mat.', 'White Mat.']  # classes model was trained on. ordering matters
a = 0.3  # heatmap transparency


def generate_heatmap_single_layer(img_path, save_path):
    # image to do heatmap on
    im = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    # get heatmap of specified class
    heatmap_class = 'Gray Mat.'

    cam = overlay_single_layered_cam_large_image(model, trained_img_size, classes, im, conv_block, heatmap_class,
                                                 overlay_alpha=a,
                                                 overlay_predictions=overlay_pred,
                                                 overlay_text_color=text_color)

    # save heatmap generated
    cv2.imwrite(save_path, cv2.cvtColor(cam, cv2.COLOR_RGB2BGR))

    return cam


def generate_heatmap_multi(img_path, save_path):
    # image to do heatmap on
    im = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    # get heatmap with top 3 images in each tile
    cam = overlay_multi_layered_cam_large_image(model, trained_img_size, classes, im, conv_block, overlay_alpha=a,
                                                overlay_predictions=overlay_pred, overlay_text_color=text_color)

    # save heatmap generated
    cv2.imwrite(save_path, cv2.cvtColor(cam, cv2.COLOR_RGB2BGR))

    return cam


if __name__ == '__main__':
    start_time = time.clock()
    heatmap = generate_heatmap_single_layer('small_tiled_tissue.jpg', 'generated_heatmap_SINGLE.png')
    # heatmap = generate_heatmap_multi('multi_tiled_tissue.jpg', 'generated_heatmap.png')
    print(time.clock() - start_time, "seconds")
