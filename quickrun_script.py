from keras.models import load_model
import time
import cv2

from heatmap_generator import overlay_multi_layered_cam_large_image, overlay_single_layered_cam_large_image

# load model
model = load_model('VGG19_trained.h5')

### configurations
trained_img_size = 304  # size of each tile (size our cnn was trained on)
conv_block = 'block5_conv4'  # name of the final convolutional layer (for vgg) (can view layers using model.summary())
classes = ['Blank', 'Gray Mat.', 'White Mat.']  # classes model was trained on. ordering matters
a = 0.3  # heatmap transparency


def generate_heatmap_single_layer(img_path, save_path):
    # image to do heatmap on
    im = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    # get heatmap of specified class
    heatmap_class = 'Gray Mat.'

    cam, avg_pred = overlay_single_layered_cam_large_image(model, trained_img_size, classes, im, conv_block,
                                                           heatmap_class,
                                                           overlay_alpha=a)

    # save heatmap generated
    cv2.imwrite(save_path, cv2.cvtColor(cam, cv2.COLOR_RGB2BGR))

    return cam, avg_pred


def generate_heatmap_multi(img_path, save_path):
    # image to do heatmap on
    im = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    # get heatmap with top 3 images in each tile
    cam, avg_pred = overlay_multi_layered_cam_large_image(model, trained_img_size, classes, im, conv_block,
                                                          overlay_alpha=a)

    # save heatmap generated
    cv2.imwrite(save_path, cv2.cvtColor(cam, cv2.COLOR_RGB2BGR))

    return cam, avg_pred


if __name__ == '__main__':
    start_time = time.clock()
    heatmap, avg_pred = generate_heatmap_single_layer('small_tiled_tissue.jpg', 'single_class_heatmap_GRAYMATTER.jpg')
    # heatmap, avg_pred = generate_heatmap_multi('multi_tiled_tissue.jpg', 'generated_heatmap.jpg')
    print(avg_pred)
    print(time.clock() - start_time, "seconds")
