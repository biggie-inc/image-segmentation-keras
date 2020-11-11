import glob
import random
import json
import os
from numpy.core.defchararray import title
from scipy.ndimage import interpolation
import six

###
import tensorflow as tf
import matplotlib.pyplot as plt
###

import cv2
import numpy as np
from tqdm import tqdm
from time import time

from .train import find_latest_checkpoint
from .data_utils.data_loader import get_image_array, get_segmentation_array,\
    DATA_LOADER_SEED, class_colors, get_pairs_from_paths
from .models.config import IMAGE_ORDERING


random.seed(DATA_LOADER_SEED)


def model_from_checkpoint_path(checkpoints_path):

    from .models.all_models import model_from_name
    assert (os.path.isfile(checkpoints_path+"_config.json")
            ), "Checkpoint not found."
    model_config = json.loads(
        open(checkpoints_path+"_config.json", "r").read())
    latest_weights = find_latest_checkpoint(checkpoints_path)
    assert (latest_weights is not None), "Checkpoint not found."
    model = model_from_name[model_config['model_class']](
        model_config['n_classes'], input_height=model_config['input_height'],
        input_width=model_config['input_width'])
    print("loaded weights ", latest_weights)
    model.load_weights(latest_weights)
    return model


def get_colored_segmentation_image(seg_arr, n_classes, colors=class_colors):
    output_height = seg_arr.shape[0]
    output_width = seg_arr.shape[1]
    

    seg_img = np.zeros((output_height, output_width, 3))
    #####
    # print(f'seg_arr shape: {seg_arr.shape}')
    # print(f'seg_img shape: {seg_img.shape}')
    #####

    for c in range(n_classes):
        seg_arr_c = seg_arr[:, :] == c
        seg_img[:, :, 0] += ((seg_arr_c)*(colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((seg_arr_c)*(colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((seg_arr_c)*(colors[c][2])).astype('uint8')
    
    #cv2.imwrite('init_seg_img.jpg', seg_img)

    return seg_img


def get_legends(class_names, colors=class_colors):

    n_classes = len(class_names)
    legend = np.zeros(((len(class_names) * 25) + 25, 125, 3),
                      dtype="uint8") + 255

    class_names_colors = enumerate(zip(class_names[:n_classes],
                                       colors[:n_classes]))

    for (i, (class_name, color)) in class_names_colors:
        color = [int(c) for c in color]
        cv2.putText(legend, class_name, (5, (i * 25) + 17),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
        cv2.rectangle(legend, (100, (i * 25)), (125, (i * 25) + 25),
                      tuple(color), -1)

    return legend


def overlay_seg_image(inp_img, seg_img):
    orininal_h = inp_img.shape[0]
    orininal_w = inp_img.shape[1]
    seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))

    fused_img = (inp_img/2 + seg_img/2).astype('uint8')
    return fused_img


def concat_lenends(seg_img, legend_img):

    new_h = np.maximum(seg_img.shape[0], legend_img.shape[0])
    new_w = seg_img.shape[1] + legend_img.shape[1]

    out_img = np.zeros((new_h, new_w, 3)).astype('uint8') + legend_img[0, 0, 0]

    out_img[:legend_img.shape[0], :  legend_img.shape[1]] = np.copy(legend_img)
    out_img[:seg_img.shape[0], legend_img.shape[1]:] = np.copy(seg_img)

    return out_img

#######
def largest_contours(pr, n_classes):
    final = np.zeros((960, 1280), dtype='uint8')
    for i in range(1, n_classes):
        prediction = pr[:,:] == i
        prediction = prediction.astype('uint8')
        contours, _ = cv2.findContours(prediction.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        largest_contour = sorted(contours, key=cv2.contourArea, reverse= True)[0]
        cv2.drawContours(final, [largest_contour], -1, i,-1)

    return final

def get_cropped(img, coord): # coords[y1, x1, y2, x2] https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/model.py line 2482
    offset_height, offset_width, target_height, target_width = coord
    x = tf.image.crop_to_bounding_box(
        np.float32(img), offset_height, offset_width, target_height-offset_height, target_width-offset_width
    )

    return tf.keras.preprocessing.image.array_to_img(
        x, data_format=None, scale=True, dtype=None
    )

def get_theta(img,roi):
    window_img = get_cropped(img, roi)
    # print('h', window_img.height, 'w', window_img.width)
    a = 0
    b = 0
    for i in range(window_img.height):
        if np.mean(window_img.getpixel((i,0))) < 250:
            print(i, window_img.getpixel((i,0)))
            a = i
            break
    for i in range(window_img.width):
        if np.mean(window_img.getpixel((0,i))) < 250:
            print(i, window_img.getpixel((0,i)))
            b = i
            break
    # print('a',a,'b',b)
    return window_img, np.arctan(a/b)


def get_plate_xy_min_max(seg_arr):
    seg_arr2 = np.where(seg_arr==2) # outputs two arrays of [all y's] and [all x's] for class 2: plate
    
    ymax = max(seg_arr2[0])
    ymin = min(seg_arr2[0])
    xmax = max(seg_arr2[1])
    xmin = min(seg_arr2[1])

    return xmin, xmax, ymin, ymax 


def get_window_xy_min_max(seg_arr):
    seg_arr1 = np.where(seg_arr==1) # outputs two arrays of [all y's] and [all x's] for class 1: window
    
    ymax = max(seg_arr1[0])
    ymin = min(seg_arr1[0])
    xmax = max(seg_arr1[1])
    xmin = min(seg_arr1[1])
    
    # np.savetxt('y1_x1_y2_x2.txt', (ymin,xmin,ymax,xmax), delimiter=',', fmt='%i')

    return xmin, xmax, ymin, ymax 


def get_pixels_per_inch(plate_xmin, plate_xmax):
    pixels_per_inch = (plate_xmax - plate_xmin)/12 # all license plates are 12" wide
    return pixels_per_inch


def get_window_h_w_centriods(window_xmin, window_xmax, window_ymin, window_ymax, pixels_per_inch):
    window_width = (window_xmax - window_xmin)/pixels_per_inch
    window_height = (window_ymax - window_ymin)/pixels_per_inch
    w_center = int((window_xmax + window_xmin)/2)
    h_center = int((window_ymax + window_ymin)/2)
    return window_height, window_width, h_center, w_center


def plot_orig_and_overlay(inp, seg_img):

    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,6))

    ax1.imshow(inp)
    ax1.axis('off')

    ax2.imshow(seg_img)
    ax2.axis('off')

    ax3.imshow(inp)
    ax3.imshow(seg_img, alpha=0.6)
    ax3.axis('off')

    plt.tight_layout()

    return fig


def get_window_cutlines(seg_arr, seg_img, coords, window_height_adj, pixels_per_inch, hyp=None, theta=None):
    ppi = pixels_per_inch
    window_xmin, window_xmax, window_ymin, window_ymax = coords
    window_only = seg_img[window_ymin:window_ymax, window_xmin:window_xmax]
    cv2.imwrite('./window_only.jpg', window_only)
    # create a blank canvas np.zeros(window_height, seg_arr[1])
    print(f'get window cutlines window_only.shape: {window_only.shape}')
    new_dims = window_only.shape[1], int(window_height_adj*ppi)
    stretched_image = cv2.resize(window_only, new_dims, interpolation=cv2.INTER_NEAREST)

    # warp image to canvas
    # transform = cv2.getPerspectiveTransform(ordered_corners, dimensions)
    # warped_img = cv2.warpPerspective(seg_img, transform, (seg_arr[1], window_height_adj*ppi))

    # get new contours
    contours, _ = cv2.findContours(stretched_image.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    largest_contour = sorted(contours, key=cv2.contourArea, reverse= True)[0]

    # add h and v lines
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,10))

    hlines = [z for z in range(window_ymin-ppi, window_ymax+ppi, ppi)]
    vlines = [z for z in range(window_xmin-ppi, window_xmax+ppi, ppi)]

    ax2.hlines(hlines, xmin=window_xmin-ppi, xmax=window_xmax+ppi, linestyle=':', color='gray')
    ax2.vlines(vlines, ymin=window_ymin-ppi,ymax=window_ymax+ppi, linestyle=':', color='gray')

    # return figure with original and warped image

    ax1.plot(seg_arr[window_ymin:window_ymax, window_xmin:window_xmax])
    ax2.plot(largest_contour)

    ax1.set(title='Before Transform')
    ax1.axis('off')

    ax2.set(title='After Transform')
    ax2.axis('off')

    plt.savefig('cutline_before_after.jpg');

    if hyp and theta is None:
        pass

#######

def visualize_segmentation(seg_arr, inp_img=None, n_classes=None,
                           colors=class_colors, class_names=None,
                           overlay_img=False, show_legends=False,
                           prediction_width=None, prediction_height=None):

    if n_classes is None:
        n_classes = np.max(seg_arr)

    #####
    plate_xmin, plate_xmax, plate_ymin, plate_ymax = get_plate_xy_min_max(seg_arr)
    pixels_per_inch = get_pixels_per_inch(plate_xmin, plate_xmax)
    window_xmin, window_xmax, window_ymin, window_ymax = get_window_xy_min_max(seg_arr)
    window_height, window_width, h_center, w_center = get_window_h_w_centriods(window_xmin, window_xmax, window_ymin, window_ymax, pixels_per_inch)

    window_img, theta = get_theta(inp_img, [window_ymin, window_xmin, window_ymax, window_xmax])
    hyp = window_img.height / np.cos(theta)

    plate_width2 = plate_xmax - plate_xmin
    plate_height2 = plate_width2 / 2                
    window_height_adj = (hyp / plate_height2)  * 7.0
    print(f"Visible rear window: {window_width}w X {window_height_adj}h")

    #####


    seg_img = get_colored_segmentation_image(seg_arr, n_classes, colors=colors)


    #####
    # plot plate rect
    #cv2.rectangle(seg_img, (plate_xmin, plate_ymin), (plate_xmax, plate_ymax), (0,0,255), 2)
    # add window dimensions
    cv2.putText(seg_img, f'Window Height: {window_height_adj:.3f} ', (int(window_xmin), int(window_ymin - 8)),
                    cv2.FONT_HERSHEY_DUPLEX, .75, (0, 0, 0), 1)
    cv2.putText(seg_img, f'Window Width: {window_width:.3f}', (int(window_xmin), int(window_ymax + 15)),
                    cv2.FONT_HERSHEY_DUPLEX, .75, (0, 0, 0), 1)
    cv2.circle(seg_img, (w_center, window_ymax), 2, (255,255,255))
    cv2.circle(seg_img, (window_xmin, h_center), 2, (255,235,5))
    cv2.circle(seg_img, (int((plate_xmax + plate_xmin)/2), plate_ymin), 2, (255,255,255))
    cv2.circle(seg_img, (plate_xmin, int((plate_ymax + plate_ymin)/2)), 2, (255,235,5))

    get_window_cutlines(seg_arr, seg_img, [window_xmin, window_xmax, window_ymin, window_ymax], window_height_adj, pixels_per_inch)
    #####

    
    if inp_img is not None:
        orininal_h = inp_img.shape[0]
        orininal_w = inp_img.shape[1]
        seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))
        # resizes the seg_img to original image size - not needed after pr_resize var

    if (prediction_height is not None) and (prediction_width is not None):
        seg_img = cv2.resize(seg_img, (prediction_width, prediction_height))
        if inp_img is not None:
            inp_img = cv2.resize(inp_img,
                                 (prediction_width, prediction_height))

    if overlay_img:
        assert inp_img is not None
        seg_img = overlay_seg_image(inp_img, seg_img)

    if show_legends:
        assert class_names is not None
        legend_img = get_legends(class_names, colors=colors)

        seg_img = concat_lenends(seg_img, legend_img)

    return seg_img


def predict(model=None, inp=None, out_fname=None,
            checkpoints_path=None, overlay_img=False,
            class_names=None, show_legends=False, colors=class_colors,
            prediction_width=None, prediction_height=None):

    if model is None and (checkpoints_path is not None):
        model = model_from_checkpoint_path(checkpoints_path)

    assert (inp is not None)
    assert ((type(inp) is np.ndarray) or isinstance(inp, six.string_types)),\
        "Input should be the CV image or the input file name"

    #####
    filename = inp.split("/")[-1].split(".")[0]
    #####

    if isinstance(inp, six.string_types):
        inp = cv2.imread(inp)

    assert len(inp.shape) == 3, "Image should be h,w,3 "

    output_width = model.output_width
    output_height = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes

    x = get_image_array(inp, input_width, input_height,
                        ordering=IMAGE_ORDERING)
    # x appears to be a normalized output
    pr = model.predict(np.array([x]))[0]
    pr = pr.reshape((output_height,  output_width, n_classes)).argmax(axis=2)
    # pr is the pixel-wise class output = 0,1,2
    
    #####
    pr_reshape = pr.reshape((output_height, output_width, 1)).astype('uint8')
    pr_resized = cv2.resize(pr_reshape, dsize=(inp.shape[1], inp.shape[0]), interpolation=cv2.INTER_NEAREST) #(960,1280,1)
    # np.savetxt('pr_resized.txt', pr_resized, delimiter=',', fmt='%i')
    pr_main_contours = largest_contours(pr_resized, n_classes) # returns numpy array with largest contour of each class
    #print(f'window cntr only shape: {window_cntr_only.shape}')
    #print(f'window cntr only unique: {np.unique(window_cntr_only)}')
    #print(f'window cntr only: {window_cntr_only}')

    # wco_coords  = get_window_xy_min_max(window_cntr_only)
    # #print(f'window cntr only min maxs: {wco_xmin}, {wco_xmax}, {wco_ymin}, {wco_ymax}')
    # window_cntr_only = window_cntr_only.reshape(960, 1280, 1)
    # window_contour_cropped = trim_axes(window_cntr_only, wco_coords)
    # print(window_contour_cropped)

    # try:
    #     # from imageio import imread
    #     # test_img = imread(window_contour_cropped) 
    #     # plt.imsave('window_contour_cropped.png', test_img)
    #     plt.imsave('window_contour_cropped.png', window_contour_cropped)

    # except:
    #     pass
    # #####


    seg_img = visualize_segmentation(pr_main_contours, inp, n_classes=n_classes,
                                     colors=colors, overlay_img=overlay_img,
                                     show_legends=show_legends,
                                     class_names=class_names,
                                     prediction_width=prediction_width,
                                     prediction_height=prediction_height)
    # seg_img returns per-pixel (B,G,R) output

    #####   
    inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)

    fig = plot_orig_and_overlay(inp, seg_img)
    #####

    if out_fname is not None:
        # cv2.imwrite('./predictions/cropped_window_contour.jpg', window_contour_cropped)
        fig.savefig(out_fname, dpi=300)
    else:
        #cv2.imwrite(f'./predictions/{filename}__pred.png', fig)
        fig.savefig(f'./predictions/{filename}__pred.png', dpi=300)

    return pr


def predict_multiple(model=None, inps=None, inp_dir=None, out_dir=None,
                     checkpoints_path=None, overlay_img=False,
                     class_names=None, show_legends=False, colors=class_colors,
                     prediction_width=None, prediction_height=None):

    if model is None and (checkpoints_path is not None):
        model = model_from_checkpoint_path(checkpoints_path)

    if inps is None and (inp_dir is not None):
        inps = glob.glob(os.path.join(inp_dir, "*.jpg")) + glob.glob(
            os.path.join(inp_dir, "*.png")) + \
            glob.glob(os.path.join(inp_dir, "*.jpeg"))
        inps = sorted(inps)

    assert type(inps) is list

    all_prs = []

    for i, inp in enumerate(tqdm(inps)):
        if out_dir is None:
            out_fname = None
        else:
            if isinstance(inp, six.string_types):
                out_fname = os.path.join(out_dir, os.path.basename(inp))
            else:
                out_fname = os.path.join(out_dir, str(i) + ".jpg")

        pr = predict(model, inp, out_fname,
                     overlay_img=overlay_img, class_names=class_names,
                     show_legends=show_legends, colors=colors,
                     prediction_width=prediction_width,
                     prediction_height=prediction_height)

        all_prs.append(pr)


    return all_prs


def set_video(inp, video_name):
    cap = cv2.VideoCapture(inp)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (video_width, video_height)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video = cv2.VideoWriter(video_name, fourcc, fps, size)
    return cap, video, fps


def predict_video(model=None, inp=None, output=None,
                  checkpoints_path=None, display=False, overlay_img=False,
                  class_names=None, show_legends=False, colors=class_colors,
                  prediction_width=None, prediction_height=None):

    if model is None and (checkpoints_path is not None):
        model = model_from_checkpoint_path(checkpoints_path)
    n_classes = model.n_classes

    cap, video, fps = set_video(inp, output)
    while(cap.isOpened()):
        prev_time = time()
        ret, frame = cap.read()
        if frame is not None:
            pr = predict(model=model, inp=frame)
            fused_img = visualize_segmentation(
                pr, frame, n_classes=n_classes,
                colors=colors,
                overlay_img=overlay_img,
                show_legends=show_legends,
                class_names=class_names,
                prediction_width=prediction_width,
                prediction_height=prediction_height
                )
        else:
            break
        print("FPS: {}".format(1/(time() - prev_time)))
        if output is not None:
            video.write(fused_img)
        if display:
            cv2.imshow('Frame masked', fused_img)
            if cv2.waitKey(fps) & 0xFF == ord('q'):
                break
    cap.release()
    if output is not None:
        video.release()
    cv2.destroyAllWindows()


def evaluate(model=None, inp_images=None, annotations=None,
             inp_images_dir=None, annotations_dir=None, checkpoints_path=None):

    if model is None:
        assert (checkpoints_path is not None),\
                "Please provide the model or the checkpoints_path"
        model = model_from_checkpoint_path(checkpoints_path)

    if inp_images is None:
        assert (inp_images_dir is not None),\
                "Please provide inp_images or inp_images_dir"
        assert (annotations_dir is not None),\
            "Please provide inp_images or inp_images_dir"

        paths = get_pairs_from_paths(inp_images_dir, annotations_dir)
        paths = list(zip(*paths))
        inp_images = list(paths[0])
        annotations = list(paths[1])

    assert type(inp_images) is list
    assert type(annotations) is list

    tp = np.zeros(model.n_classes)
    fp = np.zeros(model.n_classes)
    fn = np.zeros(model.n_classes)
    n_pixels = np.zeros(model.n_classes)

    for inp, ann in tqdm(zip(inp_images, annotations)):
        pr = predict(model, inp)
        gt = get_segmentation_array(ann, model.n_classes,
                                    model.output_width, model.output_height,
                                    no_reshape=True)
        gt = gt.argmax(-1)
        pr = pr.flatten()
        gt = gt.flatten()

        for cl_i in range(model.n_classes):

            tp[cl_i] += np.sum((pr == cl_i) * (gt == cl_i))
            fp[cl_i] += np.sum((pr == cl_i) * ((gt != cl_i)))
            fn[cl_i] += np.sum((pr != cl_i) * ((gt == cl_i)))
            n_pixels[cl_i] += np.sum(gt == cl_i)

    cl_wise_score = tp / (tp + fp + fn + 0.000000000001)
    n_pixels_norm = n_pixels / np.sum(n_pixels)
    frequency_weighted_IU = np.sum(cl_wise_score*n_pixels_norm)
    mean_IU = np.mean(cl_wise_score)

    return {
        "frequency_weighted_IU": frequency_weighted_IU,
        "mean_IU": mean_IU,
        "class_wise_IU": cl_wise_score
    }
