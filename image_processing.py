from PIL import Image, ExifTags, ImageOps
from sahi.prediction import ObjectPrediction
from sahi.utils.cv import visualize_object_predictions, read_image
from ultralyticsplus import YOLO
import numpy as np
import tensorflow as tf
from huggingface_hub import from_pretrained_keras
import cv2
import clustering

model_segmentation = from_pretrained_keras("keras-io/deeplabv3p-resnet50")
model_yolo = YOLO('kadirnar/yolov8m-v8.0')


def yolov8_inference(image, image_size=512, conf_threshold=0.50, iou_threshold=0.45):
    """
    YOLOv8 inference function
    Args:
        image: Input image
        model_path: Path to the model
        image_size: Image size
        conf_threshold: Confidence threshold
        iou_threshold: IOU threshold
    Returns:
        Rendered image
    """

    model_yolo.conf = conf_threshold
    model_yolo.iou = iou_threshold
    results = model_yolo.predict(image, imgsz=image_size, return_outputs=True)
    object_prediction_list = []
    predictions = []
    for _, image_results in enumerate(results):
        if len(image_results) != 0:
            image_predictions_in_xyxy_format = image_results['det']
            for pred in image_predictions_in_xyxy_format:
                if int(pred[5]) == 0:
                    x1, y1, x2, y2 = (
                        int(pred[0]),
                        int(pred[1]),
                        int(pred[2]),
                        int(pred[3]),
                    )
                    bbox = [x1, y1, x2, y2]
                    predictions.append(bbox)
                    score = pred[4]
                    category_name = model_yolo.model.names[int(pred[5])]
                    category_id = pred[5]
                    object_prediction = ObjectPrediction(
                        bbox=bbox,
                        category_id=int(category_id),
                        score=score,
                        category_name=category_name,
                    )
                    object_prediction_list.append(object_prediction)

    #image = read_image(image)
    #output_image = visualize_object_predictions(image=image, object_prediction_list=object_prediction_list)

    return predictions

colormap = np.array([[0,0,0], [31,119,180], [44,160,44], [44, 127, 125], [52, 225, 143],
                     [217, 222, 163], [254, 128, 37], [130, 162, 128], [121, 7, 166], [136, 183, 248],
                     [85, 1, 76], [22, 23, 62], [159, 50, 15], [101, 93, 152], [252, 229, 92],
                     [167, 173, 17], [218, 252, 252], [238, 126, 197], [116, 157, 140], [214, 220, 252]], dtype=np.uint8)

img_size = 512

def read_image(image):
    image = tf.convert_to_tensor(image)
    image.set_shape([None, None, 3])
    image = tf.image.resize(images=image, size=[img_size, img_size])
    image = image / 127.5 - 1
    return image

def infer(model, image_tensor):
    predictions = model.predict(np.expand_dims((image_tensor), axis=0))
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=2)
    return predictions

def decode_segmentation_masks(mask, colormap, n_classes):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    # helmet => 1, hair => 2, neck => 10, face => 13
    for l in [1, 2, 10, 13]:
        idx = mask == l
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]
        rgb = np.stack([r, g, b], axis=2)
    return rgb

def get_overlay(image, colored_mask):
    image = tf.keras.preprocessing.image.array_to_img(image)
    image = np.array(image).astype(np.uint8)
    overlay = cv2.addWeighted(image, 0.35, colored_mask, 0.65, 0)
    return overlay

def segmentation(input_image):
    image_tensor = read_image(input_image)
    prediction_mask = infer(image_tensor=image_tensor, model=model_segmentation)
    prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, 20)
    overlay = get_overlay(image_tensor, prediction_colormap)
    return (overlay, prediction_colormap)


def exif_transpose(img):
    if not img:
        return img

    exif_orientation_tag = 274

    # Check for EXIF data (only present on some files)
    if hasattr(img, "_getexif") and isinstance(img._getexif(), dict) and exif_orientation_tag in img._getexif():
        exif_data = img._getexif()
        orientation = exif_data[exif_orientation_tag]

        # Handle EXIF Orientation
        if orientation == 1:
            # Normal image - nothing to do!
            pass
        elif orientation == 2:
            # Mirrored left to right
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            # Rotated 180 degrees
            img = img.rotate(180)
        elif orientation == 4:
            # Mirrored top to bottom
            img = img.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 5:
            # Mirrored along top-left diagonal
            img = img.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            # Rotated 90 degrees
            img = img.rotate(-90, expand=True)
        elif orientation == 7:
            # Mirrored along top-right diagonal
            img = img.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            # Rotated 270 degrees
            img = img.rotate(90, expand=True)

    return img


def check_orientation(filepath):
    try:
        image = Image.open(filepath)
        t_image = exif_transpose(image)
        t_image.save(filepath)
    except:
        pass


def process_image(image, background):
    predictions = []
    check_orientation(image)
    try:
        predictions = yolov8_inference(image=image)
    except:
        print("YOLOv8 inference failed")
        return None

    if len(predictions) == 0:
        return None

    max_area = 0
    max_pred = None
    for pred in predictions:
        x1, y1, x2, y2 = pred
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            max_pred = pred

    work_image = Image.open(image)
    work_image = work_image.crop(max_pred)
    image_size = 512

    x1, y1, x2, y2 = 0, 0, work_image.size[0], work_image.size[1]
    try:

        image_tensor = read_image(work_image.convert('RGB'))
        prediction_mask = infer(image_tensor=image_tensor, model=model_segmentation)
        prediction_mask[(prediction_mask != 1) & (prediction_mask != 2) & (prediction_mask != 10) & (prediction_mask != 13)] = 0
        select_mask = np.where(clustering.filter_clusters(prediction_mask))
        x1, y1, x2, y2 = np.min(select_mask[1]), np.min(select_mask[0]), np.max(select_mask[1]), np.max(select_mask[0])
    except:
        print("Segmentation inference failed")

    x1, y1, x2, y2 = (x1/512)*work_image.width, (y1/512)*work_image.height, (x2/512)*work_image.width, (y2/512)*work_image.height

    final_image = work_image.crop((x1, y1, x2, y2))
    w, h = final_image.size
    scale_percent = image_size / max([w, h])

    width = int(w * scale_percent)
    height = int(h * scale_percent)
    dim = (width, height)
    final_image = final_image.resize(dim)

    proc_img = Image.open(background)
    proc_img.paste(final_image, (int((image_size - width) / 2), int((image_size - height) / 2)))

    return proc_img

BACKGROUNDS_FOLDER = 'backgrounds'
wi = 'bucket/sergio.pardo@oracle.com/2E372EE0-6710-4910-873F-03D986D45192.jpeg'


check_orientation(wi)

'''
result = process_image('bucket/sergio.pardo@oracle.com/2E372EE0-6710-4910-873F-03D986D45192.jpeg', BACKGROUNDS_FOLDER + '/bck' + str(1 % 20) + '.png')
image_file_name = 'result.png'
if result is not None:
    with open(image_file_name, 'wb') as f:
        result.save(f)
'''