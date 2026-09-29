import os
import sys
from pathlib import Path
import torch

# self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
# ---------------------------
# Dynamically set ROOT path
# ---------------------------
FILE = Path(__file__).resolve()                # Absolute path of this file
ROOT = FILE.parents[0]                         # YoloLib directory (where models folder exists)

# Add ROOT to sys.path if not already present
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))             # insert at front to prioritize

# Optional: relative path for clarity (not necessary, but nice)
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# ---------------------------
# Import your local models
# ---------------------------
import models
import models.yolo


from ultralytics.utils.plotting import Annotator, colors, save_one_box
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages, LoadImagesOCR
# from utils.dataloaders import LoadImages
from utils.general import (
    Profile,
    check_img_size,
    cv2,
    non_max_suppression,
    scale_boxes,

)
from utils.torch_utils import select_device
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
from collections import deque

deque = deque(maxlen=1)
is_drawing = False
org_frame = None

parameters_dict = {
    "imgsz": (640, 640),  # inference size (height, width)
    "iou_thres": 0.45,  # NMS IOU threshold
    "max_det": 1000,  # maximum detections per image
    "device": None,  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    "view_img": False,  # show results
    "save_txt": False,  # save results to *.txt
    "save_csv": False,  # save results in CSV format
    "save_conf": False,  # save confidences in --save-txt labels
    "save_crop": False,  # save cropped prediction boxes
    "nosave": False,  # do not save images/videos
    "classes": None,  # filter by class: --class 0, or --class 0 2 3
    "agnostic_nms": False,  # class-agnostic NMS
    "augment": False,  # augmented inference
    "visualize": False,  # visualize features
    "update": False,  # update all models
    "project": ROOT / "runs/detect",  # save results to project/name
    "name": "exp",  # save results to project/name
    "exist_ok": False,  # existing project/name ok, do not increment
    "line_thickness": 3,  # bounding box thickness (pixels)
    "hide_labels": False,  # hide labels
    "hide_conf": False,  # hide confidences
    "half": False,  # use FP16 half-precision inference
    "dnn": False,  # use OpenCV DNN for ONNX inference
    "vid_stride": 1,  # video frame-rate stride
}


def get_model(myModel, coco):
    device = select_device("")
    model_ = DetectMultiBackend(myModel, device=device, dnn=False,
                                data=coco, fp16=False)
    # model_ = DetectMultiBackend(parameters_dict.get("weights"), device=device, dnn=False,
    #                             data=parameters_dict.get("data"), fp16=False)

    return device, model_


def find_img_size(stride):
    return check_img_size((640, 640), s=stride)


def get_dataset(source, imgsz, stride, pt):
    return LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=5)


def get_non_max_suprr(ocr_pred):
    ocr_pred = non_max_suppression(ocr_pred, 0.20, 0.45,
                                   None, False,
                                   max_det=1000)
    return ocr_pred


def get_ocr_dataset(source, imgsz, stride, pt):
    return LoadImagesOCR(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=5)


def get_ocr_annotator(ocr_img0, names):
    annotator = Annotator(ocr_img0, line_width=3, example=str(names))
    return annotator

def get_ocr_scale_boxes(ocr_im, det, ocr_im0):
    return scale_boxes(ocr_im.shape[2:], det[:, :4], ocr_im0.shape).round()

def get_ocr_color(cls):
    return colors(int(cls), True)

def get_profile(device):
    return Profile(device=device), Profile(device=device), Profile(device=device)


def is_inside_drawing_box(obj_box, drawing_box):
    """
    Check if the center of an object's bounding box lies inside the drawing box.
    obj_box: (x1, y1, x2, y2)
    drawing_box: (rx, ry, rw, rh) where (rx, ry) is top-left, rw=width, rh=height
    """
    (x1, y1, x2, y2) = obj_box
    (rx, ry, rw, rh) = drawing_box

    # Find center of obj_box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # Return True if center lies inside drawing_box, else False
    return (rx <= cx <= rx + rw) and (ry <= cy <= ry + rh)
def extract_faces(model, names, dt, im, im0s, object_list, thres_h, line_coordinate):
    global org_frame
    with dt[0]:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
    with dt[1]:
        pred = model(im, augment=parameters_dict.get("augment"), visualize=False)
    with dt[2]:
        pred = non_max_suppression(pred, thres_h, parameters_dict.get("iou_thres"),
                                   parameters_dict.get("classes"), parameters_dict.get("agnostic_nms"),
                                   max_det=parameters_dict.get("max_det"))
        # pred = non_max_suppression(pred, parameters_dict.get("conf_thres"), parameters_dict.get("iou_thres"),
        #                            parameters_dict.get("classes"), parameters_dict.get("agnostic_nms"),
        #                            max_det=parameters_dict.get("max_det"))
    # for i, det in enumerate(pred):  # per image
    #     im0 = im0s.copy()
    #     annotator = Annotator(im0, line_width=parameters_dict.get("line_thickness"), example=str(names))
    #     current_coordinate = 0
    #     cropped_img = 0
    #     if len(det):
    #         det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
    #         for *xyxy, conf, cls in reversed(det):
    #             x1, y1, x2, y2 = map(int, xyxy)
    #             cropped_img = im0s[y1:y2, x1:x2]
    #             if names[int(cls)] in object_list:
    #                 bbox = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
    #                 current_coordinate = bbox
    #                 annotator.box_label(bbox, f"{names[int(cls)]} {conf:.2f}", color=colors(int(cls), True))
    #                 return annotator.result(), f"{names[int(cls)]}", True, current_coordinate, cropped_img
    #             else:
    #                 return annotator.result(), None, False, current_coordinate, cropped_img
    #     else:
    #         return annotator.result(), None, False, current_coordinate, cropped_img
    for i, det in enumerate(pred):  # per image
        im0 = im0s.copy()
        annotator = Annotator(im0, line_width=parameters_dict.get("line_thickness"), example=str(names))
        current_coordinate = 0
        cropped_img = 0

        if line_coordinate:
            x, y = line_coordinate[0][0], line_coordinate[0][1]
            w, h = line_coordinate[1][0], line_coordinate[1][1]
            cv2.rectangle(im0, (x, y), (x + w, y + h), color=colors(2, True), thickness=3, lineType=cv2.LINE_AA)

        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = map(int, xyxy)
                cropped_img = im0s[y1:y2, x1:x2]
                ##
                c = int(cls)
                if line_coordinate:
                    x, y = line_coordinate[0][0], line_coordinate[0][1]
                    w, h = line_coordinate[1][0], line_coordinate[1][1]
                    drawing_box = (x, y, w, h)
                    objbox = (x1, y1, x2, y2)
                    sts = is_inside_drawing_box(objbox, drawing_box)
                    if sts:
                        annotator.box_label([x1, y1, x2, y2], f"{names[c]} {conf:.2f}",
                                            color=colors(c, True))
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        # person.append((cx, cy))
                        cv2.circle(im0, (cx, cy), 5, (0, 255, 0), -1)
                        current_coordinate = sts
                        label = None if parameters_dict.get("hide_labels") else (
                            names[c] if parameters_dict.get("hide_conf") else f"{names[c]} {conf:.2f}")
                        # object_name.append(label.split()[0])
                        return annotator.result(), f"{names[int(cls)]}", True, current_coordinate, cropped_img
                    else:
                        current_coordinate = sts
                        return annotator.result(), None, False, current_coordinate, cropped_img
                # ##
                # if names[int(cls)] in object_list:
                #     bbox = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                #     current_coordinate = bbox
                #     annotator.box_label(bbox, f"{names[int(cls)]} {conf:.2f}", color=colors(int(cls), True))
                #     return annotator.result(), f"{names[int(cls)]}", True, current_coordinate, cropped_img
                # else:
                #     return annotator.result(), None, False, current_coordinate, cropped_img
        else:
            return annotator.result(), None, False, current_coordinate, cropped_img
