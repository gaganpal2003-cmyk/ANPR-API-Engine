from PyQt5.QtCore import *
from .objectdetection import ObjectDet


class FaceRecog(QObject):
    def __init__(self):
        super(FaceRecog, self).__init__()

    def getFinalLabelImage(self, model, names, dt, im, im0s, object_list, thres_h, lineCoordinate):
        img = ObjectDet.extract_faces(model, names, dt, im, im0s, object_list, thres_h, lineCoordinate)
        return img

    def ExtractDeviceModelDataset(self, source, myModel, coco):
        device_and_model = ObjectDet.get_model(myModel, coco)
        device = device_and_model[0]
        model = device_and_model[1]

        stride, names, pt = model.stride, model.names, model.pt
        imgsz = ObjectDet.find_img_size(stride)  # check image size
        dataset = ObjectDet.get_dataset(source, imgsz, stride, pt)
        dt = ObjectDet.get_profile(device)
        return device, model, dataset, dt

    def ExtractImageSize(self, stride):
        imgSize = ObjectDet.find_img_size(stride)
        return imgSize

    def ExtractOcrDataset(self, temp_img_path, imgsz, stride, pt):
        dSet = ObjectDet.get_ocr_dataset(temp_img_path, imgsz, stride, pt)
        return dSet


    def getNonMaxSuppresion(self, ocr_pred):
        ocrPred = ObjectDet.get_non_max_suprr(ocr_pred)
        return ocrPred

    def getOcrAnnotator(self, ocr_img0, names):
        annotator = ObjectDet.get_ocr_annotator(ocr_img0,names)
        return annotator

    def getOCRScaleBoxes(self, ocr_im, det, ocr_im0):
        return ObjectDet.get_ocr_scale_boxes(ocr_im, det, ocr_im0)

    def getOcrColors(self, cls):
        return ObjectDet.get_ocr_color(cls)
