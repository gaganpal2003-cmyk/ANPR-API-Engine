import os
import sys
import json
import base64
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union

import cv2
import numpy as np
import torch

# PyTorch 2.6+ compatibility: defaults weights_only=True which breaks custom pickled YOLO checkpoints
try:
    _orig_torch_load = torch.load
    def _compat_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            try:
                return _orig_torch_load(*args, weights_only=False, **kwargs)
            except TypeError:
                return _orig_torch_load(*args, **kwargs)
        return _orig_torch_load(*args, **kwargs)
    torch.load = _compat_torch_load
except Exception:
    pass

# Ensure YOLO internal imports resolve
BASE_DIR = Path(__file__).resolve().parent.parent
YOLO_DIR = BASE_DIR / "Modules" / "ObjectDetectionModel" / "objectdetection"
if str(YOLO_DIR) not in sys.path:
    sys.path.insert(0, str(YOLO_DIR))

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes, check_img_size

# Indian State Codes Mapping
INDIAN_STATES = {
    "AN": "Andaman and Nicobar Islands",
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BH": "Bharat Series",
    "BR": "Bihar",
    "CH": "Chandigarh",
    "CG": "Chhattisgarh",
    "DD": "Daman and Diu",
    "DL": "Delhi",
    "DN": "Dadra and Nagar Haveli",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HR": "Haryana",
    "HP": "Himachal Pradesh",
    "JH": "Jharkhand",
    "JK": "Jammu and Kashmir",
    "KA": "Karnataka",
    "KL": "Kerala",
    "LA": "Ladakh",
    "LD": "Lakshadweep",
    "MP": "Madhya Pradesh",
    "MH": "Maharashtra",
    "MN": "Manipur",
    "ML": "Meghalaya",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OD": "Odisha",
    "OR": "Odisha",
    "PB": "Punjab",
    "PY": "Puducherry",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TR": "Tripura",
    "TS": "Telangana",
    "UK": "Uttarakhand",
    "UA": "Uttarakhand",
    "UP": "Uttar Pradesh",
    "WB": "West Bengal"
}


class ANPRInferenceEngine:
    _instance = None

    @classmethod
    def get_instance(cls, **kwargs):
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    def __init__(
        self,
        plate_model_path: Optional[str] = None,
        ocr_model_path: Optional[str] = None,
        coco_path: Optional[str] = None,
        device_str: Optional[str] = None,
        conf_threshold: float = 0.15,
        ocr_conf_threshold: float = 0.15,
    ):
        self.base_dir = BASE_DIR
        self.conf_threshold = conf_threshold
        self.ocr_conf_threshold = ocr_conf_threshold

        # Paths
        self.plate_model_path = plate_model_path or str(self.base_dir / "Detection" / "model" / "best_plate.pt")
        self.ocr_model_path = ocr_model_path or str(self.base_dir / "Detection" / "model" / "best_ocr.pt")
        self.coco_path = coco_path or str(self.base_dir / "Detection" / "data" / "coco.yaml")

        # Select Compute Device
        if device_str:
            self.device = torch.device(device_str)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[ANPR Engine] Initializing models on device: {self.device}")

        # Load RTO database
        self.rto_dict = self._load_rto_codes()

        # Load Models
        self.plate_model = None
        self.ocr_model = None
        self._load_models()

    def _load_rto_codes(self) -> Dict[str, List[str]]:
        rto_file = self.base_dir / "config" / "RTO_CODES.txt"
        if rto_file.exists():
            try:
                with open(rto_file, "r", encoding="utf-8") as f:
                    return json.loads(f.read())
            except Exception as e:
                print(f"[ANPR Engine] Warning: Failed to load RTO_CODES.txt: {e}")
        return {}

    def _load_models(self):
        # 1. Load Plate Model
        print(f"[ANPR Engine] Loading Plate Detection Model from {self.plate_model_path}...")
        self.plate_model = DetectMultiBackend(
            self.plate_model_path,
            device=self.device,
            dnn=False,
            data=self.coco_path,
            fp16=False
        )
        self.plate_stride = self.plate_model.stride
        self.plate_names = self.plate_model.names
        self.plate_pt = self.plate_model.pt
        self.plate_imgsz = check_img_size((640, 640), s=self.plate_stride)

        # 2. Load OCR Model
        print(f"[ANPR Engine] Loading OCR Model from {self.ocr_model_path}...")
        self.ocr_model = DetectMultiBackend(
            self.ocr_model_path,
            device=self.device,
            dnn=False,
            fp16=False
        )
        self.ocr_stride = self.ocr_model.stride
        self.ocr_names = self.ocr_model.names
        self.ocr_pt = self.ocr_model.pt
        self.ocr_imgsz = check_img_size((640, 640), s=self.ocr_stride)
        print("[ANPR Engine] Models successfully loaded and ready for inference!")

    def _prepare_tensor(self, img_bgr: np.ndarray, imgsz: int, stride: int, pt: bool) -> Tuple[torch.Tensor, np.ndarray]:
        """Letterbox and convert OpenCV BGR to PyTorch inference tensor."""
        im = letterbox(img_bgr, imgsz, stride=stride, auto=pt)[0]
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        tensor = torch.from_numpy(im).to(self.device).float() / 255.0
        if len(tensor.shape) == 3:
            tensor = tensor[None]  # Batch dimension
        return tensor, im

    def detect_plates(
        self,
        image_bgr: np.ndarray,
        conf_thresh: Optional[float] = None
    ) -> List[Dict]:
        """
        Detects license plates (MP, SP) in an image.
        Returns list of dicts: {'type': 'MP'|'SP', 'conf': float, 'box': [x1, y1, x2, y2]}
        """
        thresh = conf_thresh if conf_thresh is not None else self.conf_threshold
        tensor, im_prep = self._prepare_tensor(image_bgr, self.plate_imgsz, self.plate_stride, self.plate_pt)

        with torch.no_grad():
            pred = self.plate_model(tensor)
            pred = non_max_suppression(pred, thresh, 0.45, None, False, max_det=50)

        plates = []
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(tensor.shape[2:], det[:, :4], image_bgr.shape).round()
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = [int(max(0, v)) for v in xyxy]
                    # Clamp to image dimensions
                    x1 = min(x1, image_bgr.shape[1] - 1)
                    x2 = min(x2, image_bgr.shape[1])
                    y1 = min(y1, image_bgr.shape[0] - 1)
                    y2 = min(y2, image_bgr.shape[0])

                    if x2 > x1 and y2 > y1:
                        cls_idx = int(cls)
                        cls_name = self.plate_names[cls_idx] if isinstance(self.plate_names, dict) else self.plate_names[cls_idx]
                        plates.append({
                            "type": str(cls_name),
                            "conf": float(conf),
                            "box": [x1, y1, x2, y2]
                        })

        # Sort plates by highest confidence first
        plates.sort(key=lambda p: p["conf"], reverse=True)
        return plates

    def run_ocr_on_plate(
        self,
        plate_crop_bgr: np.ndarray,
        plate_type: str = "SP",
        ocr_conf_thresh: Optional[float] = None
    ) -> Dict:
        """
        Runs OCR on a cropped plate image, sorts characters, and reconstructs number plate string.
        """
        thresh = ocr_conf_thresh if ocr_conf_thresh is not None else self.ocr_conf_threshold
        h, w = plate_crop_bgr.shape[:2]
        if h < 10 or w < 10:
            return {"plate_number": "", "confidence": 0.0, "characters": [], "is_valid_rto": False}

        tensor, im_prep = self._prepare_tensor(plate_crop_bgr, self.ocr_imgsz, self.ocr_stride, self.ocr_pt)

        with torch.no_grad():
            pred = self.ocr_model(tensor)
            pred = non_max_suppression(pred, thresh, 0.45, None, False, max_det=100)

        chars = []
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(tensor.shape[2:], det[:, :4], plate_crop_bgr.shape).round()
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = [int(v) for v in xyxy]
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    char_str = str(self.ocr_names[int(cls)])
                    chars.append({
                        "char": char_str,
                        "box": [x1, y1, x2, y2],
                        "center": (cx, cy),
                        "conf": float(conf)
                    })

        if not chars:
            return {
                "plate_number": "",
                "confidence": 0.0,
                "characters": [],
                "is_valid_rto": False,
                "state_code": None,
                "state_name": None
            }

        # Sorting logic: Multi-line (MP) vs Single-line (SP)
        plate_number = self._sort_and_build_plate_string(chars, plate_type, h)

        # Average confidence of detected characters
        avg_conf = float(np.mean([c["conf"] for c in chars])) if chars else 0.0

        # State code and RTO check
        state_code, state_name, is_valid_rto = self._validate_rto(plate_number)

        return {
            "plate_number": plate_number,
            "confidence": round(avg_conf, 4),
            "characters": chars,
            "state_code": state_code,
            "state_name": state_name,
            "is_valid_rto": is_valid_rto
        }

    def _sort_and_build_plate_string(self, chars: List[Dict], plate_type: str, plate_height: int) -> str:
        """
        Sort characters according to plate layout:
        For MP (Multi-line plate): splits characters into Top and Bottom rows.
        For SP (Single-line plate): sorts left-to-right.
        """
        if not chars:
            return ""

        # Check if plate layout has 2 distinct rows
        y_centers = [c["center"][1] for c in chars]
        min_y, max_y = min(y_centers), max(y_centers)
        vertical_spread = max_y - min_y

        is_multi_line = (plate_type == "MP") or (vertical_spread > (plate_height * 0.25))

        if is_multi_line and len(chars) >= 4:
            mid_y = (min_y + max_y) / 2.0
            row1 = [c for c in chars if c["center"][1] <= mid_y]
            row2 = [c for c in chars if c["center"][1] > mid_y]

            # Sort each row left to right
            row1.sort(key=lambda c: c["center"][0])
            row2.sort(key=lambda c: c["center"][0])

            text_row1 = "".join(c["char"] for c in row1)
            text_row2 = "".join(c["char"] for c in row2)
            return (text_row1 + text_row2).upper()
        else:
            # Single-line: sort by X
            chars.sort(key=lambda c: c["center"][0])
            return "".join(c["char"] for c in chars).upper()

    def _validate_rto(self, plate_str: str) -> Tuple[Optional[str], Optional[str], bool]:
        """Extract Indian state code and check validity against RTO dataset."""
        cleaned = "".join(ch for ch in plate_str if ch.isalnum()).upper()
        if len(cleaned) < 2:
            return None, None, False

        state_cand = cleaned[:2]
        state_name = INDIAN_STATES.get(state_cand, None)

        is_valid = False
        if state_name is not None:
            if self.rto_dict and state_cand in self.rto_dict:
                # Check 4-char prefix like 'CG04', 'AP39', 'GJ01'
                if len(cleaned) >= 4:
                    prefix_4 = cleaned[:4]
                    if prefix_4 in self.rto_dict[state_cand]:
                        is_valid = True
                else:
                    is_valid = True
            else:
                is_valid = True

        return state_cand if state_name else None, state_name, is_valid

    def process_image(
        self,
        image_bgr: np.ndarray,
        conf_thresh: Optional[float] = None,
        ocr_conf_thresh: Optional[float] = None,
        return_annotated: bool = False,
        return_crop: bool = False
    ) -> Dict:
        """
        Complete ANPR pipeline on an image:
        1. Detect license plate(s)
        2. Fallback: If 0 plates found, attempt OCR on entire image
        3. Crop and recognize text on each plate
        4. Optionally annotate and encode crops to Base64
        """
        t0 = time.time()
        plates_detected = self.detect_plates(image_bgr, conf_thresh=conf_thresh)
        results = []

        annotated_image = image_bgr.copy() if return_annotated else None

        if len(plates_detected) == 0:
            # Fallback: Image might already be a tightly cropped plate!
            ocr_res = self.run_ocr_on_plate(image_bgr, plate_type="SP", ocr_conf_thresh=ocr_conf_thresh)
            if len(ocr_res["characters"]) >= 3:
                h, w = image_bgr.shape[:2]
                plate_entry = {
                    "plate_number": ocr_res["plate_number"],
                    "plate_type": "SP (Direct)",
                    "confidence": ocr_res["confidence"],
                    "plate_box": [0, 0, w, h],
                    "state_code": ocr_res["state_code"],
                    "state_name": ocr_res["state_name"],
                    "is_valid_rto": ocr_res["is_valid_rto"],
                    "characters": ocr_res["characters"],
                    "cropped_image_base64": self._encode_base64(image_bgr) if return_crop else None
                }
                results.append(plate_entry)
                if return_annotated and annotated_image is not None:
                    self._draw_box_and_label(annotated_image, [0, 0, w, h], ocr_res["plate_number"], ocr_res["confidence"])
        else:
            h_img, w_img = image_bgr.shape[:2]
            for p in plates_detected:
                x1, y1, x2, y2 = p["box"]
                # Add small margin (5%) for OCR margin
                margin_x = int((x2 - x1) * 0.03)
                margin_y = int((y2 - y1) * 0.05)
                cx1 = max(0, x1 - margin_x)
                cy1 = max(0, y1 - margin_y)
                cx2 = min(w_img, x2 + margin_x)
                cy2 = min(h_img, y2 + margin_y)

                plate_crop = image_bgr[cy1:cy2, cx1:cx2]
                ocr_res = self.run_ocr_on_plate(plate_crop, plate_type=p["type"], ocr_conf_thresh=ocr_conf_thresh)

                plate_entry = {
                    "plate_number": ocr_res["plate_number"],
                    "plate_type": p["type"],
                    "confidence": p["conf"],
                    "plate_box": [x1, y1, x2, y2],
                    "state_code": ocr_res["state_code"],
                    "state_name": ocr_res["state_name"],
                    "is_valid_rto": ocr_res["is_valid_rto"],
                    "characters": ocr_res["characters"],
                    "cropped_image_base64": self._encode_base64(plate_crop) if return_crop else None
                }
                results.append(plate_entry)

                if return_annotated and annotated_image is not None:
                    label = f"{ocr_res['plate_number']} ({p['type']})" if ocr_res['plate_number'] else p['type']
                    self._draw_box_and_label(annotated_image, [x1, y1, x2, y2], label, p["conf"])

        latency_ms = round((time.time() - t0) * 1000, 2)
        primary_number = results[0]["plate_number"] if results and results[0]["plate_number"] else None

        return {
            "success": True,
            "message": f"Detected {len(results)} plate(s)" if results else "No license plates detected",
            "total_plates": len(results),
            "primary_plate_number": primary_number,
            "plates": results,
            "annotated_image_base64": self._encode_base64(annotated_image) if return_annotated and annotated_image is not None else None,
            "processing_time_ms": latency_ms
        }

    def _draw_box_and_label(self, img: np.ndarray, box: List[int], text: str, conf: float):
        x1, y1, x2, y2 = box
        color = (0, 215, 255)  # Bright Gold / Orange-Yellow
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)

        display_text = f"{text} [{conf:.2f}]"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(display_text, font, scale, thickness)

        label_y1 = max(0, y1 - th - 10)
        label_y2 = y1
        cv2.rectangle(img, (x1, label_y1), (x1 + tw + 10, label_y2), color, -1)
        cv2.putText(img, display_text, (x1 + 5, label_y2 - 5), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

    def _encode_base64(self, img: np.ndarray) -> Optional[str]:
        if img is None:
            return None
        success, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if success:
            return base64.b64encode(buf).decode("utf-8")
        return None
