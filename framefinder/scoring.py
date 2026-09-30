"""Local, per-frame scoring: faces, face sharpness, eyes/smile, exposure, and a duplicate hash."""
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

MODELS = Path(__file__).resolve().parent.parent / "models"


@dataclass
class Thresholds:
    min_face_px: int = 80            # smallest face height (full-res pixels) that counts as "clear"
    min_face_frac: float = 0.035     # ...and as a fraction of the frame's short side
    det_confidence: float = 0.6
    min_sharpness: float = 15.0      # contrast-normalized Laplacian variance (sharp faces ~30+, motion blur <10)
    max_blink: float = 0.55          # MediaPipe eyeBlink blendshape; above this = eyes closed
    min_face_luma: float = 45.0      # mean face brightness (0-255)
    max_face_luma: float = 225.0
    max_clipped: float = 0.20        # fraction of the frame that is pure white
    min_frame_luma: float = 22.0


@dataclass
class FrameScore:
    file: str
    t: float
    width: int = 0
    height: int = 0
    n_faces: int = 0               # all detected faces
    n_clear: int = 0               # faces big enough to use
    face_frac: float = 0.0         # largest face height / frame height
    face_sharpness: float = 0.0    # area-weighted over main faces
    min_sharpness: float = 0.0     # worst main face
    smile: float = -1.0            # 0-1, -1 if unknown
    eyes_open: float = -1.0        # 0-1, -1 if unknown
    face_luma: float = 0.0
    frame_luma: float = 0.0
    clipped: float = 0.0
    crushed: float = 0.0
    local_score: float = 0.0
    fail: str = ""                 # first filter that rejected this frame, "" if it passed
    dhash: str = ""
    faces: list = field(default_factory=list)   # [x, y, w, h] boxes of main faces (full res)

    def row(self) -> dict:
        d = asdict(self)
        d["faces"] = ";".join(",".join(str(int(v)) for v in b) for b in self.faces)
        return d


def _nms(boxes, scores, iou=0.3):
    if not boxes:
        return []
    b = np.array(boxes, float)
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    area = b[:, 2] * b[:, 3]
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        # also suppress boxes mostly contained in a kept box (tile-edge fragments)
        ov = np.maximum(inter / (area[i] + area[order[1:]] - inter), inter / np.minimum(area[i], area[order[1:]]))
        order = order[1:][ov < iou]
    return keep


def dhash(gray: np.ndarray, size: int = 16) -> str:
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return "".join("1" if b else "0" for b in bits)


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def sharpness(gray_face: np.ndarray) -> float:
    """Laplacian variance on the inner face.

    The face is resized to 128px (so big and small faces compare fairly), contrast-normalized
    (so dim or tone-mapped faces aren't mistaken for blurry), and lightly smoothed (so sensor
    noise in a dark room isn't mistaken for detail).
    """
    h, w = gray_face.shape
    inner = gray_face[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)]
    if inner.size == 0:
        return 0.0
    f = cv2.resize(inner, (128, 128), interpolation=cv2.INTER_AREA if inner.shape[0] > 128 else cv2.INTER_CUBIC)
    f = f.astype(np.float32)
    f = (f - f.mean()) / (f.std() + 1e-6) * 40
    f = cv2.GaussianBlur(f, (0, 0), 0.8)
    return float(cv2.Laplacian(f, cv2.CV_32F).var())


class Scorer:
    """Holds the MediaPipe models. Create one per process."""

    def __init__(self, th: Thresholds | None = None):
        import mediapipe as mp
        from mediapipe.tasks.python import vision, BaseOptions
        self.mp = mp
        self.th = th or Thresholds()
        self.detector = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(MODELS / "blaze_face_full_range.tflite")),
            min_detection_confidence=self.th.det_confidence))
        self.landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODELS / "face_landmarker.task")),
            output_face_blendshapes=True, num_faces=3,
            min_face_detection_confidence=0.4, min_face_presence_confidence=0.4))

    def close(self):
        self.detector.close()
        self.landmarker.close()

    def _mpimg(self, rgb):
        return self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

    def detect_faces(self, rgb: np.ndarray):
        """Full-frame pass plus a 2x2 tiled pass, so small faces in wide shots are still found."""
        H, W = rgb.shape[:2]
        scale = 1920 / max(H, W) if max(H, W) > 1920 else 1.0
        small = cv2.resize(rgb, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else rgb
        h, w = small.shape[:2]
        regions = [(0, 0, w, h)]
        tw, th = int(w * 0.6), int(h * 0.6)
        for ty in (0, h - th):
            for tx in (0, w - tw):
                regions.append((tx, ty, tw, th))
        boxes, scores = [], []
        for (rx, ry, rw, rh) in regions:
            res = self.detector.detect(self._mpimg(small[ry:ry + rh, rx:rx + rw]))
            for d in res.detections:
                bb = d.bounding_box
                boxes.append([(bb.origin_x + rx) / scale, (bb.origin_y + ry) / scale, bb.width / scale, bb.height / scale])
                scores.append(d.categories[0].score)
        keep = _nms(boxes, scores)
        return [boxes[i] for i in keep]

    def face_attrs(self, rgb: np.ndarray, box):
        """Blink + smile from face landmarker blendshapes, run on a padded crop around one face."""
        H, W = rgb.shape[:2]
        x, y, w, h = box
        pad = 0.6
        x0, y0 = int(max(0, x - w * pad)), int(max(0, y - h * pad))
        x1, y1 = int(min(W, x + w * (1 + pad))), int(min(H, y + h * (1 + pad)))
        crop = rgb[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        s = 256 / max(w, h)   # bring the face to ~256px, which the landmarker handles well
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        res = self.landmarker.detect(self._mpimg(crop))
        if not res.face_blendshapes:
            return None
        # pick the face nearest the crop's target face center
        cx, cy = (x + w / 2 - x0) * s / crop.shape[1], (y + h / 2 - y0) * s / crop.shape[0]
        best = min(range(len(res.face_landmarks)), key=lambda i: (res.face_landmarks[i][1].x - cx) ** 2
                   + (res.face_landmarks[i][1].y - cy) ** 2)
        bs = {c.category_name: c.score for c in res.face_blendshapes[best]}
        blink = max(bs.get("eyeBlinkLeft", 0), bs.get("eyeBlinkRight", 0))
        smile = (bs.get("mouthSmileLeft", 0) + bs.get("mouthSmileRight", 0)) / 2
        return blink, smile

    def score(self, path: Path, t: float) -> FrameScore:
        th = self.th
        fs = FrameScore(file=path.name, t=round(t, 3))
        bgr = cv2.imread(str(path))
        if bgr is None:
            fs.fail = "unreadable"
            return fs
        H, W = bgr.shape[:2]
        fs.width, fs.height = W, H
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        fs.dhash = dhash(gray)

        g_small = cv2.resize(gray, (480, int(480 * H / W)), interpolation=cv2.INTER_AREA)
        fs.frame_luma = float(g_small.mean())
        fs.clipped = float((g_small >= 250).mean())
        fs.crushed = float((g_small <= 8).mean())

        boxes = self.detect_faces(rgb)
        fs.n_faces = len(boxes)
        min_px = max(th.min_face_px, th.min_face_frac * min(H, W))
        clear = sorted([b for b in boxes if b[3] >= min_px], key=lambda b: -b[2] * b[3])
        fs.n_clear = len(clear)
        if not clear:
            fs.fail = "no_clear_face"
            return fs

        # "main" faces: the largest, plus any at least 40% of its area (skips tiny background faces)
        main = [b for b in clear if b[2] * b[3] >= 0.4 * clear[0][2] * clear[0][3]][:4]
        fs.faces = [[round(v) for v in b] for b in main]
        fs.face_frac = round(main[0][3] / H, 4)

        sharp, areas, lumas, blinks, smiles = [], [], [], [], []
        for (x, y, w, h) in main:
            xi, yi = int(max(0, x)), int(max(0, y))
            fg = gray[yi:int(y + h), xi:int(x + w)]
            if fg.size == 0:
                continue
            sharp.append(sharpness(fg)); areas.append(w * h); lumas.append(float(fg.mean()))
            a = self.face_attrs(rgb, (x, y, w, h))
            if a:
                blinks.append(a[0]); smiles.append(a[1])
        if not sharp:
            fs.fail = "no_clear_face"
            return fs
        areas = np.array(areas)
        fs.face_sharpness = round(float(np.average(sharp, weights=areas)), 1)
        fs.min_sharpness = round(min(sharp), 1)
        fs.face_luma = round(float(np.average(lumas, weights=areas)), 1)
        if blinks:
            fs.eyes_open = round(1 - max(blinks), 3)
            fs.smile = round(float(np.mean(smiles)), 3)

        # ---- filters (first failure wins; exposure first, since dark faces also read as blurry) ----
        if (fs.face_luma < th.min_face_luma or fs.face_luma > th.max_face_luma
                or fs.clipped > th.max_clipped or fs.frame_luma < th.min_frame_luma):
            fs.fail = "bad_exposure"
        elif fs.min_sharpness < th.min_sharpness:
            fs.fail = "blurry_face"
        elif blinks and max(blinks) > th.max_blink:
            fs.fail = "eyes_closed"

        fs.local_score = local_score(fs, th)
        return fs


def local_score(fs: FrameScore, th: Thresholds) -> float:
    """0-100. Weights: sharpness 25, face size 20, smile 20, exposure 15, eyes 10, face count 10."""
    size = min(1.0, fs.face_frac / 0.25)
    count = {0: 0, 1: 0.6}.get(fs.n_clear, 1.0)
    sharp = min(1.0, fs.face_sharpness / (th.min_sharpness * 3))
    smile = max(fs.smile, 0) if fs.smile >= 0 else 0.3
    eyes = fs.eyes_open if fs.eyes_open >= 0 else 0.5
    expo = max(0.0, 1 - abs(fs.face_luma - 140) / 110) * (1 - min(1.0, fs.clipped * 3))
    return round(100 * (0.25 * sharp + 0.20 * size + 0.20 * smile + 0.15 * expo + 0.10 * eyes + 0.10 * count), 1)


def same_layout(a: FrameScore, b: FrameScore, tol: float = 0.05) -> bool:
    """Same number of main faces, each in about the same spot and size (the same shot of the same people)."""
    if not a.faces or len(a.faces) != len(b.faces):
        return False
    for (ax, ay, aw, ah), (bx, by, bw, bh) in zip(sorted(a.faces), sorted(b.faces)):
        if (abs((ax + aw / 2) - (bx + bw / 2)) > tol * a.width or abs((ay + ah / 2) - (by + bh / 2)) > tol * a.height
                or not 0.8 <= aw / max(bw, 1) <= 1.25):
            return False
    return True


def near_duplicate(a: FrameScore, b: FrameScore, max_dist: int) -> bool:
    """Frame hash is close, or faces are laid out the same and the hash is only loosely close
    (background motion like lights or other dancers changes the hash but not the photo)."""
    d = hamming(a.dhash, b.dhash)
    return d <= max_dist or (d <= 2 * max_dist and same_layout(a, b))


def group_moments(frames: list[FrameScore], max_dist: int = 40, max_gap: float = 1.0) -> list[list[FrameScore]]:
    """Group passing frames (time-sorted) into moments of near-duplicates.

    A new moment starts when the picture changes enough (see near_duplicate; dHash is 256 bits)
    or there's a time gap longer than max_gap seconds.
    """
    groups: list[list[FrameScore]] = []
    for f in sorted(frames, key=lambda f: f.t):
        if groups:
            g = groups[-1]
            if f.t - g[-1].t <= max_gap and near_duplicate(f, g[0], max_dist):
                g.append(f)
                continue
        groups.append([f])
    return groups


def pick_shortlist(best_per_moment: list[FrameScore], n: int = 15, min_dist: int = 30,
                   min_gap: float = 1.0) -> list[FrameScore]:
    """Top-n by local score, skipping frames that look like one already picked (repeat moves)
    or that are within min_gap seconds of one, so the shortlist covers different moments."""
    out: list[FrameScore] = []
    for f in sorted(best_per_moment, key=lambda f: -f.local_score):
        if all(not near_duplicate(f, o, min_dist) and abs(f.t - o.t) >= min_gap for o in out):
            out.append(f)
        if len(out) == n:
            break
    return out
