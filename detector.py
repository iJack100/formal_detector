"""
Detector formal / informal por webcam (Ecuador) - multi persona.

Pipeline:
  webcam -> YOLOv8n (recorta a TODAS las personas) -> CLIP zero-shot por persona
         -> tracker por IoU (cada persona con su propio suavizado / histeresis)
         -> re-identificacion (Re-ID): si alguien sale y vuelve, se lo reconoce
            por su apariencia (embedding de CLIP) y opcionalmente por su cara
            (YuNet + SFace de OpenCV), asi conserva su id y no se cuenta dos veces
         -> overlay con contador de personas formales / informales

No se entrena nada: CLIP compara cada recorte contra descripciones de texto.

Uso:
  python detector.py                 # webcam
  python detector.py --camera 1      # otra webcam
  python detector.py --image foto.jpg   # probar con una foto
  python detector.py --no-yolo       # usar el frame completo sin recortar
  python detector.py --face          # ademas de la ropa, reconocer por cara
  python detector.py --camera 1 --face   # camara USB (indice 1) + cara; 1280x720 por defecto
  q / ESC para salir
"""

import argparse
import os
import time
import urllib.request
from collections import deque

import cv2
import numpy as np
import torch
from PIL import Image

# --------------------------------------------------------------------------
# Prendas por clase (en ingles porque CLIP entiende mejor asi).
#
# OJO con la manga larga: un abrigo, chaqueta, sweater o buzo tambien es de
# manga larga y NO es formal. Por eso la lista INFORMAL incluye muchas prendas
# de manga larga casuales, y las descripciones FORMAL describen la prenda por
# cuello / botones / corbata / saco, no por "manga larga".
# --------------------------------------------------------------------------
FORMAL_GARMENTS = [
    "a suit and tie",
    "a blazer or suit jacket over a dress shirt",
    "a dress shirt with a collar and buttons",
    "a dress shirt and a tie",
    "a plain button-up shirt with a collar tucked into dress pants",
    "a blouse and formal dress pants",
    "an elegant formal dress",
    "a guayabera shirt",
    "business attire",
    "a neat work uniform with a collar",
    "a formal wool overcoat over a suit",
    "formal clothes",
]

INFORMAL_GARMENTS = [
    "a t-shirt",
    "a t-shirt and jeans",
    "a long-sleeve t-shirt",
    "a hoodie or sweatshirt",
    "a tank top",
    "a football jersey",
    "a baseball cap and a t-shirt",
    "shorts and sneakers",
    "a graphic tee with a print",
    "a tracksuit or sportswear",
    "pajamas",
    # abrigos / chaquetas casuales (manga larga pero NO formal)
    "a casual zip-up jacket",
    "a puffer jacket or winter coat",
    "a fleece jacket",
    "a denim jacket",
    "a bomber jacket",
    "a leather jacket",
    "a windbreaker or rain jacket",
    "a knit sweater or cardigan",
    "a flannel shirt open over a t-shirt",
    "a sports jacket with a zipper",
    "casual clothes",
]

TEMPLATES = [
    "a photo of a person wearing {}.",
    "a webcam photo of a person wearing {}.",
    "a photo of a person at home wearing {}.",
    "a photo of a person in a bedroom wearing {}.",
    "a low quality photo of a person wearing {}.",
    "a close-up photo of someone wearing {}.",
    "a person wearing {}.",
]

CLASSES = ["FORMAL", "INFORMAL"]
COLOR = {"FORMAL": (60, 180, 75), "INFORMAL": (30, 100, 240), "...": (200, 200, 200)}


def load_clip(model_name: str, pretrained: str, device: str):
    """Devuelve (model, preprocess, garment_feats [G, D], garment_class [G])."""
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    feats, cls = [], []
    with torch.no_grad():
        for ci, garments in enumerate((FORMAL_GARMENTS, INFORMAL_GARMENTS)):
            for g in garments:
                tok = tokenizer([t.format(g) for t in TEMPLATES]).to(device)
                t = model.encode_text(tok)
                t = t / t.norm(dim=-1, keepdim=True)
                t = t.mean(dim=0)          # promedio sobre templates
                feats.append(t / t.norm())
                cls.append(ci)
    garment_feats = torch.stack(feats)                        # [G, D]
    garment_class = torch.tensor(cls, device=device)          # [G]
    return model, preprocess, (garment_feats, garment_class)


def load_yolo():
    try:
        from ultralytics import YOLO

        return YOLO("yolov8n.pt")          # se descarga solo la primera vez
    except Exception as e:  # noqa: BLE001
        print(f"[aviso] YOLO no disponible ({e}); se usa el frame completo.")
        return None


def detect_people(yolo, frame_bgr, conf=0.3, imgsz=640, min_frac=0.02, max_people=20):
    """Devuelve lista de (x1, y1, x2, y2) de TODAS las personas (mas grandes primero)."""
    if yolo is None:
        return []
    res = yolo.predict(frame_bgr, classes=[0], conf=conf, iou=0.5,
                       verbose=False, imgsz=imgsz)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return []
    h, w = frame_bgr.shape[:2]
    boxes = res.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = np.argsort(-areas)
    out = []
    for i in order[:max_people]:
        if areas[i] < min_frac * h * w:     # muy chica: casi seguro ruido / fondo
            continue
        x1, y1, x2, y2 = boxes[i]
        # recorte ajustado: cuanto menos fondo (cama, cuarto) vea CLIP, mejor
        pad = 0.02
        bw, bh = x2 - x1, y2 - y1
        x1 = max(0, int(x1 - pad * bw)); y1 = max(0, int(y1 - pad * bh))
        x2 = min(w, int(x2 + pad * bw)); y2 = min(h, int(y2 + pad * bh))
        if x2 - x1 > 8 and y2 - y1 > 8:
            out.append((x1, y1, x2, y2))
    return out


@torch.no_grad()
def classify_batch(model, preprocess, text_feats, crops_bgr, device,
                   formal_bias=0.0, score="mean", topk=3):
    """Devuelve (probs [N, 2] con [p_formal, p_informal], embs [N, D]) por recorte.

    embs es el embedding de imagen de CLIP (normalizado). Describe la apariencia
    de la persona (ropa, contextura, pelo, accesorios) y se usa para
    re-identificarla cuando sale y vuelve a entrar en camara.

    score:
      mean -> el logit de cada clase es el promedio de TODAS sus prendas
              (default; mas robusto al fondo).
      topk -> promedio de las K prendas que mejor coinciden por clase. Mas
              sensible a una prenda puntual (ej. "puffer jacket"), pero
              tambien a distractores del fondo.

    formal_bias se suma al logit de FORMAL antes del softmax (calibracion
    manual: +1.0 sube ~20 puntos una prediccion que estaba en 50/50).
    """
    if not crops_bgr:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 1), dtype=np.float32)
    garment_feats, garment_class = text_feats
    imgs = []
    for crop in crops_bgr:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        imgs.append(preprocess(Image.fromarray(rgb)))
    img = torch.stack(imgs).to(device)
    f = model.encode_image(img)
    f = f / f.norm(dim=-1, keepdim=True)
    g_logits = 100.0 * f @ garment_feats.T          # [N, G]  escala tipica de CLIP

    logits = []
    for ci in range(len(CLASSES)):
        lg = g_logits[:, garment_class == ci]       # [N, G_ci]
        if score == "mean":
            logits.append(lg.mean(dim=1))
        else:
            k = min(topk, lg.shape[1])
            logits.append(lg.topk(k, dim=1).values.mean(dim=1))
    logits = torch.stack(logits, dim=1)             # [N, 2]
    logits[:, 0] += formal_bias
    return logits.softmax(dim=-1).cpu().numpy(), f.cpu().numpy()


# --------------------------------------------------------------------------
# Cara: YuNet (deteccion) + SFace (embedding de 128 dims), ambos incluidos en
# OpenCV (cv2.FaceDetectorYN / cv2.FaceRecognizerSF). Los .onnx se descargan
# solos la primera vez desde opencv_zoo. Sirve de DESEMPATE cuando la ropa es
# igual (uniformes): si dos caras coinciden es la misma persona; si claramente
# no coinciden, son personas distintas aunque vistan igual.
#
# Limite fisico: la cara tiene que medir >= --min-face px de ancho (40 por
# defecto). A 640x480 eso es ~1.5 m de la camara; con --res 1920x1080 llega a
# ~4 m. Mas lejos, la cara no aporta y se usa solo la ropa.
# --------------------------------------------------------------------------
FACE_MODELS = {
    "face_detection_yunet_2023mar.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


class FaceEngine:
    def __init__(self, min_face=40, score=0.5):
        here = os.path.dirname(os.path.abspath(__file__))
        paths = {}
        for name, url in FACE_MODELS.items():
            p = os.path.join(here, name)
            if not os.path.exists(p):
                print(f"descargando {name}...", flush=True)
                urllib.request.urlretrieve(url, p)
            paths[name] = p
        try:                                   # silenciar avisos internos de cv2.dnn
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        except Exception:  # noqa: BLE001
            pass
        self.det = cv2.FaceDetectorYN.create(
            paths["face_detection_yunet_2023mar.onnx"], "", (320, 320), score, 0.3, 5000)
        self.rec = cv2.FaceRecognizerSF.create(
            paths["face_recognition_sface_2021dec.onnx"], "")
        self.min_face = min_face
        self.size = None

    def __call__(self, frame_bgr, boxes, needed=None):
        """Para cada caja de persona devuelve ((x, y, w, h), feat), None si no
        se vio cara, o False si no se calculo (needed[i] == False).

        YuNet corre sobre el 60 % superior del recorte de cada persona (no sobre
        el frame entero): es 4-5x mas rapido y no hay que asignar caras a cajas.
        SFace (el embedding) es lo caro (~100 ms en CPU), por eso `needed`."""
        out = [None] * len(boxes)
        H, W = frame_bgr.shape[:2]
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            if needed is not None and not needed[i]:
                out[i] = False
                continue
            cy2 = min(H, y1 + int((y2 - y1) * 0.6))
            crop = frame_bgr[y1:cy2, x1:x2]
            ch, cw = crop.shape[:2]
            if cw < self.min_face or ch < self.min_face:
                continue
            # recortes grandes (persona muy cerca) se reducen para detectar:
            # la cara ahi es enorme igual, y YuNet tarda proporcional al area
            s = min(1.0, 480.0 / max(cw, ch))
            small = crop if s >= 1.0 else cv2.resize(crop, (int(cw * s), int(ch * s)))
            sh, sw = small.shape[:2]
            if self.size != (sw, sh):
                self.det.setInputSize((sw, sh))
                self.size = (sw, sh)
            _, faces = self.det.detect(small)
            if faces is None or len(faces) == 0:
                continue
            f = max(faces, key=lambda r: r[2])     # la cara mas grande del recorte
            f = f.copy(); f[:14] /= s              # coordenadas de vuelta al recorte original
            if f[2] < self.min_face:               # muy chica: embedding poco fiable
                continue
            aligned = self.rec.alignCrop(crop, f)
            feat = _unit(self.rec.feature(aligned).flatten().astype(np.float32))
            out[i] = ((x1 + int(f[0]), y1 + int(f[1]), int(f[2]), int(f[3])), feat)
        return out


# --------------------------------------------------------------------------
# Tracker minimo por IoU: cada persona conserva su id, su historial de
# probabilidades (suavizado) y su etiqueta con histeresis, aunque se cruce
# con otra o YOLO la pierda un par de frames.
#
# Re-ID: cuando un track se pierde pasa a una GALERIA (con su embedding de
# apariencia, su cara si la hubo, y su etiqueta). Un track nuevo, mientras es
# "provisional" (edad < min_age), se compara contra la galeria:
#   - si ambos tienen cara y coinciden (>= face_thr)          -> misma persona
#   - si ambos tienen cara y claramente NO coinciden           -> descartado,
#     aunque la ropa sea identica (caso uniformes)
#   - si no, si la apariencia (ropa) coincide (>= reid)        -> misma persona
# Al re-identificar se recupera el id, el historial y la etiqueta, y el
# contador de "personas distintas" no sube. Reglas duras: nunca se fusiona
# con un track que sigue visible, y la galeria olvida a quien no se ve hace
# mas de `forget` segundos.
#
# Corto plazo: si una caja nueva no solapa con ningun track (la persona se
# movio rapido, YOLO la perdio un instante) se la compara por apariencia con
# los tracks que NO se vieron en este paso antes de crear un id nuevo.
# --------------------------------------------------------------------------
def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class Tracker:
    def __init__(self, smooth, flip, max_missed=5, iou_thr=0.3, min_age=3,
                 reid=0.0, face_thr=0.0, forget=0.0, reid_tries=8):
        self.smooth, self.flip = smooth, flip
        self.max_missed, self.iou_thr, self.min_age = max_missed, iou_thr, min_age
        self.reid, self.face_thr, self.forget = reid, face_thr, forget
        self.reid_tries = reid_tries   # pasos en que un track nuevo sigue buscandose en la galeria
        self.tracks = []          # dicts: id, box, hist, probs, label, missed, age, emb, face
        self.gallery = []         # tracks perdidos, candidatos a re-identificar
        self.next_id = 1
        self.seen_ids = set()     # ids que vivieron >= min_age pasos (acumulado)
        self.n_reid = 0           # cuantas veces se reconocio a alguien que volvio

    def update(self, boxes, probs, embs=None, faces=None, now=None):
        """boxes: lista de cajas; probs: [N, 2]; embs: [N, D] (CLIP);
        faces: lista alineada con boxes de ((x, y, w, h), feat) o None."""
        now = time.time() if now is None else now
        if embs is None or len(embs) != len(boxes):
            embs = [None] * len(boxes)
        if faces is None:
            faces = [None] * len(boxes)
        unmatched = list(range(len(boxes)))
        for t in self.tracks:
            t["matched"] = False

        # emparejar greedy por mayor IoU
        pairs = []
        for ti, t in enumerate(self.tracks):
            for bi in range(len(boxes)):
                v = iou(t["box"], boxes[bi])
                if v >= self.iou_thr:
                    pairs.append((v, ti, bi))
        pairs.sort(reverse=True)
        used_t, used_b = set(), set()
        for v, ti, bi in pairs:
            if ti in used_t or bi in used_b:
                continue
            used_t.add(ti); used_b.add(bi)
            self._feed(self.tracks[ti], boxes[bi], probs[bi], embs[bi], faces[bi], now)
        unmatched = [bi for bi in unmatched if bi not in used_b]

        # corto plazo: caja sin solape vs tracks que no se vieron en este paso
        if self.reid > 0 and unmatched:
            still = []
            for bi in unmatched:
                lost = [t for t in self.tracks if not t["matched"]]
                g = self._best_match(lost, embs[bi], faces[bi][1] if faces[bi] else None)
                if g is not None:
                    self._feed(g[0], boxes[bi], probs[bi], embs[bi], faces[bi], now)
                else:
                    still.append(bi)
            unmatched = still

        for bi in unmatched:
            t = {"id": self.next_id, "box": boxes[bi], "hist": deque(maxlen=self.smooth),
                 "probs": np.array([0.5, 0.5]), "label": None, "missed": 0, "age": 0,
                 "matched": True, "emb": None, "face": None, "face_box": None,
                 "face_n": 0, "reid": None, "last_seen": now}
            self.next_id += 1
            self._feed(t, boxes[bi], probs[bi], embs[bi], faces[bi], now)
            self.tracks.append(t)

        alive = []
        for t in self.tracks:
            if not t["matched"]:
                t["missed"] += 1
                t["face_box"] = None
            if t["missed"] <= self.max_missed:
                alive.append(t)
            elif t["age"] >= self.min_age and self.reid > 0:
                self.gallery.append(t)         # se perdio: queda para reconocerlo si vuelve
        self.tracks = alive
        if self.forget > 0:
            self.gallery = [g for g in self.gallery if now - g["last_seen"] <= self.forget]

    def _feed(self, t, box, p, emb=None, face=None, now=None):
        t["box"] = box
        t["missed"] = 0
        t["matched"] = True
        t["age"] += 1
        t["last_seen"] = time.time() if now is None else now
        t["hist"].append(p)
        t["probs"] = np.mean(t["hist"], axis=0)
        # apariencia: media movil del embedding de CLIP (ropa + contextura + pelo...)
        if emb is not None:
            t["emb"] = _unit(emb) if t["emb"] is None else _unit(0.8 * t["emb"] + 0.2 * emb)
        # cara: media movil del embedding de SFace (solo cuando se vio una cara)
        if face is False:                      # no se calculo en este paso: se mantiene
            pass
        elif face is not None:
            fb, feat = face
            t["face"] = feat if t["face"] is None else _unit(0.7 * t["face"] + 0.3 * feat)
            t["face_box"] = fb
            t["face_n"] += 1
        else:
            t["face_box"] = None
        # mientras el track es nuevo, intenta reconocerlo en la galeria
        if self.reid > 0 and t["reid"] is None and t["age"] <= self.reid_tries:
            self._try_reid(t)
        self._relabel(t)
        if t["age"] >= self.min_age:
            self.seen_ids.add(t["id"])

    def _relabel(self, t):
        pr = t["probs"]
        # Histeresis: la etiqueta solo cambia cuando la OTRA clase supera
        # --flip de forma sostenida (promedio de las ultimas --smooth
        # predicciones). Evita que una camisa formal en 55/45 "parpadee".
        if t["label"] is None:
            t["label"] = CLASSES[int(np.argmax(pr))]
        elif t["label"] == "FORMAL" and pr[1] >= self.flip:
            t["label"] = "INFORMAL"
        elif t["label"] == "INFORMAL" and pr[0] >= self.flip:
            t["label"] = "FORMAL"

    def _best_match(self, candidates, emb, face, face_n=99):
        """Mejor candidato para una apariencia (emb) y cara (face) dadas.
        Cara manda; si no hay cara, decide la ropa. Devuelve (track, kind) o None.

        Una cara claramente distinta (< face_thr - 0.2) descarta al candidato en
        este paso. No es definitivo: el track nuevo reintenta varios pasos con
        su cara promediada, asi una mala vista inicial (medido: hasta 0.13 con
        la mano en la cara) no lo condena si despues la cara coincide."""
        if emb is None and face is None:
            return None
        best, best_score, best_kind = None, -1.0, None
        for g in candidates:
            kind, score = None, -1.0
            if self.face_thr > 0 and face is not None and g["face"] is not None:
                fsim = float(face @ g["face"])
                if fsim >= self.face_thr:
                    kind, score = "cara", 1.0 + fsim        # la cara pesa mas que la ropa
                elif fsim < self.face_thr - 0.2:
                    continue                                # caras distintas: no es esa persona
            if kind is None and emb is not None and g["emb"] is not None:
                csim = float(_unit(emb) @ g["emb"])
                if csim >= self.reid:
                    kind, score = "ropa", csim
            if kind is not None and score > best_score:
                best, best_score, best_kind = g, score, kind
        return None if best is None else (best, best_kind)

    def _try_reid(self, t):
        """Busca en la galeria a quien mejor coincida con t. Devuelve True si fusiono."""
        m = self._best_match(self.gallery, t["emb"], t["face"], t["face_n"])
        if m is None:
            return False
        best, best_kind = m
        # fusionar: t hereda id, historial, etiqueta y apariencia acumulada
        self.gallery.remove(best)
        self.seen_ids.discard(t["id"])         # el id provisional no cuenta
        t["id"] = best["id"]
        t["hist"] = deque(list(best["hist"]) + list(t["hist"]), maxlen=self.smooth)
        t["probs"] = np.mean(t["hist"], axis=0)
        t["label"] = best["label"]
        t["age"] += best["age"]
        if t["emb"] is not None and best["emb"] is not None:
            t["emb"] = _unit(0.5 * t["emb"] + 0.5 * best["emb"])
        elif t["emb"] is None:
            t["emb"] = best["emb"]
        if t["face"] is not None and best["face"] is not None:
            t["face"] = _unit(0.5 * t["face"] + 0.5 * best["face"])
        elif t["face"] is None:
            t["face"] = best["face"]
        t["reid"] = best_kind
        self.n_reid += 1
        return True

    def face_needed(self, boxes, step, refresh=5, enough=10):
        """Para que cajas vale la pena calcular la cara en este paso: las que
        no tienen track (nuevas), las de tracks nuevos o con pocas vistas de
        cara, y el resto solo cada `refresh` pasos (para refrescar el promedio)."""
        out = []
        for b in boxes:
            t = max(self.tracks, key=lambda tr: iou(tr["box"], b), default=None)
            if t is None or iou(t["box"], b) < self.iou_thr:
                out.append(True)
            elif t["age"] <= self.reid_tries or t["face_n"] < enough:
                out.append(True)
            else:
                out.append(step % refresh == 0)
        return out

    def visible(self):
        return [t for t in self.tracks if t["missed"] == 0]

    def counts(self):
        vis = self.visible()
        f = sum(1 for t in vis if t["label"] == "FORMAL")
        i = sum(1 for t in vis if t["label"] == "INFORMAL")
        return len(vis), f, i, len(self.seen_ids)


# --------------------------------------------------------------------------
# Dibujo
# --------------------------------------------------------------------------
def draw_person(frame, box, probs, label, pid=None, reid=None, face_box=None):
    color = COLOR.get(label, COLOR["..."])
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    conf = float(probs[0] if label == "FORMAL" else probs[1])
    txt = f"{label} {conf*100:.0f}%" if label in CLASSES else "..."
    if pid is not None:
        txt = f"#{pid} " + txt
    if reid:                                   # volvio y se lo reconocio (por cara o ropa)
        txt += f" (volvio:{reid})"
    if face_box is not None:
        fx, fy, fw, fh = face_box
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 200, 0), 1)
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    ty = y1 - 6 if y1 - th - 10 > 0 else y1 + th + 6
    cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 6, ty + 4), color, -1)
    cv2.putText(frame, txt, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2, cv2.LINE_AA)
    # mini barra formal <-> informal bajo la caja
    bw = max(40, x2 - x1)
    by = min(frame.shape[0] - 6, y2 + 4)
    cv2.rectangle(frame, (x1, by), (x1 + bw, by + 5), (80, 80, 80), -1)
    cv2.rectangle(frame, (x1, by), (x1 + int(bw * float(probs[0])), by + 5),
                  COLOR["FORMAL"], -1)


def draw_panel(frame, n_people, n_formal, n_informal, n_total, fps, n_reid=0):
    cv2.rectangle(frame, (10, 10), (300, 100), (0, 0, 0), -1)
    cv2.putText(frame, f"Personas: {n_people}", (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Formal: {n_formal}", (20, 66),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR["FORMAL"], 2, cv2.LINE_AA)
    cv2.putText(frame, f"Informal: {n_informal}", (150, 66),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR["INFORMAL"], 2, cv2.LINE_AA)
    cv2.putText(frame, f"distintas: {n_total}   volvieron: {n_reid}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    if fps > 0:
        cv2.putText(frame, f"{fps:.1f} fps", (frame.shape[1] - 90, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


def crops_for(frame, boxes):
    if not boxes:
        return [frame], [(0, 0, frame.shape[1], frame.shape[0])]
    crops = [frame[y1:y2, x1:x2] for (x1, y1, x2, y2) in boxes]
    keep = [(c, b) for c, b in zip(crops, boxes) if c.size > 0]
    return [c for c, _ in keep], [b for _, b in keep]


# --------------------------------------------------------------------------
# Modos
# --------------------------------------------------------------------------
def run_image(args, model, preprocess, text_feats, yolo, device):
    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"No pude abrir {args.image}")
    boxes = detect_people(yolo, frame, conf=args.conf, imgsz=args.imgsz,
                          min_frac=args.min_size, max_people=args.max_people)
    crops, boxes = crops_for(frame, boxes)
    probs, _ = classify_batch(model, preprocess, text_feats, crops, device,
                              args.formal_bias, args.score, args.topk)
    labels = [CLASSES[int(np.argmax(p))] for p in probs]
    n_f = labels.count("FORMAL"); n_i = labels.count("INFORMAL")
    print(f"{args.image}: {len(labels)} persona(s) -> formal={n_f} informal={n_i}")
    for k, (p, lab) in enumerate(zip(probs, labels), 1):
        print(f"  #{k}: {lab}  (formal={p[0]:.2f}, informal={p[1]:.2f})")
    if not args.no_show:
        for k, (b, p, lab) in enumerate(zip(boxes, probs, labels), 1):
            draw_person(frame, b, p, lab, k)
        draw_panel(frame, len(labels), n_f, n_i, len(labels), 0.0)
        out = args.image.rsplit(".", 1)[0] + "_result.jpg"
        cv2.imwrite(out, frame)
        print(f"guardado: {out}")
    return labels


def open_camera(spec, res=(640, 480)):
    """Abre la camara. spec: indice ("0"), ruta a video, o "auto" (prueba 0..3
    y se queda con la primera que entregue un frame). res: (ancho, alto) pedido."""
    if spec == "auto":
        candidates = [0, 1, 2, 3]
    elif spec.isdigit():
        candidates = [int(spec)]
    else:
        candidates = [spec]
    for src in candidates:
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW) if isinstance(src, int) else cv2.VideoCapture(src)
        if not cap.isOpened():
            cap.release()
            continue
        if isinstance(src, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
            # MJPG: sin esto muchas webcams USB entregan YUY2 sin comprimir y a
            # 720p/1080p se quedan en 5 fps. Pedirlo DESPUES de la resolucion y
            # no tocar CAP_PROP_FPS despues (eso vuelve a YUY2 en la Brio 100).
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        for _ in range(10):                # algunas camaras tardan en dar el primer frame
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                fcc = int(cap.get(cv2.CAP_PROP_FOURCC))
                fcc = "".join(chr((fcc >> (8 * i)) & 0xFF) for i in range(4)) if fcc > 0 else "?"
                print(f"camara: {src} ({frame.shape[1]}x{frame.shape[0]}, {fcc})", flush=True)
                if isinstance(src, int) and fcc != "MJPG" and frame.shape[1] > 640:
                    print("[aviso] la camara no acepto MJPG a esta resolucion; si va lento "
                          "proba --res 1280x720 o 640x480", flush=True)
                return cap
        cap.release()
    raise SystemExit(f"No pude abrir la camara {spec} (probe con --camera 0, 1...)")


def parse_res(s):
    try:
        w, h = s.lower().split("x")
        return int(w), int(h)
    except Exception:  # noqa: BLE001
        raise SystemExit(f"--res debe ser ANCHOxALTO, ej. 1280x720 (recibi {s!r})")


def run_webcam(args, model, preprocess, text_feats, yolo, device, face_engine=None):
    cap = open_camera(args.camera, parse_res(args.res))

    tracker = Tracker(smooth=args.smooth, flip=args.flip,
                      reid=args.reid, face_thr=(args.face_thr if face_engine else 0.0),
                      forget=args.forget * 60.0)
    n = 0
    t_prev = time.time()
    fps = 0.0
    fails = 0

    print("Listo. q o ESC para salir.", flush=True)
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            fails += 1                     # frame perdido: reintenta antes de rendirse
            if fails > 30:
                print("La camara dejo de entregar imagen; saliendo.", flush=True)
                break
            cv2.waitKey(30)
            continue
        fails = 0
        n += 1

        if n % args.every == 0:            # clasifica cada N frames (CPU friendly)
            boxes = detect_people(yolo, frame, conf=args.conf, imgsz=args.imgsz,
                                  min_frac=args.min_size, max_people=args.max_people)
            if yolo is None:
                boxes = [(0, 0, frame.shape[1], frame.shape[0])]
            crops, boxes = crops_for(frame, boxes) if boxes else ([], [])
            probs, embs = classify_batch(model, preprocess, text_feats, crops, device,
                                         args.formal_bias, args.score, args.topk)
            faces = None
            if face_engine is not None:
                faces = face_engine(frame, boxes, tracker.face_needed(boxes, n // args.every))
            tracker.update(boxes, probs, embs, faces)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
        t_prev = now

        n_p, n_f, n_i, n_t = tracker.counts()
        if args.no_show:
            if n % 30 == 0:
                det = ", ".join(
                    f"#{t['id']}:{t['label']}({t['probs'][0]:.2f})"
                    + (f"[volvio:{t['reid']}]" if t["reid"] else "")
                    + ("[cara]" if t["face_box"] is not None else "")
                    for t in tracker.visible())
                print(f"frame {n}: personas={n_p} formal={n_f} informal={n_i} "
                      f"distintas={n_t} volvieron={tracker.n_reid}  [{det}]")
            continue

        for t in tracker.visible():
            draw_person(frame, t["box"], t["probs"], t["label"] or "...", t["id"],
                        t["reid"], t["face_box"])
        draw_panel(frame, n_p, n_f, n_i, n_t, fps, tracker.n_reid)
        if n == 1:                             # ventana redimensionable (1080p no entra en pantalla)
            cv2.namedWindow("Formal / Informal", cv2.WINDOW_NORMAL)
            h, w = frame.shape[:2]
            if w > 1280:
                cv2.resizeWindow("Formal / Informal", 1280, int(1280 * h / w))
        cv2.imshow("Formal / Informal", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Detector formal/informal por webcam (multi persona)")
    ap.add_argument("--camera", type=str, default="auto",
                    help="indice de la webcam (0, 1...), ruta a un video, o auto (usa la primera que funcione)")
    ap.add_argument("--image", type=str, default=None, help="clasificar una foto en vez de webcam")
    ap.add_argument("--no-show", action="store_true",
                    help="no abrir ventana ni guardar imagen; con video/webcam imprime en consola")
    ap.add_argument("--no-yolo", action="store_true", help="no recortar a la persona")
    ap.add_argument("--every", type=int, default=3, help="clasificar cada N frames")
    ap.add_argument("--smooth", type=int, default=12, help="promediar ultimas N predicciones (por persona)")
    ap.add_argument("--flip", type=float, default=0.60,
                    help="la etiqueta cambia solo si la otra clase supera este umbral (0.5 = sin histeresis)")
    ap.add_argument("--formal-bias", type=float, default=0.0,
                    help="calibracion: suma al logit de FORMAL (+1.0 ~ +20 puntos en 50/50; negativo favorece INFORMAL)")
    ap.add_argument("--score", choices=["mean", "topk"], default="mean",
                    help="mean: promedio de todas las prendas de la clase; topk: solo las K que mejor coinciden")
    ap.add_argument("--topk", type=int, default=3, help="K para --score topk")
    ap.add_argument("--conf", type=float, default=0.3,
                    help="confianza minima de YOLO para contar una persona (bajalo para detectar mas)")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="tamano de entrada de YOLO (640 detecta mejor personas chicas; 416 es mas rapido)")
    ap.add_argument("--min-size", type=float, default=0.02,
                    help="area minima de la persona como fraccion del frame (filtra gente muy lejos)")
    ap.add_argument("--max-people", type=int, default=20, help="maximo de personas a clasificar por frame")
    ap.add_argument("--model", type=str, default="ViT-B-32")
    ap.add_argument("--pretrained", type=str, default="openai")
    # --- re-identificacion ---
    ap.add_argument("--reid", type=float, default=0.82,
                    help="Re-ID por apariencia: similitud minima (coseno, 0-1) del embedding CLIP para "
                         "decir que alguien que vuelve es la misma persona. 0 = apagado")
    ap.add_argument("--forget", type=float, default=15.0,
                    help="minutos que se recuerda a alguien que salio de camara (0 = para siempre)")
    ap.add_argument("--face", action="store_true",
                    help="reconocer tambien por cara (YuNet + SFace de OpenCV); desempata cuando la ropa es igual")
    ap.add_argument("--face-thr", type=float, default=0.40,
                    help="similitud minima entre caras para decir que es la misma persona (SFace coseno; 0.36-0.45)")
    ap.add_argument("--min-face", type=int, default=40,
                    help="ancho minimo de la cara en px para usarla (mas chica = poco fiable)")
    ap.add_argument("--res", type=str, default="1280x720",
                    help="resolucion pedida a la camara, ej. 1280x720 o 1920x1080 (mas px = cara util a mas distancia)")
    args = ap.parse_args()
    if not 0.5 <= args.flip < 1.0:
        raise SystemExit("--flip debe estar entre 0.5 y 1.0")
    if not 0.0 <= args.reid < 1.0:
        raise SystemExit("--reid debe estar entre 0 (apagado) y 1.0")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | cargando CLIP {args.model}/{args.pretrained}...")
    model, preprocess, text_feats = load_clip(args.model, args.pretrained, device)
    yolo = None if args.no_yolo else load_yolo()
    face_engine = None
    if args.face and not args.image:
        try:
            face_engine = FaceEngine(min_face=args.min_face)
            print("cara: YuNet + SFace activos", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[aviso] no pude activar el reconocimiento por cara ({e}); sigo solo con ropa.")

    if args.image:
        run_image(args, model, preprocess, text_feats, yolo, device)
    else:
        run_webcam(args, model, preprocess, text_feats, yolo, device, face_engine)


if __name__ == "__main__":
    main()
