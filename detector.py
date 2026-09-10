"""
Detector formal / informal por webcam (Ecuador).

Pipeline:
  webcam -> YOLOv8n (recorta a la persona) -> CLIP zero-shot (formal vs informal) -> overlay

No se entrena nada: CLIP compara la imagen contra descripciones de texto.

Uso:
  python detector.py                 # webcam 0
  python detector.py --camera 1      # otra webcam
  python detector.py --image foto.jpg   # probar con una foto
  python detector.py --no-yolo       # usar el frame completo sin recortar
  q / ESC para salir
"""

import argparse
import time
from collections import deque

import cv2
import numpy as np
import torch
from PIL import Image

# --------------------------------------------------------------------------
# Prendas por clase (en ingles porque CLIP entiende mejor asi).
# Cada prenda se combina con TODOS los TEMPLATES de abajo. Los templates
# mencionan el contexto (webcam, en casa, baja calidad...) en AMBAS clases,
# asi el fondo (cama, cuarto) y la calidad de la camara dejan de inclinar la
# balanza y CLIP se fija en la ropa.
# --------------------------------------------------------------------------
FORMAL_GARMENTS = [
    "a suit and tie",
    "a blazer or suit jacket",
    "a long-sleeve dress shirt with a collar and buttons",
    "a plain button-up shirt with a collar",
    "a dress shirt tucked into dress pants",
    "a blouse and formal dress pants",
    "an elegant formal dress",
    "a guayabera shirt",
    "business attire",
    "a neat work uniform",
    "formal clothes",
    "a shirt and tie",
]

INFORMAL_GARMENTS = [
    "a t-shirt",
    "a t-shirt and jeans",
    "a hoodie or sweatshirt",
    "a tank top",
    "a football jersey",
    "a baseball cap and a t-shirt",
    "shorts and sneakers",
    "a graphic tee with a print",
    "a tracksuit or sportswear",
    "a casual zip-up jacket",
    "pajamas",
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


def load_clip(model_name: str, pretrained: str, device: str):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    with torch.no_grad():
        feats = []
        for garments in (FORMAL_GARMENTS, INFORMAL_GARMENTS):
            per_garment = []
            for g in garments:
                tok = tokenizer([t.format(g) for t in TEMPLATES]).to(device)
                t = model.encode_text(tok)
                t = t / t.norm(dim=-1, keepdim=True)
                t = t.mean(dim=0)          # promedio sobre templates
                per_garment.append(t / t.norm())
            c = torch.stack(per_garment).mean(dim=0)   # promedio sobre prendas
            feats.append(c / c.norm())
        text_feats = torch.stack(feats)    # [2, D]

    return model, preprocess, text_feats


def load_yolo():
    try:
        from ultralytics import YOLO

        return YOLO("yolov8n.pt")          # se descarga solo la primera vez
    except Exception as e:  # noqa: BLE001
        print(f"[aviso] YOLO no disponible ({e}); se usa el frame completo.")
        return None


def detect_person(yolo, frame_bgr):
    """Devuelve (x1, y1, x2, y2) de la persona mas grande, o None."""
    if yolo is None:
        return None
    res = yolo.predict(frame_bgr, classes=[0], conf=0.4, verbose=False, imgsz=416)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None
    boxes = res.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    x1, y1, x2, y2 = boxes[int(areas.argmax())]
    # recorte ajustado: cuanto menos fondo (cama, cuarto) vea CLIP, mejor
    h, w = frame_bgr.shape[:2]
    pad = 0.02
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - pad * bw)); y1 = max(0, int(y1 - pad * bh))
    x2 = min(w, int(x2 + pad * bw)); y2 = min(h, int(y2 + pad * bh))
    return x1, y1, x2, y2


@torch.no_grad()
def classify(model, preprocess, text_feats, crop_bgr, device, formal_bias=0.0):
    """Devuelve probabilidades [p_formal, p_informal].

    formal_bias se suma al logit de FORMAL antes del softmax (calibracion
    manual: +1.0 sube ~20 puntos una prediccion que estaba en 50/50).
    """
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img = preprocess(Image.fromarray(rgb)).unsqueeze(0).to(device)
    f = model.encode_image(img)
    f = f / f.norm(dim=-1, keepdim=True)
    logits = 100.0 * f @ text_feats.T        # escala tipica de CLIP
    logits[0, 0] += formal_bias
    return logits.softmax(dim=-1)[0].cpu().numpy()


def draw_overlay(frame, box, probs, label, fps):
    p_formal, p_informal = float(probs[0]), float(probs[1])
    conf = p_formal if label == "FORMAL" else p_informal
    color = (60, 180, 75) if label == "FORMAL" else (30, 100, 240)

    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # panel
    cv2.rectangle(frame, (10, 10), (330, 105), (0, 0, 0), -1)
    cv2.putText(frame, f"{label}  {conf*100:.0f}%", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    # barra formal <-> informal
    bar_x, bar_y, bar_w = 20, 65, 290
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 14), (80, 80, 80), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * p_formal), bar_y + 14),
                  (60, 180, 75), -1)
    cv2.putText(frame, f"formal {p_formal*100:.0f}%   informal {p_informal*100:.0f}%",
                (20, 97), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:.1f} fps", (frame.shape[1] - 90, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


def run_image(args, model, preprocess, text_feats, yolo, device):
    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"No pude abrir {args.image}")
    box = detect_person(yolo, frame)
    crop = frame if box is None else frame[box[1]:box[3], box[0]:box[2]]
    probs = classify(model, preprocess, text_feats, crop, device, args.formal_bias)
    label = CLASSES[int(np.argmax(probs))]
    print(f"{args.image}: {label}  (formal={probs[0]:.2f}, informal={probs[1]:.2f})")
    if not args.no_show:
        draw_overlay(frame, box, probs, label, 0.0)
        out = args.image.rsplit(".", 1)[0] + "_result.jpg"
        cv2.imwrite(out, frame)
        print(f"guardado: {out}")
    return label


def run_webcam(args, model, preprocess, text_feats, yolo, device):
    src = int(args.camera) if args.camera.isdigit() else args.camera
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"No pude abrir la camara {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    history = deque(maxlen=args.smooth)   # suaviza para que no parpadee
    probs = np.array([0.5, 0.5])
    label = None                          # etiqueta "pegajosa" (histeresis)
    box = None
    n = 0
    t_prev = time.time()
    fps = 0.0

    print("Listo. q o ESC para salir.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1

        if n % args.every == 0:            # clasifica cada N frames (CPU friendly)
            box = detect_person(yolo, frame)
            crop = frame if box is None else frame[box[1]:box[3], box[0]:box[2]]
            if crop.size > 0:
                history.append(classify(model, preprocess, text_feats, crop, device,
                                        args.formal_bias))
                probs = np.mean(history, axis=0)

            # Histeresis: la etiqueta solo cambia cuando la OTRA clase supera
            # --flip de forma sostenida (promedio de las ultimas --smooth
            # predicciones). Evita que una camisa formal en 55/45 "parpadee".
            if label is None:
                label = CLASSES[int(np.argmax(probs))]
            elif label == "FORMAL" and probs[1] >= args.flip:
                label = "INFORMAL"
            elif label == "INFORMAL" and probs[0] >= args.flip:
                label = "FORMAL"

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
        t_prev = now

        if args.no_show:
            if n % 30 == 0:
                print(f"frame {n}: {label}  formal={probs[0]:.2f} informal={probs[1]:.2f}")
            continue

        draw_overlay(frame, box, probs, label or "...", fps)
        cv2.imshow("Formal / Informal", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Detector formal/informal por webcam")
    ap.add_argument("--camera", type=str, default="0",
                    help="indice de la webcam (0, 1...) o ruta a un video")
    ap.add_argument("--image", type=str, default=None, help="clasificar una foto en vez de webcam")
    ap.add_argument("--no-show", action="store_true",
                    help="no abrir ventana ni guardar imagen; con video/webcam imprime en consola")
    ap.add_argument("--no-yolo", action="store_true", help="no recortar a la persona")
    ap.add_argument("--every", type=int, default=3, help="clasificar cada N frames")
    ap.add_argument("--smooth", type=int, default=12, help="promediar ultimas N predicciones")
    ap.add_argument("--flip", type=float, default=0.60,
                    help="la etiqueta cambia solo si la otra clase supera este umbral (0.5 = sin histeresis)")
    ap.add_argument("--formal-bias", type=float, default=0.0,
                    help="calibracion: suma al logit de FORMAL (+1.0 ~ +20 puntos en 50/50; negativo favorece INFORMAL)")
    ap.add_argument("--model", type=str, default="ViT-B-32")
    ap.add_argument("--pretrained", type=str, default="openai")
    args = ap.parse_args()
    if not 0.5 <= args.flip < 1.0:
        raise SystemExit("--flip debe estar entre 0.5 y 1.0")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | cargando CLIP {args.model}/{args.pretrained}...")
    model, preprocess, text_feats = load_clip(args.model, args.pretrained, device)
    yolo = None if args.no_yolo else load_yolo()

    if args.image:
        run_image(args, model, preprocess, text_feats, yolo, device)
    else:
        run_webcam(args, model, preprocess, text_feats, yolo, device)


if __name__ == "__main__":
    main()
