# Detector formal / informal por webcam

Clasifica en vivo si la persona frente a la webcam esta vestida **formal** o **informal**.
No se entrena nada: usa YOLOv8n para recortar a la persona y CLIP (zero-shot) para
comparar la imagen contra descripciones de ropa formal / casual.

## Instalar

```bash
python -m venv venv
venv\Scripts\activate        # Windows   (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
```

Si no tenes GPU y queres que torch pese menos:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

La primera vez descarga solos los pesos de CLIP (~350 MB) y YOLOv8n (~6 MB).

## Usar

```bash
python detector.py                    # webcam 0
python detector.py --camera 1         # otra camara
python detector.py --image foto.jpg   # probar con una foto (guarda foto_result.jpg)
python detector.py --camera video.mp4 # probar con un video
```

`q` o `ESC` para salir.

## Opciones utiles

| flag | default | que hace |
|---|---|---|
| `--every N` | 3 | clasifica cada N frames (subilo si va lento en CPU) |
| `--smooth N` | 12 | promedia las ultimas N predicciones para que no parpadee |
| `--flip X` | 0.60 | histeresis: la etiqueta cambia solo si la otra clase supera X sostenido (0.5 = apagada) |
| `--formal-bias B` | 0.0 | calibracion manual: suma B al logit de FORMAL (+1.0 ~ +20 puntos en 50/50; negativo favorece INFORMAL) |
| `--no-show` | | no abre ventana; con `--camera video.mp4` imprime el resultado en consola |
| `--no-yolo` | | usa el frame completo sin recortar a la persona |
| `--model` / `--pretrained` | ViT-B-32 / openai | otro modelo CLIP (ej. `--model ViT-B-16`) |

## Ajustar a tu contexto

Todo el "conocimiento" esta en las listas `FORMAL_GARMENTS` e `INFORMAL_GARMENTS`
al inicio de `detector.py` (prendas, en ingles). Cada prenda se combina con las
frases de `TEMPLATES` ("a webcam photo of a person at home wearing {}...") para que
el fondo y la calidad de la camara no inclinen el resultado. Si algo se clasifica
mal, agrega o quita prendas. Ya incluye guayabera y uniforme; si en tu caso el
uniforme cuenta como informal, mové esa linea a la otra lista.

Si con tu camara/luz la ropa formal queda sistematicamente cerca del 50 %, calibra
con `--formal-bias 0.5` (o `1.0`); si pasa al reves, usa un valor negativo.
Para que la etiqueta no cambie ante variaciones chicas hay histeresis (`--flip`):
estando en FORMAL solo pasa a INFORMAL cuando informal supera el 60 % de forma
sostenida, y viceversa. La barra siempre muestra el porcentaje crudo suavizado.

## Como funciona

1. OpenCV lee la webcam.
2. YOLOv8n detecta a la persona mas grande y recorta el bounding box (con margen).
3. CLIP codifica el recorte y lo compara (similitud coseno) con el embedding promedio
   de cada clase (prendas x templates). Softmax sobre las dos similitudes = probabilidad.
4. Se promedian las ultimas N predicciones, se aplica histeresis a la etiqueta y se
   dibuja el resultado sobre el video.
# viernes-che
