# Detector formal / informal por webcam

Clasifica en vivo si **cada persona** frente a la webcam esta vestida **formal** o
**informal**, y muestra un contador: cuantas personas hay, cuantas formales y cuantas
informales. No se entrena nada: usa YOLOv8n para detectar y recortar a todas las
personas y CLIP (zero-shot) para comparar cada recorte contra descripciones de ropa
formal / casual.

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
python detector.py                    # camara automatica (prueba 0, 1, 2, 3 y usa la primera que funcione)
python detector.py --camera 1         # una camara especifica
python detector.py --image foto.jpg   # probar con una foto (guarda foto_result.jpg)
python detector.py --camera video.mp4 # probar con un video
python detector.py --camera 1 --face  # camara USB (indice 1) y reconocer por cara a quien vuelve
```

`q` o `ESC` para salir.

## Opciones utiles

| flag | default | que hace |
|---|---|---|
| `--every N` | 3 | clasifica cada N frames (subilo si va lento en CPU) |
| `--smooth N` | 12 | promedia las ultimas N predicciones de cada persona para que no parpadee |
| `--flip X` | 0.60 | histeresis: la etiqueta cambia solo si la otra clase supera X sostenido (0.5 = apagada) |
| `--formal-bias B` | 0.0 | calibracion manual: suma B al logit de FORMAL (+1.0 ~ +20 puntos en 50/50; negativo favorece INFORMAL) |
| `--conf C` | 0.3 | confianza minima de YOLO para contar una persona (bajalo a 0.2 si no detecta a alguien) |
| `--imgsz N` | 640 | tamano de entrada de YOLO (640 detecta mejor gente chica/lejos; 416 es mas rapido) |
| `--min-size F` | 0.02 | area minima de una persona como fraccion del frame (filtra gente muy lejos / ruido) |
| `--max-people N` | 20 | maximo de personas a clasificar por frame |
| `--score` | mean | `mean`: promedio de todas las prendas de la clase; `topk`: solo las K que mejor coinciden (mas sensible a una prenda puntual pero tambien al fondo) |
| `--reid X` | 0.82 | Re-ID por apariencia: similitud minima (0-1) para decir que alguien que vuelve es la misma persona (0 = apagado) |
| `--forget M` | 15 | minutos que se recuerda a alguien que salio de camara (0 = para siempre) |
| `--face` | | reconocer tambien por cara (YuNet + SFace, incluidos en OpenCV; descarga 2 `.onnx` la primera vez) |
| `--face-thr X` | 0.40 | similitud minima entre caras para decir que es la misma persona (0.36 a 0.45 es razonable) |
| `--min-face N` | 40 | ancho minimo de la cara en px para usarla (mas chica = poco fiable) |
| `--res WxH` | 1280x720 | resolucion pedida a la camara (se pide MJPG para que no caiga a 5 fps; 640x480 si va lento) |
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

**Manga larga no es formal.** La lista INFORMAL incluye abrigos, chaquetas, sweaters,
buzos y remeras de manga larga (puffer, fleece, jean, bomber, cuero, rompeviento,
cardigan, camisa leñadora abierta...), y las descripciones FORMAL hablan de cuello,
botones, corbata y saco en vez de "manga larga". Asi alguien con una campera o
un sweater no cae en FORMAL solo por tener los brazos cubiertos.

Si con tu camara/luz la ropa formal queda sistematicamente cerca del 50 %, calibra
con `--formal-bias 0.5` (o `1.0`); si pasa al reves, usa un valor negativo.
Para que la etiqueta no cambie ante variaciones chicas hay histeresis (`--flip`):
estando en FORMAL solo pasa a INFORMAL cuando informal supera el 60 % de forma
sostenida, y viceversa. La barra siempre muestra el porcentaje crudo suavizado.

## No contar dos veces a quien sale y vuelve (Re-ID)

Sin esto, cada vez que alguien sale del encuadre y vuelve, recibe un id nuevo y el
contador de "personas distintas" sube otra vez. Con Re-ID (activo por defecto):

1. Cada persona acumula un **embedding de apariencia** (el mismo vector de CLIP que
   ya se calcula para clasificarla: ropa, contextura, pelo, mochila, lentes...).
   No cuesta inferencia extra.
2. Cuando la persona se pierde, pasa a una **galeria** y se la recuerda `--forget`
   minutos.
3. Cuando aparece alguien nuevo, en sus primeros frames se lo compara contra la
   galeria. Si la apariencia coincide (`--reid`), recupera su id, su etiqueta y su
   historial, y el contador **no sube**. En pantalla se ve `(volvio:ropa)`.

Reglas duras: nunca se fusiona con alguien que sigue visible (dos personas en
pantalla a la vez son distintas por definicion) y la galeria olvida a quien no se
ve hace mas de `--forget` minutos.

**Uniformes iguales.** Con ropa sola, dos personas con el mismo uniforme son
indistinguibles. Para eso esta `--face`: se detecta la cara (YuNet) y se saca un
embedding (SFace). La cara manda como desempate:

- las caras coinciden (>= `--face-thr`) -> misma persona, aunque la ropa difiera un poco
- las caras claramente NO coinciden -> persona distinta, aunque el uniforme sea identico
- no se ve la cara (de espaldas, muy lejos) -> decide la ropa como antes

En pantalla la cara detectada se marca con un recuadro celeste y la persona
reconocida muestra `(volvio:cara)`.

**Limite fisico de la cara.** SFace necesita la cara de al menos ~40 px de ancho
(`--min-face`). Tamano aproximado de una cara segun distancia y resolucion:

| distancia | 640x480 | 1280x720 | 1920x1080 |
|---|---|---|---|
| 1.5 m | ~50 px | ~100 px | ~150 px |
| 2 m | ~38 px | ~76 px | ~114 px |
| 3 m | ~25 px | ~50 px | ~76 px |
| 4 m | ~19 px | ~38 px | ~57 px |

Para una carpa o un salon: poner la camara **en la entrada, a altura de ojos,
apuntando por donde la gente entra de frente**. La resolucion por defecto es
1280x720; 1920x1080 solo sirve si la camara lo entrega en MJPG (la Brio 100 por
DirectShow a 1080p cae a 5 fps, asi que quedate en 720p). Asi la cara se registra al entrar, y adentro
el seguimiento sigue por ropa. Con la gente lejos y mirando para otro lado, la cara
no aporta y el sistema cae a ropa: con uniformes iguales va a haber algun error en
ambos sentidos (fusionar a dos, o contar dos veces a uno). Es un contador
aproximado, no un registro exacto.

Si ves que fusiona a personas distintas, subi `--reid` (0.88) o baja `--forget`.
Si ves que cuenta dos veces a la misma, baja `--reid` (0.78) o `--face-thr` (0.36).
(Medido con la Brio 100: la misma persona moviendose da similitud de ropa entre 0.81
y 1.0, por eso el default es 0.82.)
`--reid 0` apaga todo y vuelve al comportamiento anterior.

## Camaras

Si hay mas de una (integrada + USB), `auto` toma la primera que responda, que suele
ser la integrada. Para usar la USB pasa su indice: `--camera 1`. Para saber cual
es cual, `Get-PnpDevice -Class Camera,Image -Status OK` en PowerShell lista los
nombres; en la practica la que entrega mas resolucion es la USB.

## Como funciona

1. OpenCV lee la webcam (pidiendo MJPG a 30 fps).
2. YOLOv8n detecta a **todas** las personas del frame y recorta cada bounding box
   (con margen chico para que CLIP vea poco fondo).
3. CLIP codifica todos los recortes en un solo batch y compara cada uno (similitud
   coseno) con el embedding de cada clase (prendas x templates). Softmax sobre las
   dos similitudes = probabilidad por persona.
4. Un tracker simple por IoU le asigna un id a cada persona y lo mantiene entre
   frames, asi cada una tiene su propio suavizado (`--smooth`) e histeresis (`--flip`).
5. Si alguien se pierde y vuelve, se lo re-identifica por apariencia (embedding de
   CLIP) y, con `--face`, por cara (YuNet + SFace), para que conserve su id y no
   se cuente dos veces.
6. Se dibuja cada caja con `#id ETIQUETA %` y un panel con el contador:
   personas en pantalla, cuantas formales, cuantas informales, el acumulado de
   personas distintas y cuantas veces se reconocio a alguien que volvio.
# viernes-che
