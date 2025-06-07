import cv2
from ultralytics import YOLO
import numpy as np
from tkinter import Tk, filedialog

Tk().withdraw()
img_path = filedialog.askopenfilename(title="Select image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
if not img_path:
    print("No image selected, exiting...")
    exit()

model = YOLO("yolov8n.pt")
class_names = model.names

orig = cv2.imread(img_path)
if orig is None:
    print("Error loading image")
    exit()

padding = 125
orig = cv2.copyMakeBorder(orig, padding, padding, padding, padding,
                          cv2.BORDER_CONSTANT, value=[255, 255, 255])

scale_percent = 30
width = int(orig.shape[1] * scale_percent / 100)
height = int(orig.shape[0] * scale_percent / 100)
img = cv2.resize(orig, (width, height))
clone = img.copy()

results = model(img)[0]
confidence_threshold = 0.7

confirmed_boxes = []
changed_boxes = []  
uncertain_boxes = []

if len(results.boxes) > 0:
    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        x1, y1, x2, y2 = map(int, box)
        class_id = int(cls)
        confidence = float(conf)
        label = class_names.get(class_id, "unknown")

        if confidence >= confidence_threshold:
            confirmed_boxes.append(((x1, y1, x2, y2), label))
            changed_boxes.append(False) 
        else:
            uncertain_boxes.append(((x1, y1, x2, y2), label, confidence))
else:
    print("No objects detected by YOLO.")

drawing = False
ix, iy = -1, -1
new_boxes = []

def draw_rectangle(event, x, y, flags, param):
    global drawing, ix, iy, img, clone, new_boxes
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img = clone.copy()
            cv2.rectangle(img, (ix, iy), (x, y), (0, 0, 255), 2)
            cv2.imshow("Reviewing Uncertain", img)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (ix, iy), (x, y), (0, 0, 255), 2)
        new_boxes.append((ix, iy, x, y))
        clone[:] = img[:]
        cv2.imshow("Reviewing Uncertain", img)

i = 0
while i < len(uncertain_boxes):
    (x1, y1, x2, y2), old_label, conf = uncertain_boxes[i]

    img = clone.copy()

    for idx, ((bx1, by1, bx2, by2), lbl) in enumerate(confirmed_boxes):
        color = (0, 0, 255) if changed_boxes[idx] else (0, 255, 0) 
        cv2.rectangle(img, (bx1, by1), (bx2, by2), color, 2)
        cv2.putText(img, lbl, (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    for j in range(i + 1, len(uncertain_boxes)):
        (ux1, uy1, ux2, uy2), ulbl, _ = uncertain_boxes[j]
        cv2.rectangle(img, (ux1, uy1), (ux2, uy2), (180, 180, 180), 1)

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 2)
    cv2.putText(img, f"{old_label} ({conf:.2f})", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    cv2.imshow("Reviewing Uncertain", img)
    cv2.waitKey(1)

    new_label = input(f"\nBox at ({x1},{y1}) → ({x2},{y2}) | Current label: '{old_label}' | Confidence: {conf:.2f}\nEnter new label or press Enter to keep, or type 'remove' to delete: ").strip()

    if new_label.lower() == "remove":
        print("❌ Box removed.")
        uncertain_boxes.pop(i)
        continue
    elif new_label and new_label != old_label:
        confirmed_boxes.append(((x1, y1, x2, y2), new_label))
        changed_boxes.append(True) 
        print(f"🔴 Label changed to '{new_label}'")
    else:
        confirmed_boxes.append(((x1, y1, x2, y2), old_label))
        changed_boxes.append(False)  
        print(f"🟢 Label confirmed as '{old_label}'")

    i += 1

if len(results.boxes) == 0:
    img = clone.copy()
    cv2.imshow("Reviewing Uncertain", img)
    cv2.setMouseCallback("Reviewing Uncertain", draw_rectangle)
    print("\n--- No objects detected, draw new boxes with mouse. Press ENTER (while window focused) when done ---")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  
            break

    for (x1, y1, x2, y2) in new_boxes:
        label = input(f"Enter label for new box at ({x1},{y1}) → ({x2},{y2}): ").strip()
        if label:
            confirmed_boxes.append(((x1, y1, x2, y2), label))
            changed_boxes.append(True) 

img = clone.copy()
for idx, ((x1, y1, x2, y2), label) in enumerate(confirmed_boxes):
    color = (0, 0, 255) if changed_boxes[idx] else (0, 255, 0)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

cv2.imshow("Final Annotated Image", img)
print("\n✅ Annotation complete. Close the window to finish.")
cv2.waitKey(0)
cv2.destroyAllWindows()
