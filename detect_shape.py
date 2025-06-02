import cv2
import numpy as np

# Charger l'image
img = cv2.imread("Yenway_3_edges3.tiff")


# Flouter et détecter les contours
blur = cv2.GaussianBlur(img, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

# Trouver les contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
original_contours = contours[:]

# Supprimer les petits contours contenus dans d'autres
def is_contained(inner, outer):
    return all(cv2.pointPolygonTest(outer, (float(pt[0][0]), float(pt[0][1])), False) >= 0 for pt in inner)

filtered_by_contours = []
for i, c1 in enumerate(contours):
    if cv2.contourArea(c1) < 30:
        continue
    contained = False
    for j, c2 in enumerate(contours):
        if i != j and is_contained(c1, c2):
            contained = True
            break
    if not contained:
        filtered_by_contours.append(c1)

# Filtrage basé sur les rectangles englobants
def rect_contains(inner, outer):
    x1, y1, w1, h1 = inner
    x2, y2, w2, h2 = outer
    return x1 >= x2 and y1 >= y2 and x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2

rects = [cv2.boundingRect(c) for c in original_contours if cv2.contourArea(c) >= 30]
filtered_by_rectangles = []
for i, r1 in enumerate(rects):
    contained = False
    for j, r2 in enumerate(rects):
        if i != j and rect_contains(r1, r2):
            contained = True
            break
    if not contained:
        filtered_by_rectangles.append(r1)
        
    

nbr_contours_1 = 0
nbr_contours_2 = 0

# Dessiner les contours filtrés par la méthode des contours imbriqués
for i, contour in enumerate(filtered_by_contours):
    area = cv2.contourArea(contour)
    if area < 30:
        continue
    x, y, w, h = cv2.boundingRect(contour)
    
    # Dessiner le contour et les mesures
    cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
    #cv2.putText(img, f"Contours {i}: {area:.0f}px", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    nbr_contours_1 += 1

# Dessiner les rectangles filtrés par la méthode des rectangles englobants
for i, (x, y, w, h) in enumerate(filtered_by_rectangles):
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(img, f"Rectangle {i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    nbr_contours_2 += 1
    

# Afficher le résultat
cv2.imwrite("Yenway_3_sizes.jpg", img)
print(nbr_contours_1, "shaped detected by contours")
print(nbr_contours_2, "shaped detected by rectangles")