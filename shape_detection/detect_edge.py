import cv2
import sys
import numpy as np

def detect_edges(input_path, output_path):
    # Read the image
    image = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Erreur : impossible de lire l'image {input_path}")
        sys.exit(1)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect edges using Canny
    edges = cv2.Canny(gray, 100, 300)

    # Dilate edges to make them thicker
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # taille ajustable
    edges_thick = cv2.dilate(edges, kernel)

    # Save the result (optionnel : inversion des couleurs)
    cv2.imwrite(output_path, cv2.bitwise_not(edges_thick))
    print(f"Contours détectés enregistrés dans {output_path}")
    



def filtre_passe_haut_sombres(input_path, output_path):
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Erreur : impossible de lire l'image {input_path}")
        return

    # 1. Flouter l'image pour créer une version "basse fréquence"
    #flou = cv2.GaussianBlur(image, (21, 21), 0)

    # 2. Passe-haut : original - flou
    #passe_haut = cv2.subtract(image, flou)

    # 3. Masque : garder seulement les zones sombres
    _, masque_sombre = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV)

    # 4. Appliquer le masque sombre au résultat du passe-haut
    resultat = cv2.bitwise_and(image, image, mask=masque_sombre)
    
    # 5. Repasser en cristaux blancs sur fond noirs
    resultat = cv2.bitwise_not(resultat)
    
    # 6. Re assombrir les cristaux car ils sont plus gris actuellement
    seuil = 180  # seuil de luminosité à partir duquel on assombrit
    facteur = 0.5  # intensité d'assombrissement
    resultat[resultat < seuil] = (resultat[resultat < seuil] * facteur).astype(np.uint8)

    # Sauvegarde
    cv2.imwrite(output_path, resultat)
    print(f"Filtre passe-haut (zones sombres) enregistré dans {output_path}")
    
    

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage : python detect_edge.py <image_entree> <image_sortie>")
        sys.exit(1)
    filtre_passe_haut_sombres(sys.argv[1], sys.argv[2])