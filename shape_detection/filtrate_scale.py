import sys
import cv2
import pytesseract

 # Fonction qui extrait l echelle
def extract_scale(img_input):
    img = img_input[0:100, 0:150]
    texte = pytesseract.image_to_string(img, config='--psm 6 digits')
    print(texte)
    return
 
 # Fonction qui supprime le carre de l echelle
def delete_scale(img_input, img_output):
    return "Hello World"

if __name__ == "__main__":
    
    # Vérification des inputs
    if len(sys.argv) == 2:
        if sys.argv[1] == "-help":
            print("Supprime le carre en haut a gauche qui donne l echelle et la calcule")
    elif len(sys.argv) != 3:
        print("Usage : python filtrate_scale.py <image_entree> <image_sortie>\npython filtrate_scale.py -help")
        sys.exit(1)
        
        
    extract_scale(cv2.imread(sys.argv[1]))
    # Extraction de l'echelle
    # nbr_pixels, nbr_unite, unite = extract_scale(cv2.imread(sys.argv[1]))
    # print(x, " pixels = ", echelle)