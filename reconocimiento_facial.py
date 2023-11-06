import os
import face_recognition
import cv2

# Directorio donde se encuentran las imágenes de rostros conocidos
images_dir = "images"

# Cargar imágenes de las caras conocidas y sus nombres
known_face_encodings = []
known_face_names = []

# Listar archivos en el directorio de imágenes
for filename in os.listdir(images_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Obtener el nombre del archivo (sin la extensión) como el nombre del usuario
        name = os.path.splitext(filename)[0]

        # Cargar la imagen
        image_path = os.path.join(images_dir, filename)
        face_image = face_recognition.load_image_file(image_path)

        # Detectar caras en la imagen
        face_locations = face_recognition.face_locations(face_image)

        if len(face_locations) > 0:
            # Si se detecta al menos una cara, codificar el rostro
            face_encoding = face_recognition.face_encodings(face_image)[0]

            # Agregar la codificación y el nombre a la base de datos
            known_face_encodings.append(face_encoding)
            known_face_names.append(name)
        else:
            print(f"No se encontraron caras en {filename}")

# Inicializar la cámara
video_capture = cv2.VideoCapture(0)  # Puedes cambiar el índice para usar una cámara diferente

while True:
    # Capturar un frame de la cámara
    ret, frame = video_capture.read()

    # Convertir el frame a RGB (face_recognition trabaja con imágenes en formato RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Encontrar todas las caras en el frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Iterar sobre las caras encontradas en el frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Comprobar si la cara coincide con alguna de las caras conocidas
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Desconocido"  # Nombre predeterminado si no se encuentra ninguna coincidencia

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Dibujar un recuadro alrededor de la cara y mostrar el nombre
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Mostrar el frame resultante
    cv2.imshow('Video', frame)

    # Si se presiona la tecla 'q', salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar la ventana de visualización
video_capture.release()
cv2.destroyAllWindows()
