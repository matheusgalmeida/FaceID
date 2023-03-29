import cv2 as cv
import face_recognition
import os 
import numpy as np

face_location = [] 
cap = cv.VideoCapture(0)
tracker = cv.TrackerCSRT_create()

pasta_treinamento = "FaceID/treinamento/"
encondings_conhecidos = []
nomes_conhecidos = []
imagens_conhecidas = []

for diretorio_raiz, subdiretorios, arquivos in os.walk(pasta_treinamento):
    for subdiretorio in subdiretorios:
        for arquivo in os.listdir(os.path.join(diretorio_raiz, subdiretorio)):
            imagem = face_recognition.load_image_file(os.path.join(diretorio_raiz, subdiretorio, arquivo))
            enconding = face_recognition.face_encodings(imagem)[0]
            encondings_conhecidos.append(enconding)
            nomes_conhecidos.append(subdiretorio)



while True:

    ret, frame = cap.read()

    face_locations = face_recognition.face_locations(frame) 

    face_encodings = face_recognition.face_encodings(frame, face_locations) 

    for (top, right, bottom, left), face_encodings in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(encondings_conhecidos, face_encodings)
        best_match_index = np.argmin(face_recognition.face_distance(encondings_conhecidos,face_encodings))
        if matches[best_match_index]:
            top, right, bottom, left = face_locations[0]
            bbox = (left, top, right-left, bottom-top)
            tracker.init(frame, bbox)

    ret, bbox = tracker.update(frame)

    if ret:
        (x, y, w, h) = [int(v) for v in bbox]
        if w > 0 and h > 0:
            cv.rectangle(frame,(x, y), (x + w, y + h), (0, 255, 0), 2)

            face_encondig_atual = face_recognition.face_encodings(frame, [(top, right, bottom, left)])[0]

            distancias = face_recognition.face_distance(encondings_conhecidos, face_encondig_atual)

            limite_distancia = 0.75

            if min(distancias) < limite_distancia:
                nome = nomes_conhecidos[best_match_index]
                cv.putText(frame, nome, (x, y-20), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                print("Não foi possível identificar o nome da pessoa.")
                nome = "Desconhecido"
                cv.putText(frame, nome, (x-15, y-20), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

             

    cv.imshow("Webcam Capture", frame)

   # if cv.waitKey(1) & 0xFF == ord('q'):
   #     break
    if cv.waitKey(1) == 27:
        break
cap.release()
cv.destroyAllWindows()