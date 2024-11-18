import cv2
import mysql.connector
import numpy as np

# Conectar ao banco de dados MySQL
conn = mysql.connector.connect(
    host="localhost",
    port=3306,
    user="root",
    password="root",
    database="casainteligente"
)

# Carregar o classificador Haar para detecção de rostos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar o modelo LBPH para reconhecimento facial
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Listas para armazenar os rostos e IDs
faces = []
ids = []

if conn.is_connected():
    cursor = conn.cursor()
    cursor.execute("SELECT id, photo FROM faces") 

    for id_pessoa, face_data in cursor.fetchall():
        # Converter o dado binário para uma imagem usando numpy
        np_array = np.frombuffer(face_data, dtype=np.uint8)
        imagem = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        # Converter para escala de cinza
        gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

        # Detectar rostos na imagem
        faces_detectadas = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Extrair os rostos detectados e associar ao ID
        for (x, y, w, h) in faces_detectadas:
            rosto = gray[y:y + h, x:x + w]
            faces.append(rosto)
            ids.append(id_pessoa)

    # Treinar o modelo com as imagens
    recognizer.train(faces, np.array(ids))

    # Salvar o modelo treinado
    recognizer.save('modelo_face.yml')
    print("Modelo treinado e salvo com sucesso!")

    # Fechar o cursor e a conexão
    cursor.close()
conn.close()
