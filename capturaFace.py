from flask import Flask, request, jsonify
import cv2
import numpy as np
import mysql.connector
import base64

app = Flask(__name__)

@app.route('/executar', methods=['POST'])
def executar_codigo():
    data = request.get_json()
    image_base64 = data.get('image')

    if not image_base64:
        return jsonify({"error": "Imagem não fornecida"}), 400

    # Decodificar a imagem de base64 para um array de bytes
    image_data = base64.b64decode(image_base64)
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)

    # Configuração do classificador Haar para detecção de rosto
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # Conectar ao banco de dados e salvar o rosto detectado
        conn = mysql.connector.connect(
            host="localhost",
            port=3306,
            user="root",
            password="root",
            database="casainteligente"
        )

        cursor = conn.cursor()
        cursor.execute("SELECT id FROM faces ORDER BY id DESC LIMIT 1")
        result = cursor.fetchone()
        id_pessoa = result[0] + 1 if result else 1

        # Codificar o rosto como imagem JPEG para armazenamento no banco
        _, buffer = cv2.imencode('.jpg', img)
        rosto_binario = buffer.tobytes()

        # Inserir no banco
        cursor.execute("INSERT INTO faces (id, photo) VALUES (%s, %s)", (id_pessoa, rosto_binario))
        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({"message": "Imagem processada e salva com sucesso!"})
    else:
        return jsonify({"message": "Nenhum rosto detectado."})

if __name__ == "__main__":
    app.run(debug=True)
