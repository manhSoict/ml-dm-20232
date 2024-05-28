import numpy as np
import tensorflow as tf
from keras._tf_keras.keras.models import model_from_json
import matplotlib.pyplot as plt
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

emotion_dict = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised",
}

# Tải mô hình đã huấn luyện từ file JSON và trọng số đã lưu
json_file = open("Model\\emotion_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# emotion_model.load_weights("Model\\emotion_model.weights.h5")
emotion_model.load_weights("Model\\emotion_model.h5")

print("Loaded model from disk")

# Tiền xử lý dữ liệu kiểm tra
test_data_gen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_data_gen.flow_from_directory(
    "Data\\test",
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical",
    shuffle=False,  # Đảm bảo rằng thứ tự của ảnh được giữ nguyên
)

# Sử dụng mô hình để dự đoán trên tập dữ liệu kiểm tra
predictions = emotion_model.predict(
    test_generator, steps=test_generator.samples // test_generator.batch_size + 1
)

# Tính toán ma trận nhầm lẫn và báo cáo phân loại để đánh giá hiệu suất của mô hình
c_matrix = confusion_matrix(test_generator.classes, predictions.argmax(axis=1))
print(c_matrix)
cm_display = ConfusionMatrixDisplay(
    confusion_matrix=c_matrix, display_labels=[v for k, v in emotion_dict.items()]
)
cm_display.plot(cmap=plt.cm.Blues)
plt.show()

print("#################################################################")
print(
    classification_report(
        test_generator.classes,
        predictions.argmax(axis=1),
        target_names=[v for k, v in emotion_dict.items()],
    )
)
