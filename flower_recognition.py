import sys
import cv2
import numpy as np
import os
import sqlite3
import threading
import webbrowser
from PyQt5 import QtWidgets, QtGui, QtCore
from tensorflow.keras.models import load_model
import pyttsx3

model = load_model("flower_classification_model.h5")
class_indices = np.load("class_indices.npy", allow_pickle=True).item()
class_names = list(class_indices.keys())

engine = pyttsx3.init()
engine.setProperty('rate', 150)
voices = engine.getProperty('voices')
for voice in voices:
    if "english" in voice.name.lower() or "en" in voice.id.lower():
        engine.setProperty('voice', voice.id)

default_care_guide = {
    'Rose': 'Requires full sun and fertile soil. Water at base, avoiding foliage. Regular rose fertilizer applications needed. Essential spring pruning and deadheading. Vulnerable to fungal diseases - preventive treatments recommended.',
    'Chitrak': 'Drought-resistant; prefers sunny, rocky soils. Water sparingly. Little fertilization needed. Used in Ayurveda. Pruning optional for shaping.',
    'Sunflower': 'Requires full sun and regular watering. Plant in fertile soil. Potassium fertilizers boost growth. Tall varieties need support. Leave flower heads for birds after blooming.',
    'Bush Clock Vine': 'Requires bright sunlight but can grow in partial shade. Moderate watering is needed, with well-draining soil. Blooms profusely with regular feeding of balanced fertilizer. Pruning helps maintain bush shape. Drought-tolerant but benefits from extra watering during extreme heat.',
    'Common Lantana': 'Thrives in full sun and well-drained soil. Water moderately as it is drought-resistant. Continuous flowering occurs when spent blooms are removed regularly. Monthly fertilization enhances blooming. May attract whiteflies, requiring occasional insecticide treatment.',
    'Datura': 'Prefers full sun and fertile soil. Water regularly but avoid waterlogging. Produces large fragrant flowers, especially when fed phosphorus-potassium fertilizer. All plant parts are poisonous - handle with care. In cold climates, winter protection may be needed.',
    'Hibiscus': 'Needs bright light and consistent watering. Plant in nutrient-rich, well-draining soil. Fertilize every 2 weeks during flowering season. Spring pruning encourages new growth. Bud drop occurs if underwatered.',
    'Jatropha': 'Drought-tolerant; prefers full sun and light soils. Water infrequently but deeply. Apply mineral fertilizer every 2 months. Pruning maintains compact shape. Toxic - handle carefully.',
    'Marigold': 'Grows best in full sun with moderate watering. Adapts to most soils but prefers loose, well-drained. Deadheading prolongs blooming. Pest-resistant but may develop fungus if overwatered. Often used as natural pest deterrent for other plants.',
    'Nityakalyani': 'Prefers bright light but tolerates partial shade. Water regularly without overwatering. Monthly feeding boosts flowering. Post-bloom pruning maintains appearance. Resistant to most pests.',
    'Yellow Daisy': 'Prefers full sun and light soils. Water moderately - drought-tolerant. Monthly feeding extends blooming. Deadheading promotes new flowers. Rarely affected by pests.',
    'Adathoda': 'Grows in sun or partial shade. Water regularly without waterlogging. Feed with organic matter every 2 months. Pruning shapes the bush. Used in traditional medicine.',
    'Champaka': 'Prefers full sun and well-drained soil. Water moderately but consistently. Compost enhances growth and flowering. Fragrant flowers need adequate light. Post-bloom pruning maintains shape.',
    'Crown Flower': 'Needs bright light and moderate watering. Plant in well-draining soil. Fertilize every 2 months. Pruning encourages branching. All parts are poisonous.',
    'Four O-Clock Flower': 'Grows in sun or partial shade. Water moderately. Blooms profusely with minimal care. Monthly feeding increases flowering. Self-seeds prolifically - can become invasive.',
    'Honeysuckle': 'Prefers sun to light shade. Keep soil consistently moist. Fertilize in spring and summer. Prune after flowering. Fragrant blooms attract pollinators.',
    'Indian Mallow': 'Low-maintenance; grows in sun or shade. Water moderately. Fertilization optional. Pruning controls growth. Used in folk medicine.',
    'Malabar Melastome': 'Prefers partial shade and moist soil. Water regularly. Fertilize every 2 months. Prune to maintain shape. Produces purple flowers.',
    'Nagapoovu': 'Needs full sun and well-drained soil. Water moderately. Feed monthly. Prune after flowering. Fragrant flowers used in rituals.',
    'Pinwheel Flower': 'Grows in sun or partial shade. Water moderately. Fertilize every 2 months. Pruning unnecessary. Nearly year-round flowering in warm climates.',
}

conn = sqlite3.connect("flowers.db")
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS flowers (
    name TEXT PRIMARY KEY,
    care TEXT
)
""")

for flower, tip in default_care_guide.items():
    cursor.execute("INSERT OR IGNORE INTO flowers (name, care) VALUES (?, ?)", (flower, tip))
conn.commit()

def get_flower_prediction(frame):
    resized = cv2.resize(frame, (128, 128)).astype("float32") / 255.0
    resized = np.expand_dims(resized, axis=0)
    pred = model.predict(resized)
    class_idx = np.argmax(pred)
    class_name = class_names[class_idx]
    confidence = np.max(pred)
    return class_name, confidence

def speak_text(text):
    engine.say(text)
    engine.runAndWait()

def is_blurry(image, threshold=100.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

def get_care_tip(flower_name):
    cursor.execute("SELECT care FROM flowers WHERE LOWER(name) = ?", (flower_name.lower(),))
    result = cursor.fetchone()
    return result[0] if result else "No info."

class FlowerApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🌸 Flower Recognition App")
        self.setGeometry(100, 100, 1200, 700)
        self.setStyleSheet("background-color: #121212; color: white;")
        self.dark_mode = True

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QHBoxLayout(central_widget)

        self.video_label = QtWidgets.QLabel()
        self.video_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.video_label, 2)

        info_panel = QtWidgets.QVBoxLayout()

        self.result_label = QtWidgets.QLabel("Detected: None")
        self.care_label = QtWidgets.QLabel("Care Tips: ...")
        self.quality_label = QtWidgets.QLabel("Image Quality: OK")
        for label in [self.result_label, self.care_label, self.quality_label]:
            label.setWordWrap(True)
            info_panel.addWidget(label)

        self.search_input = QtWidgets.QLineEdit()
        self.search_input.setPlaceholderText("Search flower...")
        self.search_input.textChanged.connect(self.search_flowers)
        info_panel.addWidget(self.search_input)

        self.search_results = QtWidgets.QListWidget()
        self.search_results.itemClicked.connect(self.select_flower_from_search)
        info_panel.addWidget(self.search_results)

        buttons = [
            ("🔊 Speak", self.speak_info),
            ("🌐 Wikipedia", self.open_wikipedia),
            ("➕ Add Flower", self.add_flower),
            ("🎞 Load Video", self.load_video_file),
            ("🌓 Toggle Theme", self.toggle_theme)
        ]

        for label, func in buttons:
            btn = QtWidgets.QPushButton(label)
            btn.clicked.connect(func)
            info_panel.addWidget(btn)

        info_panel.addStretch()
        layout.addLayout(info_panel, 1)

        # --- Video Timer ---
        self.capture = cv2.VideoCapture(0)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            return

        class_name, confidence = get_flower_prediction(frame)
        blurry = is_blurry(frame)

        care_tip = get_care_tip(class_name)

        self.result_label.setText(f"Detected: {class_name} ({confidence:.2f})")
        self.care_label.setText(f"Care Tips: {care_tip}")
        self.quality_label.setText("Image Quality: Blurry" if blurry else "Image Quality: OK")

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_img = QtGui.QImage(rgb_image.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qt_img).scaled(
            self.video_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.video_label.setPixmap(pixmap)

    def speak_info(self):
        text = self.result_label.text() + ". " + self.care_label.text()
        threading.Thread(target=speak_text, args=(text,), daemon=True).start()

    def open_wikipedia(self):
        flower = self.result_label.text().split(':')[1].split('(')[0].strip()
        webbrowser.open(f"https://en.wikipedia.org/wiki/{flower.replace(' ', '_')}")

    def add_flower(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Add Flower", "Enter flower name:")
        if ok and name:
            tips, ok2 = QtWidgets.QInputDialog.getMultiLineText(self, "Care Tips", f"Enter care tips for {name}:")
            if ok2 and tips:
                cursor.execute("INSERT OR REPLACE INTO flowers (name, care) VALUES (?, ?)", (name, tips))
                conn.commit()
                QtWidgets.QMessageBox.information(self, "Success", f"{name} added to care guide.")

    def load_video_file(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi)")
        if filename:
            self.capture.release()
            self.capture = cv2.VideoCapture(filename)

    def toggle_theme(self):
        if self.dark_mode:
            self.setStyleSheet("background-color: white; color: black;")
        else:
            self.setStyleSheet("background-color: #121212; color: white;")
        self.dark_mode = not self.dark_mode

    def search_flowers(self):
        query = self.search_input.text().strip().lower()
        self.search_results.clear()
        if not query:
            return
        cursor.execute("SELECT name FROM flowers WHERE LOWER(name) LIKE ?", ('%' + query + '%',))
        matches = cursor.fetchall()
        self.search_results.addItems([m[0] for m in matches])

    def select_flower_from_search(self, item):
        name = item.text()
        care = get_care_tip(name)
        self.result_label.setText(f"Detected: {name} (Manual)")
        self.care_label.setText(f"Care Tips: {care}")

    def closeEvent(self, event):
        self.capture.release()
        conn.close()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = FlowerApp()
    win.show()
    sys.exit(app.exec_())
