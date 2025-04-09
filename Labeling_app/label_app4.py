import cv2
import os
import numpy as np
import random
import tkinter as tk
from tkinter import messagebox
import threading
import yaml 

class FaceScannerApp:
    def __init__(self, root, output_dir="yolo_face_dataset", train_ratio=0.7, val_ratio=0.15):
        self.root = root
        self.root.title("Face Scanner for YOLO")
        self.root.geometry("400x400")

        self.output_dir = output_dir
        self.train_dir = os.path.join(output_dir, "train")
        self.val_dir = os.path.join(output_dir, "valid")
        self.test_dir = os.path.join(output_dir, "test")
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.face_count = 0
        self.running = False
        self.people_names = []
        self.name_entries = []

        # Tạo thư mục cho train, val, test
        for dir_path in [self.train_dir, self.val_dir, self.test_dir]:
            for sub_dir in ["images", "labels"]:
                os.makedirs(os.path.join(dir_path, sub_dir), exist_ok=True)

        # Khởi tạo bộ phát hiện khuôn mặt
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # UI Elements
        tk.Label(root, text="Nhập số người:").pack(pady=5)
        self.num_people_entry = tk.Entry(root)
        self.num_people_entry.pack(pady=5)

        self.add_names_button = tk.Button(root, text="Thêm tên", command=self.create_name_entries)
        self.add_names_button.pack(pady=5)

        self.names_frame = tk.Frame(root)
        self.names_frame.pack(pady=5)

        self.start_button = tk.Button(root, text="Bắt đầu quét", command=self.start_scanning, state="disabled")
        self.start_button.pack(pady=10)

        self.stop_button = tk.Button(root, text="Dừng", command=self.stop_scanning, state="disabled")
        self.stop_button.pack(pady=10)

        self.status_label = tk.Label(root, text="Trạng thái: Chưa bắt đầu")
        self.status_label.pack(pady=10)

        self.count_label = tk.Label(root, text="Số khuôn mặt đã lưu: 0")
        self.count_label.pack(pady=10)

    def create_name_entries(self):
        try:
            num_people = int(self.num_people_entry.get().strip())
            if num_people <= 0:
                raise ValueError("Số người phải lớn hơn 0!")
        except ValueError:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập số người hợp lệ!")
            return

        # Xóa các ô nhập cũ (nếu có)
        for widget in self.names_frame.winfo_children():
            widget.destroy()
        self.name_entries.clear()
        self.people_names.clear()

        # Tạo ô nhập tên cho từng người
        for i in range(num_people):
            tk.Label(self.names_frame, text=f"Tên người {i+1}:").grid(row=i, column=0, pady=2)
            entry = tk.Entry(self.names_frame)
            entry.grid(row=i, column=1, pady=2)
            self.name_entries.append(entry)

        self.start_button.config(state="normal")

    def create_yolo_label(self, class_id, x, y, w, h, img_width, img_height):
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        width = w / img_width
        height = h / img_height
        return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

    def expand_bounding_box(self, x, y, w, h, img_width, img_height, scale_factor=1.5):
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        new_x = max(0, x - (new_w - w) // 2)
        new_y = max(0, y - (new_h - h) // 2)
        new_x = min(new_x, img_width - new_w)
        new_y = min(new_y, img_height - new_h)
        new_w = min(new_w, img_width - new_x)
        new_h = min(new_h, img_height - new_y)
        return new_x, new_y, new_w, new_h

    def create_classes_file(self):
        """Tạo file classes.txt chứa danh sách các class (tên người)"""
        classes_file = os.path.join(self.output_dir, "classes.txt")
        with open(classes_file, 'w', encoding='utf-8') as f:
            for name in self.people_names:
                f.write(f"{name}\n")
        print(f"Đã tạo file classes.txt tại: {classes_file}")

    def create_yaml_file(self):
        """Tạo file YAML cho YOLO dataset"""
        yaml_data = {
            'path': os.path.abspath(self.output_dir),  # Đường dẫn tuyệt đối đến thư mục dataset
            'train': os.path.join('train', 'images'),
            'val': os.path.join('valid', 'images'),
            'test': os.path.join('test', 'images'),
            'nc': len(self.people_names),  # Số lượng class
            'names': self.people_names  # Danh sách tên class
        }

        yaml_file = os.path.join(self.output_dir, "data.yaml")
        with open(yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, allow_unicode=True, sort_keys=False)
        print(f"Đã tạo file data.yaml tại: {yaml_file}")

    def save_file(self, frame, faces, img_width, img_height):
        rand = random.random()
        if rand < self.train_ratio:
            target_dir = self.train_dir
        elif rand < (self.train_ratio + self.val_ratio):
            target_dir = self.val_dir
        else:
            target_dir = self.test_dir

        img_filename = os.path.join(target_dir, "images", f"face_{self.face_count}.jpg")
        label_filename = os.path.join(target_dir, "labels", f"face_{self.face_count}.txt")

        cv2.imwrite(img_filename, frame)
        with open(label_filename, 'w') as f:
            for i, (x, y, w, h) in enumerate(faces):
                class_id = min(i, len(self.people_names) - 1)  # Đảm bảo không vượt quá số người
                new_x, new_y, new_w, new_h = self.expand_bounding_box(x, y, w, h, img_width, img_height)
                label = self.create_yolo_label(class_id, new_x, new_y, new_w, new_h, img_width, img_height)
                f.write(label + "\n")

        return img_filename, label_filename

    def scan_faces(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Lỗi", "Không thể truy cập webcam")
            self.stop_scanning()
            return

        self.status_label.config(text="Trạng thái: Đang quét...")
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img_height, img_width = frame.shape[:2]

            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            for i, (x, y, w, h) in enumerate(faces):
                new_x, new_y, new_w, new_h = self.expand_bounding_box(x, y, w, h, img_width, img_height)
                cv2.rectangle(frame, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 255, 0), 2)
                class_id = min(i, len(self.people_names) - 1)
                name = self.people_names[class_id] if self.people_names else f"Person {class_id}"
                cv2.putText(frame, name, (new_x, new_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.putText(frame, f"Faces saved: {self.face_count}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Scanner - Press S to save', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and len(faces) > 0:
                img_filename, label_filename = self.save_file(frame, faces, img_width, img_height)
                self.face_count += 1
                self.count_label.config(text=f"Số khuôn mặt đã lưu: {self.face_count}")
                print(f"Đã lưu: {img_filename} và {label_filename}")

        cap.release()
        cv2.destroyAllWindows()
        self.status_label.config(text="Trạng thái: Đã dừng")
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")

    def start_scanning(self):
        self.people_names = [entry.get().strip() for entry in self.name_entries]
        if not all(self.people_names):
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập đầy đủ tên cho tất cả người!")
            return

        # Tạo file classes.txt và data.yaml nếu có ít nhất 1 người
        if len(self.people_names) >= 1:
            self.create_classes_file()
            self.create_yaml_file()

        self.running = True
        self.scan_thread = threading.Thread(target=self.scan_faces)
        self.scan_thread.start()

    def stop_scanning(self):
        self.running = False

    def on_closing(self):
        if self.running:
            self.stop_scanning()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = FaceScannerApp(root, train_ratio=0.7, val_ratio=0.15)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()