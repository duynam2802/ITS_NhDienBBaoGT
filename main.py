import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading
from ultralytics import YOLO
import os
import unicodedata
import time
from collections import defaultdict
import numpy as np

class TrafficSignDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🚦 Ứng dụng Nhận diện Biển báo Giao thông")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1e1e2e')
        
        # Màu sắc theme
        self.colors = {
            'bg': '#1e1e2e',
            'bg_secondary': '#2d2d44',
            'bg_card': '#3a3a5c',
            'primary': '#4a90e2',
            'primary_hover': '#5ba0f2',
            'success': '#50c878',
            'danger': '#e74c3c',
            'warning': '#f39c12',
            'text': '#ffffff',
            'text_secondary': '#b0b0b0',
            'border': '#4a4a6a'
        }
        
        # Khởi tạo YOLO model
        self.model = None
        self.load_model()
        
        # Biến điều khiển
        self.is_camera_active = False
        self.is_video_active = False
        self.is_paused = False
        self.cap = None
        self.video_path = None
        self.current_frame = None
        self.detected_history = []  # Lưu lịch sử các biển báo (list để giữ thứ tự)
        
        # Cơ chế ổn định kết quả (stabilization)
        self.detection_buffer = defaultdict(list)  # {label: [timestamps]}
        self.stable_duration = 0.5  # Thời gian ổn định để xác nhận detection (0.5 giây)
        self.buffer_timeout = 2.0  # Xóa buffer sau 2s không phát hiện
        
        # Quản lý hiển thị log và ảnh biển báo
        self.show_log = True  # Bật/tắt log
        self.sign_images = {}  # {label: {'image': cropped_img, 'first_stable': timestamp, 'last_seen': timestamp, 'widget': frame_widget}}
        self.sign_popup_text = {}  # {label: {'text': name_vie, 'first_stable': timestamp, 'last_seen': timestamp}}
        self.display_duration = 2.0  # Thời gian hiển thị sau khi mất (2 giây)
        self.capture_delay = 2.0  # Thời gian chờ trước khi hiển thị ảnh/popup (2 giây từ lúc ổn định)
        
        # Tải danh sách các lớp từ file classes_vie.txt
        self.class_names_vie = self.read_classes_file('classes_vie.txt')
        self.class_labels = self.read_classes_file('label.txt')  # Đọc file label
        
        # Tạo giao diện
        self.create_widgets()
        self.setup_styles()
        
    def load_model(self):
        """Tải mô hình YOLO"""
        try:
            self.model = YOLO('model/bestv1.pt')
            print("Đã tải mô hình YOLO thành công!")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải mô hình YOLO: {str(e)}")
            self.model = None
    
    def strip_accents(self, s: str) -> str:
        nf = unicodedata.normalize('NFD', s)
        no_marks = ''.join(c for c in nf if unicodedata.category(c) != 'Mn')
        return no_marks.replace('Đ', 'D').replace('đ', 'd')

    def read_classes_file(self, file_path):
        """Đọc file classes_vie.txt và trả về danh sách các lớp"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                classes_vie = [line.strip() for line in f if line.strip()]
            return classes_vie
        except FileNotFoundError:
            print(f"Không tìm thấy file: {file_path}")
            return []
        except Exception as e:
            print(f"Lỗi khi đọc file {file_path}: {e}")
            return []
    
    def setup_styles(self):
        """Thiết lập style cho các widget"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Cấu hình style cho buttons
        style.configure('Primary.TButton',
                       background=self.colors['primary'],
                       foreground=self.colors['text'],
                       borderwidth=0,
                       focuscolor='none',
                       padding=15,
                       font=('Segoe UI', 11, 'bold'))
        style.map('Primary.TButton',
                 background=[('active', self.colors['primary_hover']),
                           ('pressed', self.colors['primary'])])
        
        style.configure('Success.TButton',
                       background=self.colors['success'],
                       foreground=self.colors['text'],
                       borderwidth=0,
                       focuscolor='none',
                       padding=15,
                       font=('Segoe UI', 11, 'bold'))
        style.map('Success.TButton',
                 background=[('active', '#60d888'),
                           ('pressed', self.colors['success'])])
        
        style.configure('Danger.TButton',
                       background=self.colors['danger'],
                       foreground=self.colors['text'],
                       borderwidth=0,
                       focuscolor='none',
                       padding=15,
                       font=('Segoe UI', 11, 'bold'))
        style.map('Danger.TButton',
                 background=[('active', '#f75c4c'),
                           ('pressed', self.colors['danger'])])
        
        # Style cho LabelFrame
        style.configure('Card.TLabelframe',
                       background=self.colors['bg_card'],
                       foreground=self.colors['text'],
                       borderwidth=2,
                       relief='flat')
        style.configure('Card.TLabelframe.Label',
                       background=self.colors['bg_card'],
                       foreground=self.colors['text'],
                       font=('Segoe UI', 12, 'bold'))
    
    def create_widgets(self):
        """Tạo giao diện người dùng"""
        # Header
        header_frame = tk.Frame(self.root, bg=self.colors['bg_secondary'], height=80)
        header_frame.pack(fill=tk.X, padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame,
                               text="🚦 NHẬN DIỆN BIỂN BÁO GIAO THÔNG",
                               font=('Segoe UI', 20, 'bold'),
                               bg=self.colors['bg_secondary'],
                               fg=self.colors['text'])
        title_label.pack(pady=20)
        
        # Frame chính
        main_frame = tk.Frame(self.root, bg=self.colors['bg'], padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Frame điều khiển với card style
        control_card = tk.Frame(main_frame, bg=self.colors['bg_card'], relief=tk.FLAT, bd=0)
        control_card.pack(fill=tk.X, pady=(0, 20))
        
        control_inner = tk.Frame(control_card, bg=self.colors['bg_card'], padx=20, pady=20)
        control_inner.pack(fill=tk.X)
        
        control_title = tk.Label(control_inner,
                                text="🎮 Điều khiển",
                                font=('Segoe UI', 14, 'bold'),
                                bg=self.colors['bg_card'],
                                fg=self.colors['text'])
        control_title.pack(anchor=tk.W, pady=(0, 15))
        
        button_frame = tk.Frame(control_inner, bg=self.colors['bg_card'])
        button_frame.pack(fill=tk.X)
        
        # Nút chọn video
        btn_video = ttk.Button(button_frame,
                              text="📹 Chọn Video",
                              command=self.select_video,
                              style='Primary.TButton',
                              width=18)
        btn_video.pack(side=tk.LEFT, padx=10)
        
        # Nút bật/tắt camera
        self.btn_camera = ttk.Button(button_frame,
                                     text="📷 Bật Camera",
                                     command=self.toggle_camera,
                                     style='Success.TButton',
                                     width=18)
        self.btn_camera.pack(side=tk.LEFT, padx=10)
        
        # Nút pause/resume video
        self.btn_pause = ttk.Button(button_frame,
                                    text="⏸ Pause",
                                    command=self.toggle_pause,
                                    style='Primary.TButton',
                                    width=18)
        self.btn_pause.pack(side=tk.LEFT, padx=10)
        self.btn_pause.config(state='disabled')
        
        # Nút dừng
        btn_stop = ttk.Button(button_frame,
                             text="⏹ Dừng",
                             command=self.stop_all,
                             style='Danger.TButton',
                             width=18)
        btn_stop.pack(side=tk.LEFT, padx=10)
        
        # Nút bật/tắt log
        self.btn_toggle_log = ttk.Button(button_frame,
                                        text="📋 Tắt Log",
                                        command=self.toggle_log,
                                        style='Primary.TButton',
                                        width=18)
        self.btn_toggle_log.pack(side=tk.LEFT, padx=10)
        
        # Frame hiển thị video với card style
        video_card = tk.Frame(main_frame, bg=self.colors['bg_card'], relief=tk.FLAT, bd=0)
        video_card.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        video_inner = tk.Frame(video_card, bg=self.colors['bg_card'], padx=15, pady=15)
        video_inner.pack(fill=tk.BOTH, expand=True)
        
        video_title = tk.Label(video_inner,
                              text="📺 Video Preview",
                              font=('Segoe UI', 14, 'bold'),
                              bg=self.colors['bg_card'],
                              fg=self.colors['text'])
        video_title.pack(anchor=tk.W, pady=(0, 10))
        
        # Label hiển thị video với border
        video_display_frame = tk.Frame(video_inner, bg=self.colors['border'], padx=3, pady=3)
        video_display_frame.pack(fill=tk.BOTH, expand=True)
        
        self.video_label = tk.Label(video_display_frame,
                                    text="Chưa có video\n\nChọn video hoặc bật camera để bắt đầu",
                                    background="#000000",
                                    foreground=self.colors['text_secondary'],
                                    font=('Segoe UI', 12),
                                    anchor=tk.CENTER,
                                    justify=tk.CENTER)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Panel hiển thị ảnh biển báo đã nhận diện (góc trên trái)
        self.sign_images_panel = tk.Frame(video_display_frame,
                                         bg="#1a1a1a",
                                         bd=2,
                                         relief=tk.SOLID)
        self.sign_images_panel.place(x=10, y=10)
        
        # Label tiêu đề cho panel ảnh
        self.sign_images_title = tk.Label(self.sign_images_panel,
                                         text="📸 Biển báo đã phát hiện",
                                         bg="#1a1a1a",
                                         fg="#00ff00",
                                         font=('Courier New', 9, 'bold'),
                                         padx=5, pady=3)
        self.sign_images_title.pack()
        
        # Frame chứa các ảnh biển báo
        self.sign_images_container = tk.Frame(self.sign_images_panel, bg="#1a1a1a")
        self.sign_images_container.pack(padx=5, pady=5)
        
        # Panel overlay log biển báo (góc dưới trái)
        self.overlay_panel = tk.Label(video_display_frame,
                                      text="Log: Chưa phát hiện",
                                      bg="#1a1a1a",
                                      fg="#00ff00",
                                      font=('Courier New', 9, 'bold'),
                                      justify=tk.LEFT,
                                      anchor=tk.SW,
                                      padx=10, pady=8,
                                      bd=1,
                                      relief=tk.SOLID,
                                      borderwidth=1)
        # Đặt ở góc dưới trái (sẽ cập nhật vị trí động sau)
        self.overlay_panel.place(x=10, rely=1.0, y=-10, anchor=tk.SW)
        
        # Frame thông tin với card style
        info_card = tk.Frame(main_frame, bg=self.colors['bg_card'], relief=tk.FLAT, bd=0)
        info_card.pack(fill=tk.X, pady=(0, 10))
        
        info_inner = tk.Frame(info_card, bg=self.colors['bg_card'], padx=20, pady=15)
        info_inner.pack(fill=tk.X)
        
        info_title = tk.Label(info_inner,
                             text="ℹ️ Thông tin phát hiện",
                             font=('Segoe UI', 14, 'bold'),
                             bg=self.colors['bg_card'],
                             fg=self.colors['text'])
        info_title.pack(anchor=tk.W, pady=(0, 10))
        
        self.info_label = tk.Label(info_inner,
                                   text="Sẵn sàng. Chọn video hoặc bật camera để bắt đầu.",
                                   font=('Segoe UI', 11),
                                   bg=self.colors['bg_card'],
                                   fg=self.colors['text_secondary'],
                                   anchor=tk.W,
                                   justify=tk.LEFT)
        self.info_label.pack(fill=tk.X)
        
        # Status bar
        status_frame = tk.Frame(self.root, bg=self.colors['bg_secondary'], height=40)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)
        
        self.status_indicator = tk.Label(status_frame,
                                        text="●",
                                        font=('Segoe UI', 14),
                                        bg=self.colors['bg_secondary'],
                                        fg=self.colors['text_secondary'])
        self.status_indicator.pack(side=tk.LEFT, padx=(20, 10))
        
        self.status_label = tk.Label(status_frame,
                                     text="Trạng thái: Chờ",
                                     font=('Segoe UI', 10),
                                     bg=self.colors['bg_secondary'],
                                     fg=self.colors['text_secondary'])
        self.status_label.pack(side=tk.LEFT)
    
    def select_video(self):
        """Chọn file video"""
        if self.is_camera_active:
            self.stop_all()
        
        file_path = filedialog.askopenfilename(
            title="Chọn file video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            # Reset log khi chạy video mới
            self.detected_history.clear()
            self.detection_buffer.clear()
            self.update_detection_log()
            
            self.video_path = file_path
            self.is_video_active = True
            self.is_paused = False
            self.btn_pause.config(state='normal', text="⏸ Pause")
            self.status_label.config(text=f"Trạng thái: Đang xử lý video - {os.path.basename(file_path)}", 
                                   fg=self.colors['primary'])
            self.status_indicator.config(fg=self.colors['primary'])
            self.process_video()
    
    def toggle_camera(self):
        """Bật/tắt camera"""
        if self.is_camera_active:
            self.stop_camera()
        else:
            self.start_camera()
    
    def toggle_pause(self):
        """Pause/Resume video"""
        if not self.is_video_active:
            return
        
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.btn_pause.config(text="▶ Resume")
            self.status_label.config(text="Trạng thái: Video đã tạm dừng", 
                                   fg=self.colors['warning'])
        else:
            self.btn_pause.config(text="⏸ Pause")
            self.status_label.config(text=f"Trạng thái: Đang xử lý video - {os.path.basename(self.video_path)}", 
                                   fg=self.colors['primary'])
    
    def toggle_log(self):
        """Bật/tắt hiển thị log"""
        self.show_log = not self.show_log
        if self.show_log:
            self.btn_toggle_log.config(text="📋 Tắt Log")
            self.overlay_panel.place(x=10, rely=1.0, y=-10, anchor=tk.SW)
        else:
            self.btn_toggle_log.config(text="📋 Bật Log")
            self.overlay_panel.place_forget()
    
    def start_camera(self):
        """Bắt đầu sử dụng camera"""
        if self.is_video_active:
            self.stop_all()
        
        # Reset buffer khi bật camera
        self.detection_buffer.clear()
        self.detected_history.clear()
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Lỗi", "Không thể mở camera!")
            return
        
        self.is_camera_active = True
        self.btn_camera.config(text="📷 Tắt Camera")
        self.status_label.config(text="Trạng thái: Đang sử dụng camera", 
                               fg=self.colors['success'])
        self.status_indicator.config(fg=self.colors['success'])
        self.process_camera()
    
    def stop_camera(self):
        """Dừng camera"""
        self.is_camera_active = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_camera.config(text="📷 Bật Camera")
        self.status_label.config(text="Trạng thái: Đã dừng camera", 
                               fg=self.colors['text_secondary'])
        self.status_indicator.config(fg=self.colors['text_secondary'])
        self.video_label.config(image='', 
                               text="Chưa có video\n\nChọn video hoặc bật camera để bắt đầu",
                               fg=self.colors['text_secondary'])
        self.detected_history.clear()
        self.detection_buffer.clear()
        self.sign_images.clear()
        self.sign_popup_text.clear()
        self.update_detection_log()
        self.update_sign_images_display()
    
    def stop_all(self):
        """Dừng tất cả"""
        self.is_video_active = False
        self.is_paused = False
        self.btn_pause.config(state='disabled', text="⏸ Pause")
        self.stop_camera()
        self.status_label.config(text="Trạng thái: Đã dừng", 
                               fg=self.colors['text_secondary'])
        self.status_indicator.config(fg=self.colors['text_secondary'])
    
    def process_video(self):
        """Xử lý video file"""
        if not self.model:
            messagebox.showerror("Lỗi", "Mô hình YOLO chưa được tải!")
            return
        
        def video_thread():
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                messagebox.showerror("Lỗi", "Không thể mở file video!")
                self.is_video_active = False
                return
            
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            delay = int(1000 / fps) if fps > 0 else 30
            
            while self.is_video_active:
                if not self.is_paused:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Nhận diện biển báo
                    frame = self.detect_traffic_signs(frame)
                    
                    # Hiển thị frame
                    self.display_frame(frame)
                    
                    # Điều chỉnh tốc độ phát
                    cv2.waitKey(delay)
                else:
                    # Khi pause, chỉ đợi một chút
                    cv2.waitKey(100)
            
            cap.release()
            self.is_video_active = False
            self.is_paused = False
            self.btn_pause.config(state='disabled', text="⏸ Pause")
            self.status_label.config(text="Trạng thái: Video đã kết thúc", 
                                   fg=self.colors['text_secondary'])
            self.status_indicator.config(fg=self.colors['text_secondary'])
        
        thread = threading.Thread(target=video_thread, daemon=True)
        thread.start()
    
    def process_camera(self):
        """Xử lý camera"""
        if not self.model:
            messagebox.showerror("Lỗi", "Mô hình YOLO chưa được tải!")
            return
        
        def camera_thread():
            while self.is_camera_active and self.cap:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Nhận diện biển báo
                frame = self.detect_traffic_signs(frame)
                
                # Hiển thị frame
                self.display_frame(frame)
            
            if self.cap:
                self.cap.release()
        
        thread = threading.Thread(target=camera_thread, daemon=True)
        thread.start()
    
    def update_detection_log(self):
        """Cập nhật log biển báo đã phát hiện (biển mới nhất ở đầu)"""
        classesVie = self.read_classes_file('classes_vie.txt')
        if not self.detected_history:
            log_text = "Log: Chưa phát hiện"
        else:
            log_lines = ["=== LOG BIỂN BÁO ==="]
            # Hiển thị theo thứ tự ngược (mới nhất ở đầu)
            for sign in self.detected_history:
                log_lines.append(f"✓ {sign} {classesVie[int(sign)]}")
            log_text = "\n".join(log_lines)
        self.overlay_panel.config(text=log_text)
    
    def is_detection_stable(self, label):
        """
        Kiểm tra xem một detection có ổn định hay không
        Chỉ trả về True nếu label được phát hiện liên tục trong stable_duration giây
        """
        current_time = time.time()
        timestamps = self.detection_buffer[label]
        
        # Lọc bỏ các timestamp cũ (ngoài buffer_timeout)
        timestamps = [t for t in timestamps if current_time - t < self.buffer_timeout]
        self.detection_buffer[label] = timestamps
        
        if not timestamps:
            return False
        
        # Kiểm tra khoảng thời gian từ lần phát hiện đầu đến lần cuối
        time_span = current_time - timestamps[0]
        
        # Ổn định nếu: đã phát hiện liên tục >= stable_duration
        return time_span >= self.stable_duration
    
    def add_detection_to_buffer(self, label):
        """Thêm detection vào buffer với timestamp hiện tại"""
        current_time = time.time()
        self.detection_buffer[label].append(current_time)
    
    def update_sign_images_display(self):
        """Cập nhật hiển thị các ảnh biển báo đã phát hiện - CHỈ VẼ MỘT LẦN"""
        current_time = time.time()
        labels_to_remove = []
        
        # Kiểm tra và xóa các ảnh đã hết thời gian
        for label, data in list(self.sign_images.items()):
            # Kiểm tra nếu quá 2s kể từ lần cuối nhìn thấy
            if current_time - data['last_seen'] > self.display_duration:
                # Xóa widget nếu có
                if 'widget' in data and data['widget']:
                    data['widget'].destroy()
                labels_to_remove.append(label)
                continue
            
            # Kiểm tra nếu chưa đủ 2s từ lần đầu ổn định
            if current_time - data['first_stable'] < self.capture_delay:
                continue
            
            # Nếu widget chưa được tạo, tạo mới
            if 'widget' not in data or data['widget'] is None:
                try:
                    # Tạo frame cho mỗi ảnh
                    img_frame = tk.Frame(self.sign_images_container, bg="#1a1a1a", bd=1, relief=tk.SOLID)
                    img_frame.pack(side=tk.LEFT, padx=3, pady=3)
                    
                    # Chuyển đổi ảnh OpenCV sang PIL
                    img_rgb = cv2.cvtColor(data['image'], cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                    
                    # Resize ảnh nhỏ lại
                    max_size = 80
                    img_pil.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                    
                    # Chuyển sang PhotoImage
                    photo = ImageTk.PhotoImage(img_pil)
                    
                    # Label hiển thị ảnh
                    img_label = tk.Label(img_frame, image=photo, bg="#1a1a1a")
                    img_label.image = photo  # Giữ reference
                    img_label.pack()
                    
                    # Label tên biển báo
                    classesVie = self.read_classes_file('classes_vie.txt')
                    if classesVie and int(label) < len(classesVie):
                        name_vie = classesVie[int(label)]
                    else:
                        name_vie = label
                    
                    name_label = tk.Label(img_frame, 
                                        text=label,
                                        bg="#1a1a1a",
                                        fg="#00ff00",
                                        font=('Courier New', 8, 'bold'))
                    name_label.pack()
                    
                    # Lưu widget vào data
                    data['widget'] = img_frame
                    
                except Exception as e:
                    print(f"Lỗi hiển thị ảnh biển báo {label}: {e}")
        
        # Xóa các ảnh đã hết thời gian hiển thị
        for label in labels_to_remove:
            del self.sign_images[label]
    
    def crop_sign_image(self, frame, box):
        """Cắt ảnh biển báo từ frame"""
        try:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Đảm bảo tọa độ trong phạm vi frame
            h, w = frame.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            # Cắt ảnh
            cropped = frame[y1:y2, x1:x2].copy()
            return cropped
        except Exception as e:
            print(f"Lỗi khi cắt ảnh: {e}")
            return None
    
    def get_sign_color(self, label_index):
        """
        Lấy màu theo loại biển báo dựa trên ký tự đầu tiên của label
        P: Red (Prohibitory - Cấm)
        W: Orange (Warning - Cảnh báo)
        R: Light Blue (Regulatory - Chỉ dẫn)
        I: Blue (Information - Thông tin)
        """
        try:
            if self.class_labels and int(label_index) < len(self.class_labels):
                label_code = self.class_labels[int(label_index)]
                first_char = label_code[0].upper()
                
                if first_char == 'P':
                    return (220, 20, 60)  # Red - Crimson
                elif first_char == 'W':
                    return (255, 140, 0)  # Orange
                elif first_char == 'R':
                    return (135, 206, 250)  # Light Blue
                elif first_char == 'I':
                    return (30, 144, 255)  # Dodger Blue
                else:
                    return (0, 200, 0)  # Default Green
            else:
                return (0, 200, 0)  # Default Green
        except:
            return (0, 200, 0)  # Default Green
    
    def draw_popup_notifications(self, frame):
        """Vẽ popup thông báo tên biển báo trên video với hỗ trợ font tiếng Việt"""
        current_time = time.time()
        labels_to_remove = []
        
        # Chuyển frame sang PIL Image để vẽ text tiếng Việt
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_img)
        
        # Vị trí bắt đầu vẽ popup (từ trên xuống, dời xuống 20px)
        y_offset = 130
        
        # Tính font size động dựa trên chiều rộng frame để đồng nhất
        frame_h, frame_w = pil_img.size[1], pil_img.size[0]
        # Font size = 4% chiều rộng frame (tối thiểu 30, tối đa 80)
        dynamic_font_size = max(30, min(80, int(frame_w * 0.04)))
        
        # Thử tải font tiếng Việt, nếu không có dùng font mặc định
        try:
            font = ImageFont.truetype("arial.ttf", dynamic_font_size)
        except:
            try:
                font = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", dynamic_font_size)
            except:
                font = ImageFont.load_default()
        
        for label, data in self.sign_popup_text.items():
            # Kiểm tra nếu chưa đủ 2s từ lần đầu ổn định, bỏ qua
            if current_time - data['first_stable'] < self.capture_delay:
                continue
            
            # Kiểm tra nếu quá 2s kể từ lần cuối nhìn thấy
            if current_time - data['last_seen'] > self.display_duration:
                labels_to_remove.append(label)
                continue
            
            # Text hiển thị
            text = f"🚦 {data['text']}"
            
            # Lấy màu theo loại biển báo
            bg_color = self.get_sign_color(label)
            
            # Tính kích thước text
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Vị trí hiển thị (giữa màn hình, từ trên xuống)
            frame_h, frame_w = pil_img.size[1], pil_img.size[0]
            x = (frame_w - text_width) // 2
            y = y_offset
            
            # Vẽ nền cho text với màu theo loại biển báo
            padding = 20
            draw.rectangle(
                [(x - padding, y - padding),
                 (x + text_width + padding, y + text_height + padding)],
                fill=bg_color,
                outline=(255, 255, 255),
                width=4
            )
            
            # Vẽ text
            draw.text((x, y), text, font=font, fill=(255, 255, 255))
            
            y_offset += text_height + 2 * padding + 15
        
        # Xóa các popup đã hết thời gian
        for label in labels_to_remove:
            del self.sign_popup_text[label]
        
        # Chuyển PIL Image về OpenCV format
        frame_result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return frame_result

    def detect_traffic_signs(self, frame):
        """Nhận diện biển báo với cơ chế ổn định kết quả"""
        if self.model is None:
            return frame
        try:
            results = self.model(frame, conf=0.25, verbose=False)
            detections = results[0].boxes
            annotated = frame.copy()
            
            current_time = time.time()
            detected_labels_this_frame = set()
            
            if len(detections) > 0:
                current_signs = []
                stable_signs = []  # Các biển đã ổn định
                
                for box in detections:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Lấy tên trực tiếp từ model
                    label = self.model.names.get(cls_id, f"cls_{cls_id}")
                    current_signs.append(label)
                    detected_labels_this_frame.add(label)
                    
                    # Thêm vào buffer
                    self.add_detection_to_buffer(label)
                    
                    # Kiểm tra xem detection có ổn định chưa
                    is_stable = self.is_detection_stable(label)
                    
                    # Chỉ thêm vào history nếu đã ổn định
                    if is_stable:
                        if label not in self.detected_history:
                            self.detected_history.insert(0, label)
                            
                            # Chụp ảnh biển báo 1 LẦN DUY NHẤT khi lần đầu ổn định
                            cropped_img = self.crop_sign_image(frame, (x1, y1, x2, y2))
                            if cropped_img is not None:
                                self.sign_images[label] = {
                                    'image': cropped_img,
                                    'first_stable': current_time,
                                    'last_seen': current_time,
                                    'widget': None  # Widget sẽ được tạo sau
                                }
                            
                            # Thêm popup text với first_stable timestamp
                            classesVie = self.read_classes_file('classes_vie.txt')
                            if classesVie and int(label) < len(classesVie):
                                name_vie = classesVie[int(label)]
                            else:
                                name_vie = label
                            
                            self.sign_popup_text[label] = {
                                'text': name_vie,
                                'first_stable': current_time,
                                'last_seen': current_time
                            }
                        else:
                            # Chỉ cập nhật thời gian last_seen, KHÔNG cập nhật ảnh
                            if label in self.sign_images:
                                self.sign_images[label]['last_seen'] = current_time
                            if label in self.sign_popup_text:
                                self.sign_popup_text[label]['last_seen'] = current_time
                        
                        stable_signs.append(label)
                    
                    # Vẽ bounding box (màu khác nhau cho stable/unstable)
                    color = (0, 255, 0) if is_stable else (0, 165, 255)  # Xanh lá nếu stable, cam nếu chưa
                    status = "✓" if is_stable else "..."
                    text = f"{status} {label} {conf:.2f}"
                    
                    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated, (int(x1), int(y1)-th-8), (int(x1)+tw+4, int(y1)), color, -1)
                    cv2.putText(annotated, text, (int(x1)+2, int(y1)-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                
                # Cập nhật thông tin
                unique_current = list(dict.fromkeys(current_signs))
                if stable_signs:
                    info_text = f"✅ Phát hiện ổn định: {', '.join(list(dict.fromkeys(stable_signs)))} | Đang phát hiện: {len(detections)}"
                else:
                    info_text = f"🔄 Đang xác nhận... ({len(detections)} đối tượng)"
                self.info_label.config(text=info_text, fg=self.colors['success'])
                
                # Cập nhật log chỉ với các detection ổn định
                self.update_detection_log()
            else:
                self.info_label.config(text="🔍 Đang quét... Không phát hiện biển báo",
                                       fg=self.colors['text_secondary'])
            
            # Xóa các buffer không còn được phát hiện (sau buffer_timeout)
            labels_to_remove = []
            for label in self.detection_buffer:
                if label not in detected_labels_this_frame:
                    # Lọc timestamps cũ
                    timestamps = [t for t in self.detection_buffer[label] 
                                if current_time - t < self.buffer_timeout]
                    if not timestamps:
                        labels_to_remove.append(label)
                    else:
                        self.detection_buffer[label] = timestamps
            
            for label in labels_to_remove:
                del self.detection_buffer[label]
            
            # Vẽ popup thông báo
            annotated = self.draw_popup_notifications(annotated)
            
            # Cập nhật hiển thị ảnh biển báo
            self.update_sign_images_display()
            
            return annotated
        except Exception as e:
            print(f"Lỗi khi nhận diện: {str(e)}")
            return frame
    
    def display_frame(self, frame):
        """Hiển thị frame lên GUI"""
        try:
            # Chuyển đổi BGR sang RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize frame để vừa với cửa sổ
            height, width = frame_rgb.shape[:2]
            max_width = 1200
            max_height = 600
            
            if width > max_width or height > max_height:
                scale = min(max_width / width, max_height / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
            
            # Chuyển đổi sang PIL Image
            image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image=image)
            
            # Cập nhật label
            self.video_label.config(image=photo, text="", bg="#000000")
            self.video_label.image = photo  # Giữ reference
            
        except Exception as e:
            print(f"Lỗi khi hiển thị frame: {str(e)}")

def main():
    root = tk.Tk()
    app = TrafficSignDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

