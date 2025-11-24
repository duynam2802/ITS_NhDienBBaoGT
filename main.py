import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import threading
from ultralytics import YOLO
import os
import unicodedata

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
        self.class_names_vie = []
        self.class_names_vie_ascii = []
        self.load_model()
        
        # Biến điều khiển
        self.is_camera_active = False
        self.is_video_active = False
        self.cap = None
        self.video_path = None
        self.current_frame = None
        self.last_detected_vie = []
        
        # Tạo giao diện
        self.create_widgets()
        self.setup_styles()
        
    def load_model(self):
        """Tải mô hình YOLO"""
        try:
            self.model = YOLO('model/best.pt')
            print("Đã tải mô hình YOLO thành công!")
            self.load_class_names('classes_vie.txt')
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải mô hình YOLO: {str(e)}")
            self.model = None
    
    def strip_accents(self, s: str) -> str:
        nf = unicodedata.normalize('NFD', s)
        no_marks = ''.join(c for c in nf if unicodedata.category(c) != 'Mn')
        return no_marks.replace('Đ', 'D').replace('đ', 'd')

    def load_class_names(self, path):
        """Đọc tên lớp và tạo phiên bản không dấu."""
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    lines = [ln.strip() for ln in f if ln.strip()]
                self.class_names_vie = [name.replace('*', '').strip() for name in lines]
                self.class_names_vie_ascii = [self.strip_accents(n) for n in self.class_names_vie]
                print(f"Đã tải {len(self.class_names_vie)} lớp (không dấu).")
            else:
                print("Không tìm thấy file classes_vie.txt.")
        except Exception as e:
            print(f"Lỗi đọc file classes_vie.txt: {e}")
            self.class_names_vie = []
            self.class_names_vie_ascii = []

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
        
        # Nút dừng
        btn_stop = ttk.Button(button_frame,
                             text="⏹ Dừng",
                             command=self.stop_all,
                             style='Danger.TButton',
                             width=18)
        btn_stop.pack(side=tk.LEFT, padx=10)
        
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
        
        # Panel overlay (trong khung review)
        self.overlay_panel = tk.Label(video_display_frame,
                                      text="Chưa phát hiện",
                                      bg="#000000",
                                      fg="#ffffff",
                                      font=('Segoe UI', 10),
                                      justify=tk.LEFT,
                                      anchor=tk.NW,
                                      padx=8, pady=6,
                                      bd=0)
        self.overlay_panel.place(x=10, y=10)
        
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
            self.video_path = file_path
            self.is_video_active = True
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
    
    def start_camera(self):
        """Bắt đầu sử dụng camera"""
        if self.is_video_active:
            self.stop_all()
        
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
    
    def stop_all(self):
        """Dừng tất cả"""
        self.is_video_active = False
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
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Nhận diện biển báo
                frame = self.detect_traffic_signs(frame)
                
                # Hiển thị frame
                self.display_frame(frame)
                
                # Điều chỉnh tốc độ phát
                cv2.waitKey(delay)
            
            cap.release()
            self.is_video_active = False
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
    
    def draw_detected_panel(self, img, labels):
        """Vẽ panel danh sách biển báo đã phát hiện góc trên trái."""
        if not labels:
            panel_text = "Khong phat hien"
        else:
            panel_text = "Phat hien: " + ", ".join(labels)
        # Gói dòng nếu quá dài
        max_len = 60
        lines = []
        while len(panel_text) > max_len:
            cut = panel_text[:max_len]
            # tìm vị trí dấu phẩy gần nhất để ngắt
            if "," in cut:
                pos = cut.rfind(",")
                lines.append(panel_text[:pos+1])
                panel_text = panel_text[pos+2:].strip()
            else:
                lines.append(cut)
                panel_text = panel_text[max_len:].strip()
        lines.append(panel_text)

        x0, y0 = 10, 10
        pad_x, pad_y = 8, 8
        line_h = 0
        max_w = 0
        for ln in lines:
            (tw, th), _ = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            line_h = max(line_h, th)
            max_w = max(max_w, tw)
        panel_w = max_w + pad_x * 2
        panel_h = line_h * len(lines) + pad_y * 2 + (len(lines)-1)*4

        # Nền mờ
        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
        alpha = 0.35
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        # Viền
        cv2.rectangle(img, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 255, 0), 1)

        # Vẽ text
        y_text = y0 + pad_y + line_h
        for ln in lines:
            cv2.putText(img, ln, (x0 + pad_x, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            y_text += line_h + 4
        return img

    def update_overlay_panel(self, labels_vie):
        """Cập nhật panel bên trong khung video (giữ tiếng Việt có dấu)."""
        if not labels_vie:
            txt = "Đang quét...\nKhông phát hiện."
        else:
            # Gói dòng nếu quá dài
            prefix = "Đã phát hiện:"
            joined = ", ".join(labels_vie)
            full = f"{prefix} {joined}"
            lines = []
            max_len = 55
            while len(full) > max_len:
                cut = full[:max_len]
                if "," in cut:
                    pos = cut.rfind(",")
                    lines.append(full[:pos+1])
                    full = full[pos+2:].strip()
                else:
                    lines.append(cut)
                    full = full[max_len:].strip()
            lines.append(full)
            txt = "\n".join(lines)
        self.overlay_panel.config(text=txt)

    def detect_traffic_signs(self, frame):
        """Nhận diện và vẽ nhãn + panel tổng hợp (panel hiển thị tiếng Việt có dấu)."""
        if self.model is None:
            return frame
        try:
            results = self.model(frame, conf=0.25, verbose=False)
            detections = results[0].boxes
            annotated = frame.copy()
            class_names_ascii = []
            class_names_vie = []
            if len(detections) > 0:
                for box in detections:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    # Lấy không dấu cho vẽ lên ảnh
                    if self.class_names_vie_ascii and cls_id < len(self.class_names_vie_ascii):
                        label_ascii = self.class_names_vie_ascii[cls_id]
                    else:
                        label_ascii = self.model.names.get(cls_id, f"cls_{cls_id}")
                    # Lấy có dấu cho panel
                    if self.class_names_vie and cls_id < len(self.class_names_vie):
                        label_vie = self.class_names_vie[cls_id]
                    else:
                        label_vie = label_ascii
                    class_names_ascii.append(label_ascii)
                    class_names_vie.append(label_vie)
                    text = f"{label_ascii} {conf:.2f}"
                    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated, (int(x1), int(y1)-th-8), (int(x1)+tw+4, int(y1)), (0, 255, 0), -1)
                    cv2.putText(annotated, text, (int(x1)+2, int(y1)-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                # Lọc duy nhất theo thứ tự
                seen_vie = set()
                unique_vie = []
                for n in class_names_vie:
                    if n not in seen_vie:
                        seen_vie.add(n)
                        unique_vie.append(n)
                self.last_detected_vie = unique_vie
                info_text = f"Phát hiện {len(detections)} đối tượng: {', '.join(unique_vie)}"
                self.info_label.config(text=info_text, fg=self.colors['success'])
                self.update_overlay_panel(unique_vie)
            else:
                self.last_detected_vie = []
                self.info_label.config(text="Đang quét... Không phát hiện biển báo",
                                       fg=self.colors['text_secondary'])
                self.update_overlay_panel([])
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

