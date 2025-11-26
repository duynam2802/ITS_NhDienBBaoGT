# Ứng dụng Nhận diện Biển báo Giao thông

Ứng dụng Python sử dụng YOLO để nhận diện biển báo giao thông từ camera hoặc video file.

## Tính năng

- Nhận diện biển báo giao thông sử dụng YOLO (YOLOv8)
- Hỗ trợ camera real-time
- Hỗ trợ video file (mp4, avi, mov, mkv, flv, wmv)
- Hiển thị video với detection overlay

## Yêu cầu

- Python 3.8 trở lên
- Webcam (nếu sử dụng chức năng camera)

## Cài đặt

1. **Clone repository (nếu có):**

```bash
git clone <repository-url>
cd ITS_CUOIKY
```

2. **Cài đặt các thư viện cần thiết:**

```bash
pip install -r requirements.txt
```

**Hoặc cài đặt từng thư viện:**

```bash
# YOLOv8 và xử lý hình ảnh
pip install ultralytics>=8.0.0
pip install opencv-python>=4.8.0
pip install Pillow>=10.0.0
pip install numpy>=1.24.0

# PyTorch (CPU hoặc GPU)
pip install torch>=2.0.0
pip install torchvision>=0.15.0

# Text-to-Speech và phát âm thanh
pip install gTTS>=2.3.0
pip install pygame>=2.5.0
```

**Lưu ý:** 
- Nếu sử dụng GPU NVIDIA, cài đặt PyTorch với CUDA support từ https://pytorch.org/
- Đảm bảo có kết nối internet khi chạy lần đầu để tải mô hình YOLO

3. **Chạy ứng dụng:**

```bash
python main.py
```

## Hướng dẫn sử dụng

1. **Sử dụng Camera:**
   - Click nút "Bật Camera" để bắt đầu nhận diện từ webcam
   - Click "Tắt Camera" để dừng

2. **Sử dụng Video File:**
   - Click nút "Chọn Video" để chọn file video từ máy tính
   - Video sẽ tự động phát và nhận diện biển báo

3. **Dừng:**
   - Click nút "Dừng" để dừng tất cả các chức năng


## Cấu trúc thư mục

```
ITS_CUOIKY/
├── main.py              # File chính của ứng dụng
├── requirements.txt     # Danh sách thư viện cần thiết
└── README.md           # File hướng dẫn này

## Xử lý lỗi

Nếu gặp lỗi khi tải mô hình YOLO:
- Kiểm tra kết nối internet (lần đầu tải model)
- Đảm bảo đã cài đặt đầy đủ requirements.txt

Nếu camera không hoạt động:
- Kiểm tra camera đã được kết nối
- Đảm bảo không có ứng dụng khác đang sử dụng camera

