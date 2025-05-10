"""
Hệ thống nhận diện biển số xe từ video
"""

import math
import cv2
import numpy as np
import PreProcess  # Module xử lý ảnh đầu vào

# Các tham số cho việc xác định biển số
ADAPTIVE_THRESH_BLOCK_SIZE = 19  # Kích thước khối trong adaptive threshold
ADAPTIVE_THRESH_WEIGHT = 9  # Giá trị điều chỉnh trong adaptive threshold

# Tham số cho việc xác định ký tự biển số
MIN_CHAR_AREA_RATIO = 0.015  # Diện tích nhỏ nhất của 1 ký tự (theo tỷ lệ ảnh)
MAX_CHAR_AREA_RATIO = 0.06  # Diện tích lớn nhất của 1 ký tự
MIN_CHAR_RATIO = 0.01  # Ngưỡng diện tích ký tự nhỏ nhất
MAX_CHAR_RATIO = 0.09  # Ngưỡng diện tích ký tự lớn nhất

# Tỷ lệ chiều rộng/chiều cao cho ký tự
MIN_CHAR_ASPECT_RATIO = 0.25  # Tỷ lệ chiều rộng/chiều cao nhỏ nhất cho ký tự
MAX_CHAR_ASPECT_RATIO = 0.7   # Tỷ lệ chiều rộng/chiều cao lớn nhất

# Kích thước chuẩn hoá cho ký tự nhận dạng
RESIZED_CHAR_WIDTH = 20   # Kích thước chuẩn hoá ký tự chiều rộng
RESIZED_CHAR_HEIGHT = 30  # Kích thước chuẩn hoá ký tự chiều cao

# Biến thống kê
total_frames = 0        # Tổng số khung hình đã xử lý
plates_found = 0        # Tổng số biển số đã phát hiện được

def load_knn_model():
    """
    Tải mô hình KNN đã được huấn luyện từ file
    
    Returns:
        cv2.ml.KNearest: Mô hình KNN đã huấn luyện
    """
    # Đọc dữ liệu huấn luyện
    npaClassifications = np.loadtxt("classifications.txt", np.float32)
    npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
    
    # Định dạng dữ liệu
    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))
    
    # Khởi tạo và huấn luyện mô hình KNN
    knn = cv2.ml.KNearest_create()
    knn.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)
    
    return knn

def detect_license_plates(img, imgGrayscale, imgThreshold):
    """
    Phát hiện các biển số tiềm năng trong ảnh
    
    Args:
        img: Ảnh gốc
        imgGrayscale: Ảnh grayscale
        imgThreshold: Ảnh nhị phân
        
    Returns:
        list: Danh sách các contour biển số tiềm năng
    """
    # Áp dụng thuật toán Canny để phát hiện biên
    canny_image = cv2.Canny(imgThreshold, 250, 255)
    
    # Phép giãn nở để kết nối các đường biên
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(canny_image, kernel, iterations=1)
    
    # Tìm contours trong ảnh sau khi xử lý biên
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sắp xếp contours theo diện tích, lấy top 10 lớn nhất
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    # Danh sách lưu các contour có thể là biển số
    potential_plates = []
    
    # Duyệt qua các contour để tìm biển số có 4 góc
    for c in contours:
        # Xấp xỉ đa giác
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)
        
        [x, y, w, h] = cv2.boundingRect(approx.copy())
        aspect_ratio = w / h
        
        # Biển số có hình chữ nhật: tỉ lệ w/h nằm trong khoảng hợp lý
        # - Biển số vuông: 0.8 <= ratio <= 1.5
        # - Biển số dài: 4.5 <= ratio <= 6.5
        if len(approx) == 4 and (0.8 <= aspect_ratio <= 1.5 or 4.5 <= aspect_ratio <= 6.5):
            potential_plates.append(approx)
    
    return potential_plates

def extract_and_correct_plate(img, imgGrayscale, imgThresh, contour):
    """
    Trích xuất và hiệu chỉnh góc nghiêng của biển số
    
    Args:
        img: Ảnh gốc
        imgGrayscale: Ảnh grayscale
        imgThresh: Ảnh nhị phân
        contour: Contour của biển số
        
    Returns:
        tuple: (roi, imgThresh_roi) - Vùng ảnh đã cắt và xoay
    """
    # Tạo mask cho biển số
    mask = np.zeros(imgGrayscale.shape, np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, -1)
    
    # Tìm các điểm trong mask
    (ys, xs) = np.where(mask == 255)
    (topx, topy) = (np.min(ys), np.min(xs))
    (bottomx, bottomy) = (np.max(ys), np.max(xs))
    
    # Trích xuất vùng biển số
    roi = img[topx:bottomx + 1, topy:bottomy + 1]
    imgThresh_roi = imgThresh[topx:bottomx + 1, topy:bottomy + 1]
    
    # Tính toán góc nghiêng của biển số
    pts = contour.reshape(4, 2)
    pts = sorted(pts, key=lambda x: x[1])  # Sắp xếp theo tọa độ y
    (x1, y1), (x2, y2) = pts[:2]
    
    # Tính góc nghiêng để xoay thẳng
    dy = abs(y1 - y2)
    dx = abs(x1 - x2)
    angle = math.atan2(dy, dx) * (180.0 / math.pi)
    
    # Điểm trung tâm cho phép xoay
    plate_center = ((bottomx - topx) / 2, (bottomy - topy) / 2)
    
    # Ma trận xoay
    rotation_matrix = cv2.getRotationMatrix2D(
        plate_center, 
        -angle if x1 < x2 else angle, 
        1.0
    )
    
    # Xoay biển số về ngang
    roi = cv2.warpAffine(roi, rotation_matrix, (bottomy - topy, bottomx - topx))
    imgThresh_roi = cv2.warpAffine(imgThresh_roi, rotation_matrix, (bottomy - topy, bottomx - topx))
    
    # Phóng to ảnh biển số để nhận dạng tốt hơn
    roi = cv2.resize(roi, (0, 0), fx=3, fy=3)
    imgThresh_roi = cv2.resize(imgThresh_roi, (0, 0), fx=3, fy=3)
    
    return roi, imgThresh_roi

def segment_characters(roi, thresh_img):
    """
    Phân đoạn và tìm các ký tự trong biển số
    
    Args:
        roi: Vùng ảnh biển số
        thresh_img: Ảnh nhị phân của biển số
        
    Returns:
        tuple: (char_contours, char_positions) - Các contour và vị trí ký tự
    """
    # Giãn ảnh để nổi bật ký tự
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_img = cv2.morphologyEx(thresh_img, cv2.MORPH_DILATE, kernel)
    
    # Tìm contours của các ký tự
    contours, _ = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Tính toán diện tích ROI
    height, width, _ = roi.shape
    roi_area = height * width
    
    char_positions = {}  # Dictionary lưu vị trí x và index của ký tự
    valid_chars = []     # Danh sách index của các ký tự hợp lệ
    
    # Duyệt qua các contour tìm ký tự hợp lệ
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        (x, y, w, h) = cv2.boundingRect(cnt)
        char_aspect_ratio = w / h
        
        # Kiểm tra tiêu chí về kích thước và tỷ lệ
        is_valid_size = MIN_CHAR_RATIO * roi_area < area < MAX_CHAR_RATIO * roi_area
        is_valid_ratio = MIN_CHAR_ASPECT_RATIO < char_aspect_ratio < MAX_CHAR_ASPECT_RATIO
        
        if is_valid_size and is_valid_ratio:
            if x in char_positions:  # Xử lý trùng tọa độ x
                x += 1
            char_positions[x] = i
            valid_chars.append(i)
    
    return [contours[i] for i in valid_chars], char_positions

def recognize_characters(roi, dilated_img, char_contours, char_positions, knn_model):
    """
    Nhận dạng các ký tự trong biển số xe
    
    Args:
        roi: Vùng ảnh biển số
        dilated_img: Ảnh đã giãn nở
        char_contours: Các contour ký tự
        char_positions: Vị trí các ký tự
        knn_model: Mô hình KNN đã huấn luyện
        
    Returns:
        str: Chuỗi biển số đã nhận dạng
    """
    height, _, _ = roi.shape
    char_x_positions = sorted(char_positions.keys())  # Sắp xếp các ký tự theo tọa độ x
    
    first_line = ""   # Dòng đầu biển số
    second_line = ""  # Dòng hai biển số
    
    # Nhận diện từng ký tự
    for x_pos in char_x_positions:
        cnt_idx = char_positions[x_pos]
        cnt = char_contours[char_x_positions.index(x_pos)]
        
        (x, y, w, h) = cv2.boundingRect(cnt)
        
        # Vẽ hình chữ nhật xung quanh ký tự
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Trích xuất và chuẩn hóa ký tự
        char_img = dilated_img[y:y + h, x:x + w]
        resized_char = cv2.resize(char_img, (RESIZED_CHAR_WIDTH, RESIZED_CHAR_HEIGHT))
        
        # Chuyển đổi ảnh thành vector đặc trưng
        flattened_char = resized_char.reshape((1, RESIZED_CHAR_WIDTH * RESIZED_CHAR_HEIGHT))
        flattened_char = np.float32(flattened_char)
        
        # Sử dụng KNN để nhận dạng ký tự
        _, results, _, _ = knn_model.findNearest(flattened_char, k=3)
        current_char = chr(int(results[0][0]))
        
        # Hiển thị ký tự đã nhận dạng
        cv2.putText(roi, current_char, (x, y + 50), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
        
        # Phân loại ký tự vào dòng trên hoặc dòng dưới
        if y < height / 3:  # Nằm ở 1/3 trên của biển số
            first_line += current_char
        else:
            second_line += current_char
    
    return first_line, second_line

def main():
    """
    Hàm chính xử lý video đầu vào
    """
    global total_frames, plates_found
    
    # Tải mô hình KNN
    knn_model = load_knn_model()
    
    # Đọc video đầu vào
    cap = cv2.VideoCapture('data/video/VideoTest.mp4')
    
    if not cap.isOpened():
        print("Không thể mở file video!")
        return
    
    print("Đang xử lý video, nhấn 'q' để thoát...")
    
    # Vòng lặp xử lý từng khung hình trong video
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        
        total_frames += 1
        
        # Xử lý ảnh (convert grayscale và threshold)
        imgGrayscale, imgThreshplate = PreProcess.preprocess(img)
        
        # Phát hiện biển số
        potential_plates = detect_license_plates(img, imgGrayscale, imgThreshplate)
        
        if not potential_plates:
            print(f"Frame {total_frames}: Không phát hiện biển số")
            continue
        
        # Xử lý từng biển số tìm được
        for n, plate_contour in enumerate(potential_plates, start=1):
            # Trích xuất và hiệu chỉnh biển số
            roi, roi_thresh = extract_and_correct_plate(img, imgGrayscale, imgThreshplate, plate_contour)
            
            # Tìm các ký tự trong biển số
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated_img = cv2.morphologyEx(roi_thresh, cv2.MORPH_DILATE, kernel)
            
            # Phân đoạn và tìm ký tự
            char_contours, char_positions = segment_characters(roi, roi_thresh)
            
            # Kiểm tra số lượng ký tự phù hợp (7-10 ký tự cho một biển số)
            if 7 <= len(char_positions) <= 10:
                # Vẽ contour của biển số
                cv2.drawContours(img, [plate_contour], -1, (0, 255, 0), 3)
                
                # Nhận dạng các ký tự
                first_line, second_line = recognize_characters(
                    roi, dilated_img, char_contours, char_positions, knn_model
                )
                
                # Kết hợp thành chuỗi biển số hoàn chỉnh
                plate_text = f"{first_line}-{second_line}" if second_line else first_line
                
                # Hiển thị kết quả
                print(f"\nBiển số xe {n}: {first_line}{' - ' + second_line if second_line else ''}\n")
                
                # Vẽ biển số lên ảnh gốc
                pts = plate_contour.reshape(4, 2)
                (topx, topy) = (np.min(pts[:, 1]), np.min(pts[:, 0]))
                cv2.putText(img, plate_text, (topy, topx), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1)
                
                plates_found += 1
                
                # Hiển thị vùng biển số đã xử lý
                cv2.imshow("Biển số", cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        
        # Hiển thị ảnh kết quả
        imgcopy = cv2.resize(img, None, fx=0.5, fy=0.5)
        cv2.imshow('Nhận diện biển số xe', imgcopy)
        
        # Hiển thị thống kê
        print(f"Số biển số đã phát hiện: {plates_found}")
        print(f"Tổng số frame: {total_frames}")
        if total_frames > 0:
            print(f"Tỷ lệ phát hiện: {100 * plates_found / total_frames:.2f}%")
        
        # Thoát nếu nhấn phím 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()
    
    # Thống kê cuối cùng
    print("\n----- KẾT QUẢ THỐNG KÊ -----")
    print(f"Tổng số frame đã xử lý: {total_frames}")
    print(f"Tổng số biển số đã phát hiện: {plates_found}")
    if total_frames > 0:
        print(f"Tỷ lệ phát hiện biển số: {100 * plates_found / total_frames:.2f}%")

if __name__ == "__main__":
    main()