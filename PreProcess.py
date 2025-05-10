import cv2
import numpy as np

# Các tham số cho tiền xử lý ảnh
GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)  # Kích cỡ càng to thì càng mờ
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9

def preprocess(imgOriginal):
    """
    Xử lý ảnh đầu vào để tạo ảnh nhị phân phù hợp cho việc nhận diện biển số
    
    Args:
        imgOriginal: Ảnh gốc BGR từ OpenCV
        
    Returns:
        tuple: (ảnh grayscale, ảnh nhị phân)
    """
    # Chuyển đổi ảnh sang grayscale sử dụng kênh Value của HSV
    imgGrayscale = extractValue(imgOriginal)
    
    # Tăng độ tương phản để làm nổi bật biển số
    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)
    
    # Làm mịn ảnh bằng bộ lọc Gaussian
    height, width = imgGrayscale.shape
    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
    
    # Tạo ảnh nhị phân sử dụng ngưỡng thích ứng
    imgThresh = cv2.adaptiveThreshold(
        imgBlurred, 
        255.0, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        ADAPTIVE_THRESH_BLOCK_SIZE, 
        ADAPTIVE_THRESH_WEIGHT
    )
    
    return imgGrayscale, imgThresh

def extractValue(imgOriginal):
    """
    Trích xuất kênh Value từ ảnh HSV
    
    Args:
        imgOriginal: Ảnh gốc BGR từ OpenCV
        
    Returns:
        np.array: Kênh Value (độ sáng) của ảnh
    """
    # Chuyển đổi từ BGR sang HSV
    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)
    
    # Tách các kênh HSV, sử dụng kênh Value (độ sáng)
    # Không chọn màu RGB vì vd ảnh màu đỏ sẽ còn lẫn các màu khác nên khó xác định "một màu"
    _, _, imgValue = cv2.split(imgHSV)
    
    return imgValue

def maximizeContrast(imgGrayscale):
    """
    Cải thiện độ tương phản của ảnh bằng phép toán hình thái học
    
    Args:
        imgGrayscale: Ảnh grayscale đầu vào
        
    Returns:
        np.array: Ảnh có độ tương phản được cải thiện
    """
    height, width = imgGrayscale.shape
    
    # Tạo phần tử cấu trúc cho phép toán hình thái học
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    # Top hat: nổi bật chi tiết sáng trong nền tối
    imgTopHat = cv2.morphologyEx(
        imgGrayscale, 
        cv2.MORPH_TOPHAT, 
        structuringElement, 
        iterations=10
    )
    
    # Black hat: nổi bật chi tiết tối trong nền sáng
    imgBlackHat = cv2.morphologyEx(
        imgGrayscale, 
        cv2.MORPH_BLACKHAT, 
        structuringElement, 
        iterations=10
    )
    
    # Kết hợp các ảnh để tăng độ tương phản
    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)
    
    return imgGrayscalePlusTopHatMinusBlackHat