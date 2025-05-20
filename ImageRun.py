import math
import cv2
import numpy as np
import PreProcess  # Tệp xử lý ảnh riêng bạn tự định nghĩa

# Tham số ngưỡng dùng cho nhị phân ảnh thích nghi
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9
n = 1  # Biến đếm số lượng biển số được phát hiện

# Ngưỡng diện tích tương đối tối thiểu và tối đa cho ký tự trên ảnh
Min_char = 0.01
Max_char = 0.09

# Kích thước chuẩn của ký tự sau khi resize để đưa vào KNN
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

# Đọc ảnh và resize cho đồng nhất kích thước
img = cv2.imread("Data/Image/23.jpg")
img = cv2.resize(img, dsize=(1920, 1080))

######## Tải mô hình KNN đã huấn luyện ########
npaClassifications = np.loadtxt("classifications.txt", np.float32)  # Nhãn ký tự
npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)  # Dữ liệu ảnh đã làm phẳng
npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))  # Chuyển thành mảng 2 chiều

kNearest = cv2.ml.KNearest_create()  # Tạo đối tượng KNN
kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)  # Huấn luyện mô hình KNN
###############################################

########## Xử lý ảnh gốc ##########
imgGrayscaleplate, imgThreshplate = PreProcess.preprocess(img)  # Chuyển xám + nhị phân
cv2.imshow("Ảnh xám", imgGrayscaleplate)
cv2.imshow("Ảnh nhị phân", imgThreshplate)

# Áp dụng bộ lọc Canny để phát hiện biên
canny_image = cv2.Canny(imgThreshplate, 250, 255)
cv2.imshow("Ảnh Canny", canny_image)

# Làm dày các cạnh bằng phép dãn
kernel = np.ones((3, 3), np.uint8)
dilated_image = cv2.dilate(canny_image, kernel, iterations=1)
cv2.imshow("Ảnh giãn biên", dilated_image)
##################################

########## Tìm contour và chọn contour có thể là biển số ##########
contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # Giữ lại 10 contour lớn nhất

screenCnt = []  # Danh sách contour có thể là biển số
for c in contours:
    peri = cv2.arcLength(c, True)  # Tính chu vi
    approx = cv2.approxPolyDP(c, 0.06 * peri, True)  # Làm mịn đa giác, ưu tiên 4 đỉnh
    [x, y, w, h] = cv2.boundingRect(approx.copy())
    ratio = w / h

    if (len(approx) == 4):  # Nếu contour có 4 cạnh thì đưa vào danh sách
        screenCnt.append(approx)
        cv2.putText(img, str(len(approx.copy())), (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)

if screenCnt is None:
    detected = 0
    print("Không phát hiện biển số")
else:
    detected = 1

########## Nếu phát hiện được biển số ##########
if detected == 1:
    for screenCnt in screenCnt:
        # Vẽ khung viền quanh biển số
        cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)

        # Tính góc lệch của biển số để xoay lại cho đúng
        (x1, y1) = screenCnt[0, 0]
        (x2, y2) = screenCnt[1, 0]
        (x3, y3) = screenCnt[2, 0]
        (x4, y4) = screenCnt[3, 0]
        array = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        sorted_array = array.sort(reverse=True, key=lambda x: x[1])
        (x1, y1) = array[0]
        (x2, y2) = array[1]
        doi = abs(y1 - y2)
        ke = abs(x1 - x2)
        angle = math.atan(doi / ke) * (180.0 / math.pi)  # Tính góc xoay

        # Tạo mặt nạ và cắt biển số ra khỏi ảnh gốc
        mask = np.zeros(imgGrayscaleplate.shape, np.uint8)
        new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1)

        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))

        roi = img[topx:bottomx, topy:bottomy]
        imgThresh = imgThreshplate[topx:bottomx, topy:bottomy]
        ptPlateCenter = ((bottomx - topx) / 2, (bottomy - topy) / 2)

        # Xoay biển số cho thẳng
        if x1 < x2:
            rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, -angle, 1.0)
        else:
            rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, angle, 1.0)

        roi = cv2.warpAffine(roi, rotationMatrix, (bottomy - topy, bottomx - topx))
        imgThresh = cv2.warpAffine(imgThresh, rotationMatrix, (bottomy - topy, bottomx - topx))
        roi = cv2.resize(roi, (0, 0), fx=3, fy=3)
        imgThresh = cv2.resize(imgThresh, (0, 0), fx=3, fy=3)

        ########## Tách ký tự trên biển số ##########
        kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(imgThresh, cv2.MORPH_DILATE, kerel3)
        cont, hier = cv2.findContours(thre_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.imshow(str(n + 20), thre_mor)
        cv2.drawContours(roi, cont, -1, (100, 255, 255), 2)

        ########## Lọc ra các contour có thể là ký tự ##########
        char_x_ind = {}
        char_x = []
        height, width, _ = roi.shape
        roiarea = height * width

        for ind, cnt in enumerate(cont):
            (x, y, w, h) = cv2.boundingRect(cont[ind])
            ratiochar = w / h
            char_area = w * h

            if (Min_char * roiarea < char_area < Max_char * roiarea) and (0.25 < ratiochar < 0.7):
                if x in char_x:
                    x = x + 1  # Tránh trùng key
                char_x.append(x)
                char_x_ind[x] = ind

        ########## Nhận diện ký tự bằng KNN ##########
        char_x = sorted(char_x)
        strFinalString = ""
        first_line = ""
        second_line = ""

        for i in char_x:
            (x, y, w, h) = cv2.boundingRect(cont[char_x_ind[i]])
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
            imgROI = thre_mor[y:y + h, x:x + w]

            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
            npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
            npaROIResized = np.float32(npaROIResized)

            _, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k=5)

            print("Ký tự nhận được:", npaResults)
            strCurrentChar = str(chr(int(npaResults[0][0])))  # Chuyển số ASCII sang ký tự
            cv2.putText(roi, strCurrentChar, (x, y + 50), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 3)

            # Phân biệt dòng đầu và dòng dưới của biển số
            if (y < height / 3):
                first_line = first_line + strCurrentChar
            else:
                second_line = second_line + strCurrentChar

        # In kết quả biển số
        print("\n Biển số " + str(n) + ": " + first_line + " - " + second_line + "\n")
        roi = cv2.resize(roi, None, fx=0.75, fy=0.75)
        cv2.imshow(str(n), cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

        # Vẽ kết quả nhận diện lên ảnh gốc
        cv2.putText(img, first_line + "-" + second_line, (topy, topx), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 2)
        n = n + 1

# Hiển thị ảnh gốc với kết quả
img = cv2.resize(img, None, fx=0.5, fy=0.5)
cv2.imshow('Biển số xe', img)

cv2.waitKey(0)  # Chờ nhấn phím để thoát
