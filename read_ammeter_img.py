import argparse
import cv2
import numpy as np
import math
import os
def get_args_parser():
    parser = argparse.ArgumentParser('Reading Needle', add_help=False)
    parser.add_argument('--input_img_path', type=str, default='')
    parser.add_argument('--output_img_path', type=str, default='')
    return parser

def ammeter(image_input):

    # Đọc ảnh
    img = cv2.imread(image_input)
    if img is None:
        print(f"Error: Could not read image from {image_input}")
        return
    
    # Tạo bản sao để vẽ lên
    output_img = img.copy()

    # Chuyển sang ảnh xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Làm mờ để giảm nhiễu
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Phân ngưỡng để tạo ảnh nhị phân
    _, binary_img = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Lấy kích thước ảnh
    height, width = binary_img.shape
    
    # Tạo mask để loại bỏ các vùng viền/rìa của ảnh
    margin = 20  # Khoảng cách từ viền vào
    mask = np.zeros_like(binary_img)
    mask[margin:height-margin, margin:width-margin] = 255
    
    # Áp dụng mask lên ảnh gốc
    masked_img = cv2.bitwise_and(binary_img, mask)
    
    # Tìm các connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        masked_img, connectivity=8)
    
    # Lọc các components có nền đen bao quanh
    valid_components = []
    
    for i in range(1, num_labels):  # Bỏ qua label 0 (background)
        # Tạo mask cho component hiện tại
        component_mask = np.zeros_like(binary_img)
        component_mask[labels == i] = 255
        
        # Giãn component để kiểm tra xung quanh nó
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(component_mask, kernel, iterations=1)
        
        # Lấy vùng viền bao quanh component (chỉ lấy lớp viền)
        border = cv2.subtract(dilated, component_mask)
        
        # Kiểm tra viền xung quanh có phải là màu đen (0) không
        border_values = binary_img.copy()
        border_values[border == 0] = 255  # Chỉ quan tâm đến giá trị tại viền
        
        # Đếm số pixel màu đen trong viền
        black_border_pixels = np.sum(border_values == 0)
        total_border_pixels = np.sum(border == 255)
        
        # Nếu hơn 80% viền là màu đen, coi như component này được bao quanh bởi màu đen
        if total_border_pixels > 0 and black_border_pixels / total_border_pixels >= 0.8:
            valid_components.append(i)
    
    # Tìm component có diện tích lớn nhất
    best_component = None
    max_component_area = 0
    
    for component_id in valid_components:
        area = stats[component_id, cv2.CC_STAT_AREA]
        if area > max_component_area:
            max_component_area = area
            best_component = component_id
    
    # Nếu tìm được component tốt nhất, tô màu vàng
    if best_component is not None:
        # Tạo mask cho component này
        component_mask = np.zeros_like(binary_img)
        component_mask[labels == best_component] = 255
        
        # Tô màu vàng cho component
        # output_img[component_mask == 255] = (0, 255, 255)  # Màu vàng (BGR)
        
        # Tìm các đoạn thẳng dài nhất trong component
        # Áp dụng Canny edge detection để tìm các cạnh trong component
        component_edges = cv2.Canny(component_mask, 50, 150)
        
        # Áp dụng Hough Line Transform để tìm các đoạn thẳng
        lines = cv2.HoughLinesP(
            component_edges,
            rho=1,
            theta=np.pi/180,
            threshold=10,
            minLineLength=10,
            maxLineGap=10
        )
        
        if lines is not None:
            # Tính chiều dài của từng đoạn thẳng
            line_lengths = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                line_lengths.append((length, (x1, y1, x2, y2)))
            
            # Sắp xếp các đoạn thẳng theo chiều dài (giảm dần)
            line_lengths.sort(reverse=True, key=lambda x: x[0])
            
            # Lấy 5 đoạn thẳng dài nhất
            longest_lines = line_lengths[:5] if len(line_lengths) >= 5 else line_lengths
            
            # Loại bỏ các đoạn thẳng có chiều dài dưới 100
            longest_lines = [(length, line) for length, line in longest_lines if length >= 100]
            
            # # Vẽ các đoạn thẳng dài nhất lên hình
            # colors = [
            #     (0, 0, 255),      # Đỏ
            #     (255, 0, 0),      # Xanh dương
            #     (0, 255, 0),      # Xanh lá
            #     (255, 0, 255),    # Tím
            #     (255, 255, 0),    # Cyan   
            # ]
            
            # for i, (length, (x1, y1, x2, y2)) in enumerate(longest_lines):
            #     # Lấy màu tương ứng cho đoạn thẳng
            #     color = colors[i % len(colors)]
                
            #     # Vẽ đoạn thẳng với độ dày 2 pixel
            #     cv2.line(output_img, (x1, y1), (x2, y2), color, 2)
                
            #     # Hiển thị chiều dài và số thứ tự của đoạn thẳng
            #     midpoint = ((x1 + x2) // 2, (y1 + y2) // 2)
            #     cv2.putText(
            #         output_img,
            #         f"{i+1}: {length:.1f}",
            #         midpoint,
            #         cv2.FONT_HERSHEY_SIMPLEX,
            #         0.5,
            #         color,
            #         1,
            #         cv2.LINE_AA
            #     )
            
            # Lưu trữ thông tin các đoạn thẳng dưới dạng phương trình ax + by + c = 0
            line_equations = []
            
            # Hàm xử lý cặp đường thẳng để tìm đường giữa hoặc phân giác
            def process_line_pair(line1, line2):
                a1, b1, c1 = line1
                a2, b2, c2 = line2
                
                # Kiểm tra xem hai đoạn thẳng có song song không
                is_parallel = abs(a1*b2 - a2*b1) < 1e-6
                
                if is_parallel:
                    # Tìm đường thẳng song song nằm chính giữa
                    middle_c = (c1 + c2) / 2
                    result = (a1, b1, middle_c)
                else:
                    # Tính đường phân giác
                    bisector_a = a1 + a2
                    bisector_b = b1 + b2
                    bisector_c = c1 + c2
                    
                    # Chuẩn hóa
                    norm = math.sqrt(bisector_a*bisector_a + bisector_b*bisector_b)
                    result = (bisector_a/norm, bisector_b/norm, bisector_c/norm)
                
                return result
            
            # Chuyển đổi tất cả các đoạn thẳng thành phương trình ax + by + c = 0
            for _, (x1, y1, x2, y2) in longest_lines:
                # Tính các tham số của đường thẳng (a, b, c) trong phương trình ax + by + c = 0
                if x2 != x1:  # Không phải đoạn thẳng thẳng đứng
                    a = float(y2 - y1)
                    b = float(x1 - x2)
                    c = float(x2 * y1 - x1 * y2)
                else:  # Đoạn thẳng thẳng đứng
                    a = 1.0
                    b = 0.0
                    c = -float(x1)
                
                # Chuẩn hóa để a^2 + b^2 = 1
                norm = math.sqrt(a*a + b*b)
                a, b, c = a/norm, b/norm, c/norm
                
                # Lưu thông tin của đoạn thẳng
                line_equations.append((a, b, c))
            
            # Thuật toán xử lý tuần tự các đoạn thẳng
            if len(line_equations) >= 2:  # Cần ít nhất 2 đoạn thẳng để bắt đầu
                # Bắt đầu với 2 đoạn thẳng đầu tiên
                current_line = process_line_pair(line_equations[0], line_equations[1])
                
                # Xử lý tuần tự các đoạn thẳng còn lại
                for i in range(2, len(line_equations)):
                    current_line = process_line_pair(current_line, line_equations[i])
                
                # Lấy kết quả cuối cùng
                final_a, final_b, final_c = current_line
                
                # Tính hai điểm trên đường thẳng cuối cùng để vẽ
                if abs(final_b) > 1e-6:  # Không song song với trục y
                    x1, x2 = 0, width
                    y1 = (-final_a * x1 - final_c) / final_b
                    y2 = (-final_a * x2 - final_c) / final_b
                else:  # Song song với trục y
                    y1, y2 = 0, height
                    x1 = (-final_b * y1 - final_c) / final_a
                    x2 = (-final_b * y2 - final_c) / final_a
                
                # Đảm bảo các điểm nằm trong phạm vi ảnh
                if y1 < 0:
                    x1 = (-final_b * 0 - final_c) / final_a
                    y1 = 0
                elif y1 >= height:
                    x1 = (-final_b * (height-1) - final_c) / final_a
                    y1 = height-1
                    
                if y2 < 0:
                    x2 = (-final_b * 0 - final_c) / final_a
                    y2 = 0
                elif y2 >= height:
                    x2 = (-final_b * (height-1) - final_c) / final_a
                    y2 = height-1
                    
                if x1 < 0:
                    y1 = (-final_a * 0 - final_c) / final_b
                    x1 = 0
                elif x1 >= width:
                    y1 = (-final_a * (width-1) - final_c) / final_b
                    x1 = width-1
                    
                if x2 < 0:
                    y2 = (-final_a * 0 - final_c) / final_b
                    x2 = 0
                elif x2 >= width:
                    y2 = (-final_a * (width-1) - final_c) / final_b
                    x2 = width-1
                
                # Vẽ đường thẳng cuối cùng với màu trắng và độ dày 3 pixel
                cv2.line(output_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                
                # Tính góc của đường thẳng (tính theo độ)
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

                # # Đánh dấu đường cuối cùng
                # cv2.putText(
                #     output_img,
                #     "Final Line",
                #     (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.9,
                #     (255, 255, 255),
                #     2,
                #     cv2.LINE_AA
                # )
    
    # Lưu ảnh kết quả
    # cv2.imwrite(output_path, output_img)
    # print(f"Đã lưu ảnh kết quả vào {output_path}")
    
    return output_img, angle

def main(args):
    standard_image_0 = 'Standard/0.png'
    standard_image_50 = 'Standard/50.png'
    _, angle0 = ammeter(standard_image_0)
    _, angle50 = ammeter(standard_image_50)

    input_img_path = args.input_img_path
    output_img_path = args.output_img_path
    
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_img_path, exist_ok=True)
    
    for file in os.listdir(input_img_path):
        image_input = os.path.join(input_img_path, file)
        image_output_name = file.replace('.png', '_result.png')
        image_output = os.path.join(output_img_path, image_output_name)
        result_img, angle = ammeter(image_input)
        Reading = (angle - angle0) / (angle50 - angle0) * 50
        
         # Text settings
        text = f"Reading: {Reading:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        font_thickness = 3
        text_color = (0, 0, 255)  # Red color in BGR
        
        # Get text size to create background box
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        padding = 10
        
        # Draw background rectangle
        text_pos = (30, 60)
        cv2.rectangle(
            result_img, 
            (text_pos[0] - padding, text_pos[1] - text_height - padding),
            (text_pos[0] + text_width + padding, text_pos[1] + padding),
            (255, 255, 255),  # White background
            -1  # Filled rectangle
        )
        
        # Draw border for the rectangle
        cv2.rectangle(
            result_img, 
            (text_pos[0] - padding, text_pos[1] - text_height - padding),
            (text_pos[0] + text_width + padding, text_pos[1] + padding),
            (0, 0, 0),  # Black border
            2  # Border thickness
        )
        
        # Draw text
        cv2.putText(
            result_img,
            text,
            text_pos,
            font,
            font_scale,
            text_color,
            font_thickness
        )
        
        cv2.imwrite(image_output, result_img)
    

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)