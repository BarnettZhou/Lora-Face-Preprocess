import face_recognition
from PIL import Image, ImageDraw
import numpy as np

def _load_image(image_path):
    """
    加载图片文件
    
    Args:
        image_path (str): 图片路径
        
    Returns:
        numpy.ndarray: 加载的图片数组，如果失败返回None
    """
    try:
        image = face_recognition.load_image_file(image_path)
        return image
    except FileNotFoundError:
        print(f"错误：找不到文件 '{image_path}'。请检查路径是否正确。")
        return None
    except Exception as e:
        print(f"加载图片时发生错误：{e}")
        return None

def _detect_face_landmarks(image):
    """
    检测人脸关键点
    
    Args:
        image (numpy.ndarray): 输入图片数组
        
    Returns:
        dict: 人脸关键点字典，如果未检测到人脸返回None
    """
    face_landmarks_list = face_recognition.face_landmarks(image)
    
    if not face_landmarks_list:
        print("未检测到人脸。")
        return None
    
    # 假设只处理第一张检测到的人脸
    return face_landmarks_list[0]

def _extract_key_points(landmarks):
    """
    提取眼睛和下巴的关键点
    
    Args:
        landmarks (dict): 人脸关键点字典
        
    Returns:
        tuple: (eyes_center, bottom_chin) 眼睛中心点和下巴最底部点
    """
    # 获取眼睛关键点
    left_eye = np.mean(landmarks['left_eye'], axis=0).astype(int)
    right_eye = np.mean(landmarks['right_eye'], axis=0).astype(int)
    
    # 计算两眼中心点
    eyes_center_x = (left_eye[0] + right_eye[0]) / 2
    eyes_center_y = (left_eye[1] + right_eye[1]) / 2
    eyes_center = (eyes_center_x, eyes_center_y)
    
    # 找到下巴最下方的点
    chin_points = landmarks['chin']
    bottom_chin_y = max([p[1] for p in chin_points])
    bottom_chin_x = chin_points[np.argmax([p[1] for p in chin_points])][0]
    bottom_chin = (bottom_chin_x, bottom_chin_y)
    
    return eyes_center, bottom_chin

def _calculate_crop_region(eyes_center, bottom_chin, target_eyes_y, target_chin_y, target_size):
    """
    计算裁剪区域
    
    Args:
        eyes_center (tuple): 眼睛中心点坐标
        bottom_chin (tuple): 下巴最底部点坐标
        target_eyes_y (int): 目标眼睛Y坐标
        target_chin_y (int): 目标下巴Y坐标
        target_size (tuple): 目标图片尺寸 (宽度, 高度)
        
    Returns:
        tuple: (crop_left, crop_top, crop_right, crop_bottom) 裁剪区域坐标
    """
    target_width, target_height = target_size
    
    # 计算眼睛到下巴的距离
    eyes_to_chin_distance = bottom_chin[1] - eyes_center[1]
    target_eyes_to_chin_distance = target_chin_y - target_eyes_y
    
    # 计算缩放因子
    scale_factor = target_eyes_to_chin_distance / eyes_to_chin_distance
    
    # 计算裁剪框坐标
    crop_left = eyes_center[0] - (target_width / 2) / scale_factor
    crop_top = eyes_center[1] - target_eyes_y / scale_factor
    crop_right = crop_left + target_width / scale_factor
    crop_bottom = crop_top + target_height / scale_factor
    
    return crop_left, crop_top, crop_right, crop_bottom

def _crop_and_resize_image(image, crop_coords, target_size):
    """
    裁剪和调整图片大小（等比例缩放，边缘填充）
    
    Args:
        image (numpy.ndarray): 原始图片数组
        crop_coords (tuple): 裁剪坐标 (left, top, right, bottom)
        target_size (tuple): 目标尺寸 (宽度, 高度)
        
    Returns:
        PIL.Image: 处理后的图片对象
    """
    import numpy as np
    
    crop_left, crop_top, crop_right, crop_bottom = crop_coords
    
    # 将坐标转换为整数
    crop_left, crop_top, crop_right, crop_bottom = map(int, [crop_left, crop_top, crop_right, crop_bottom])
    
    # 确保裁剪框在原始图片范围内
    img_height, img_width = image.shape[:2]
    crop_left = max(0, crop_left)
    crop_top = max(0, crop_top)
    crop_right = min(img_width, crop_right)
    crop_bottom = min(img_height, crop_bottom)
    
    # 裁剪图片
    cropped_image_array = image[crop_top:crop_bottom, crop_left:crop_right]
    cropped_image_pil = Image.fromarray(cropped_image_array)
    
    # 获取裁剪后的图片尺寸和目标尺寸
    crop_width, crop_height = cropped_image_pil.size
    target_width, target_height = target_size
    
    # 计算等比例缩放的比例
    scale_x = target_width / crop_width
    scale_y = target_height / crop_height
    scale = min(scale_x, scale_y)  # 使用较小的比例保持等比例
    
    # 计算缩放后的实际尺寸
    scaled_width = int(crop_width * scale)
    scaled_height = int(crop_height * scale)
    
    # 等比例缩放图片
    scaled_image = cropped_image_pil.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
    
    # 创建目标尺寸的画布
    result_image = Image.new('RGB', target_size)
    
    # 计算居中放置的位置
    paste_x = (target_width - scaled_width) // 2
    paste_y = (target_height - scaled_height) // 2
    
    # 将缩放后的图片粘贴到画布中心
    result_image.paste(scaled_image, (paste_x, paste_y))
    
    # 转换为numpy数组进行边缘填充
    result_array = np.array(result_image)
    
    # 使用边缘像素填充空白区域
    if paste_x > 0:  # 左右需要填充
        # 填充左边
        left_edge = result_array[:, paste_x:paste_x+1, :]
        result_array[:, :paste_x, :] = np.repeat(left_edge, paste_x, axis=1)
        
        # 填充右边
        right_edge = result_array[:, paste_x+scaled_width-1:paste_x+scaled_width, :]
        result_array[:, paste_x+scaled_width:, :] = np.repeat(right_edge, target_width-paste_x-scaled_width, axis=1)
    
    if paste_y > 0:  # 上下需要填充
        # 填充上边
        top_edge = result_array[paste_y:paste_y+1, :, :]
        result_array[:paste_y, :, :] = np.repeat(top_edge, paste_y, axis=0)
        
        # 填充下边
        bottom_edge = result_array[paste_y+scaled_height-1:paste_y+scaled_height, :, :]
        result_array[paste_y+scaled_height:, :, :] = np.repeat(bottom_edge, target_height-paste_y-scaled_height, axis=0)
    
    # 转换回PIL图片
    return Image.fromarray(result_array)

def _save_image(image, output_path, target_eyes_y, target_chin_y, portrait_type):
    """
    保存图片并输出信息
    
    Args:
        image (PIL.Image): 要保存的图片对象
        output_path (str): 输出路径
        target_eyes_y (int): 目标眼睛Y坐标
        target_chin_y (int): 目标下巴Y坐标
        portrait_type (str): 人像类型描述
    """
    try:
        image.save(output_path)
        print(f"成功生成{portrait_type}人像图片并保存到 '{output_path}'")
        print(f"眼睛位置：{target_eyes_y}像素，下巴位置：{target_chin_y}像素")
    except Exception as e:
        print(f"保存图片时发生错误：{e}")

def _generate_portrait_base(image_path, output_path, target_size, target_eyes_y, target_chin_y, portrait_type):
    """
    生成人像的基础方法
    
    Args:
        image_path (str): 输入图片路径
        output_path (str): 输出图片路径
        target_size (tuple): 目标尺寸
        target_eyes_y (int): 目标眼睛Y坐标
        target_chin_y (int): 目标下巴Y坐标
        portrait_type (str): 人像类型描述
    """
    # 加载图片
    image = _load_image(image_path)
    if image is None:
        return
    
    # 检测人脸关键点
    landmarks = _detect_face_landmarks(image)
    if landmarks is None:
        return
    
    # 提取关键点
    eyes_center, bottom_chin = _extract_key_points(landmarks)
    
    # 计算裁剪区域
    crop_coords = _calculate_crop_region(eyes_center, bottom_chin, target_eyes_y, target_chin_y, target_size)
    
    # 裁剪和调整图片大小
    processed_image = _crop_and_resize_image(image, crop_coords, target_size)
    
    # 保存图片
    _save_image(processed_image, output_path, target_eyes_y, target_chin_y, portrait_type)

def generate_portrait_face(image_path, output_path="portrait.jpg", target_size=(1024, 1024)):
    """
    检测图片中的人脸，并根据人脸位置、大小信息，生成指定大小的人像图片。
    具体来说，将人的眼睛放在高度512像素左右，下巴下缘在高度790像素左右。

    Args:
        image_path (str): 输入图片的路径。
        output_path (str): 输出人像图片的保存路径。
        target_size (tuple): 输出图片的目标尺寸 (宽度, 高度)。
    """
    _generate_portrait_base(image_path, output_path, target_size, 512, 790, "")

def generate_portrait_half_body(image_path, output_path="portrait_half_body.jpg", target_size=(1024, 1024)):
    """
    检测图片中的人脸，并根据人脸位置、大小信息，生成指定大小的上半身人像图片。
    具体来说，将人的眼睛放在高度360像素左右，下巴下缘在高度470像素左右。
    适合生成上半身照效果。

    Args:
        image_path (str): 输入图片的路径。
        output_path (str): 输出人像图片的保存路径。
        target_size (tuple): 输出图片的目标尺寸 (宽度, 高度)。
    """
    _generate_portrait_base(image_path, output_path, target_size, 360, 470, "上半身")

def generate_portrait_upper_body(image_path, output_path="portrait_upper_body.jpg", target_size=(1024, 1024)):
    """
    检测图片中的人脸，并根据人脸位置、大小信息，生成指定大小的半身人像图片。
    具体来说，将人的眼睛放在高度330像素左右，下巴下缘在高度512像素左右。
    适合生成半身照效果。

    Args:
        image_path (str): 输入图片的路径。
        output_path (str): 输出人像图片的保存路径。
        target_size (tuple): 输出图片的目标尺寸 (宽度, 高度)。
    """
    _generate_portrait_base(image_path, output_path, target_size, 330, 512, "半身")
