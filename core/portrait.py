import face_recognition
from PIL import Image
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

def _calulate_scale_and_paste_coords(eyes_center, bottom_chin, target_eyes_y, target_chin_y, target_size):
    """
    计算缩放比例和粘贴坐标

    Returns:
        tuple: (scale_factor, paste_coords) 缩放比例和粘贴坐标
    """

    target_width, target_height = target_size

    # 计算眼睛到下巴的距离
    eyes_to_chin_distance = bottom_chin[1] - eyes_center[1]
    target_eyes_to_chin_distance = target_chin_y - target_eyes_y

    # 计算缩放因子
    scale_factor = target_eyes_to_chin_distance / eyes_to_chin_distance

    # 计算粘贴坐标
    paste_x = int(target_width / 2 - eyes_center[0] * scale_factor)
    paste_y = int(target_eyes_y - eyes_center[1] * scale_factor)

    paste_coords = (paste_x, paste_y)

    return scale_factor, paste_coords

def _resize_and_paste_image(image, scale_factor, paste_coords, target_size, fill_blank):
    """
    调整图片大小并粘贴到目标位置

    Args:
        image (numpy.ndarray): 原始图片数组
        scale_factor (float): 缩放因子
        paste_coords (tuple): 粘贴坐标 (x, y)
        target_size (tuple): 目标尺寸 (宽度, 高度)
        fill_blank (bool): 是否填充空白区域

    Returns:
        PIL.Image: 处理后的图片对象
    """
    target_width, target_height = target_size

    # 调整图片大小
    resized_height = int(image.shape[0] * scale_factor)
    resized_width = int(image.shape[1] * scale_factor)
    resized_image = Image.fromarray(image).resize((resized_width, resized_height), Image.Resampling.LANCZOS)

    # 创建一个新的图片对象，大小为目标尺寸
    result_image = Image.new('RGB', target_size)

    # 粘贴原图到目标图片上
    paste_x, paste_y = paste_coords
    result_image.paste(resized_image, (paste_x, paste_y))

    # 检查是否有空白区域，并根据fill_blank决定是否填充
    has_blank_area = (
        paste_x > 0
        or paste_y > 0
        or paste_x + resized_width < target_width
        or paste_y + resized_height < target_height
    )
    if has_blank_area and fill_blank:
        result_array = np.array(result_image)
        # 填充水平方向的空白
        if paste_x > 0:
            # 填充左边
            left_edge = result_array[:, paste_x:paste_x+1, :]
            result_array[:, :paste_x, :] = np.repeat(left_edge, paste_x, axis=1)

            # 填充右边
            right_edge = result_array[:, paste_x+resized_width-1:paste_x+resized_width, :]
            result_array[:, paste_x+resized_width:, :] = np.repeat(right_edge, target_width-paste_x-resized_width, axis=1)

        # 填充垂直方向的空白
        if paste_y > 0:
            # 填充上边
            top_edge = result_array[paste_y:paste_y+1, :, :]
            result_array[:paste_y, :, :] = np.repeat(top_edge, paste_y, axis=0)

            # 填充下边
            bottom_edge = result_array[paste_y+resized_height-1:paste_y+resized_height, :, :]
            result_array[paste_y+resized_height:, :, :] = np.repeat(bottom_edge, target_height-paste_y-resized_height, axis=0)

        return Image.fromarray(result_array)
    else:
        # 不填充或没有空白区域，直接返回
        return result_image

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

def _generate_portrait_base(image_path, output_path, target_size, target_eyes_y, target_chin_y, portrait_type, fill_blank=False):
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
    # crop_coords = _calculate_crop_region(eyes_center, bottom_chin, target_eyes_y, target_chin_y, target_size)
    scale_factor, paste_coords = _calulate_scale_and_paste_coords(eyes_center, bottom_chin, target_eyes_y, target_chin_y, target_size)

    # 调整图片大小并粘贴到目标位置
    processed_image = _resize_and_paste_image(image, scale_factor, paste_coords, target_size, fill_blank)
    # 裁剪和调整图片大小
    # processed_image = _crop_and_resize_image(image, crop_coords, target_size, fill_blank)

    # 保存图片
    _save_image(processed_image, output_path, target_eyes_y, target_chin_y, portrait_type)

def _get_face_target_y_coords(target_size):
    """
    根据目标尺寸计算目标眼睛Y坐标和目标下巴Y坐标
    用于 face
    """
    target_width, target_height = target_size

    # 1:1 的情况
    if target_width == target_height:
        target_eyes_y = int(target_height * 0.5)    # 眼睛Y坐标在高度的一半
        target_chin_y = int(target_height * 0.77)   # 下巴Y坐标在高度的77%处
        return target_eyes_y, target_chin_y
    
    # 3:4 的情况
    if target_width * 4 == target_height * 3:
        target_eyes_y = int(target_height * 0.37)   # 眼睛Y坐标在高度的37%处
        target_chin_y = int(target_height * 0.72)   # 下巴Y坐标在高度的72%处
        return target_eyes_y, target_chin_y

    # 4:3 的情况
    target_eyes_y = int(target_height * 0.47)        # 眼睛Y坐标在高度的一半
    target_chin_y = int(target_height * 0.81)       # 下巴Y坐标在高度的87%处
    return target_eyes_y, target_chin_y

def generate_portrait_face(image_path, output_path, target_size=(1024, 1024), fill_blank=False):
    ""
    """
    检测图片中的人脸，并根据人脸位置、大小信息，生成指定大小的人像图片。
    具体来说，将人的眼睛放在高度512像素左右，下巴下缘在高度790像素左右。

    Args:
        image_path (str): 输入图片的路径。
        output_path (str): 输出人像图片的保存路径。
        target_size (tuple): 输出图片的目标尺寸 (宽度, 高度)。
    """

    target_eyes_y, target_chin_y = _get_face_target_y_coords(target_size)
    _generate_portrait_base(image_path, output_path, target_size, target_eyes_y, target_chin_y, "", fill_blank)

def _get_upper_body_target_y_coords(target_size):
    """
    根据目标尺寸计算目标眼睛Y坐标和目标下巴Y坐标
    用于 upper body
    """
    target_width, target_height = target_size

    # 1:1 的情况
    if target_width == target_height:
        target_eyes_y = int(target_height * 0.32)   # 眼睛Y坐标在高度的32%处
        target_chin_y = int(target_height * 0.54)   # 下巴Y坐标在高度的54%处
        return target_eyes_y, target_chin_y

    # 3:4 的情况
    if target_width * 4 == target_height * 3:
        target_eyes_y = int(target_height * 0.38)   # 眼睛Y坐标在高度的22%处
        target_chin_y = int(target_height * 0.58)   # 下巴Y坐标在高度的40%处
        return target_eyes_y, target_chin_y 

    # 4:3 的情况
    target_eyes_y = int(target_height * 0.40)       # 眼睛Y坐标在高度的37%处
    target_chin_y = int(target_height * 0.61)       # 下巴Y坐标在高度的50%处
    return target_eyes_y, target_chin_y

def generate_portrait_upper_body(image_path, output_path="portrait_upper_body.jpg", target_size=(1024, 1024), fill_blank=False):
    """
    检测图片中的人脸，并根据人脸位置、大小信息，生成指定大小的半身人像图片。
    具体来说，将人的眼睛放在高度330像素左右，下巴下缘在高度512像素左右。
    适合生成半身照效果。

    Args:
        image_path (str): 输入图片的路径。
        output_path (str): 输出人像图片的保存路径。
        target_size (tuple): 输出图片的目标尺寸 (宽度, 高度)。
    """

    target_eyes_y, target_chin_y = _get_upper_body_target_y_coords(target_size)
    _generate_portrait_base(image_path, output_path, target_size, target_eyes_y, target_chin_y, "半身", fill_blank)

def _get_half_body_target_y_coords(target_size):
    """
    根据目标尺寸计算目标眼睛Y坐标和目标下巴Y坐标
    用于 half body
    """
    target_width, target_height = target_size

    # 1:1 的情况
    if target_width == target_height:
        target_eyes_y = int(target_height * 0.22)   # 眼睛Y坐标在高度的32%处
        target_chin_y = int(target_height * 0.35)   # 下巴Y坐标在高度的54%处
        return target_eyes_y, target_chin_y

    # 3:4 的情况    
    if target_width * 4 == target_height * 3:
        target_eyes_y = int(target_height * 0.22)   # 眼睛Y坐标在高度的22%处
        target_chin_y = int(target_height * 0.35)    # 下巴Y坐标在高度的40%处
        return target_eyes_y, target_chin_y

    # 4:3 的情况
    target_eyes_y = int(target_height * 0.22)  # 眼睛Y坐标在高度的37%处
    target_chin_y = int(target_height * 0.34)   # 下巴Y坐标在高度的50%处
    return target_eyes_y, target_chin_y

def generate_portrait_half_body(image_path, output_path="portrait_half_body.jpg", target_size=(1024, 1024), fill_blank=False):
    """
    检测图片中的人脸，并根据人脸位置、大小信息，生成指定大小的上半身人像图片。
    具体来说，将人的眼睛放在高度360像素左右，下巴下缘在高度470像素左右。
    适合生成上半身照效果。

    Args:
        image_path (str): 输入图片的路径。
        output_path (str): 输出人像图片的保存路径。
        target_size (tuple): 输出图片的目标尺寸 (宽度, 高度)。
    """

    target_eyes_y, target_chin_y = _get_half_body_target_y_coords(target_size)
    _generate_portrait_base(image_path, output_path, target_size, 360, 470, "上半身", fill_blank)
