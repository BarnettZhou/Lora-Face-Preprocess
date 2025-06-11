import face_recognition
import numpy as np

def check_image_initialized():
    """
    检查图像是否已经初始化
    """
    def decorator(func):
        def wrapper(instance, *args, **kwargs):
            if not hasattr(instance, 'image'):
                raise AttributeError(f"Image not initialized. Please load an image first.")
            value = getattr(instance, 'image')
            if value is None:
                raise ValueError(f"Image not initialized. Please load an image first.")
            return func(instance, *args, **kwargs)
        return wrapper
    return decorator

def check_image_module_selected():
    """
    检查图像模块是否已经选择
    """
    def decorator(func):
        def wrapper(instance, *args, **kwargs):
            if not hasattr(instance, 'image_module'):
                raise AttributeError(f"Image module not selected. Please select an image module first.")
            value = getattr(instance, 'image_module')
            if value is None:
                raise ValueError(f"Image module not selected. Please select an image module first.")
            if value not in ["pillow", "cv2"]:
                raise ValueError(f"Image module not supported. Please select an image module from pillow or cv2.")
            return func(instance, *args, **kwargs)
        return wrapper
    return decorator

class PortraitGenerator:

    image = None
    landmarks = None
    eyes_center = (0, 0)
    bottom_chin = (0, 0)
    target_size = (1024, 1024)

    # face_portrait_image = None
    # upper_body_portrait_image = None
    # half_body_portrait_image = None

    face_target_eyes_y = 0
    face_target_chin_y = 0
    upper_body_target_eyes_y = 0
    upper_body_target_chin_y = 0
    half_body_target_eyes_y = 0
    half_body_target_chin_y = 0

    fill_blank = False

    image_module = None
    image_lib = None

    def __init__(self, image_module="pillow"):
        if image_module not in ["pillow", "cv2"]:
            raise Exception('image_module must be pillow or cv2')
        self.image_module = image_module
        if image_module == "cv2":
            import cv2
            self.image_lib = cv2
        elif image_module == "pillow":
            from PIL import Image
            self.image_lib = Image

    def load_image(self, image_path):
        self.__load_image(image_path)
        self.__detect_face()
        self.__extract_key_points()
        return self

    def set_target_size(self, target_size):
        self.target_size = target_size
        return self

    def set_fill_blank(self, fill_blank):
        self.fill_blank = fill_blank
        return self

    @check_image_module_selected()
    @check_image_initialized()
    def generate_face_portrait(self):
        """
        生成脸部肖像
        """
        self.__caculate_face_target_y_coords()
        scale_factor, paste_coords = self.__calulate_scale_and_paste_coords(self.face_target_eyes_y, self.face_target_chin_y)
        if self.image_module == "cv2":
            return self.__resize_and_paste_image_cv2(scale_factor, paste_coords)
        elif self.image_module == "pillow":
            return self.__resize_and_paste_image_pillow(scale_factor, paste_coords)
        else:
            raise Exception('image_module must be pillow or cv2')

    @check_image_module_selected()
    @check_image_initialized()
    def generate_upper_body_portrait(self):
        """
        生成胸部以上肖像
        """
        self.__caculate_upper_body_target_y_coords()
        scale_factor, paste_coords = self.__calulate_scale_and_paste_coords(self.upper_body_target_eyes_y, self.upper_body_target_chin_y)
        if self.image_module == "cv2":
            return self.__resize_and_paste_image_cv2(scale_factor, paste_coords)
        elif self.image_module == "pillow":
            return self.__resize_and_paste_image_pillow(scale_factor, paste_coords)
        else:
            raise Exception('image_module must be pillow or cv2')

    @check_image_module_selected()
    @check_image_initialized()
    def generate_half_body_portrait(self):
        """
        生成半身肖像
        """
        self.__caculate_half_body_target_y_coords()
        scale_factor, paste_coords = self.__calulate_scale_and_paste_coords(self.half_body_target_eyes_y, self.half_body_target_chin_y)
        if self.image_module == "cv2":
            return self.__resize_and_paste_image_cv2(scale_factor, paste_coords)
        elif self.image_module == "pillow":
            return self.__resize_and_paste_image_pillow(scale_factor, paste_coords)
        else:
            raise Exception('image_module must be pillow or cv2')

    def save_image(self, image, path):
        """
        保存图像
        """
        if self.image_module == "cv2":
            cv2 = self.image_lib
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, bgr_image)
        elif self.image_module == "pillow":
            image.save(path)
        else:
            raise Exception('image_module must be pillow or cv2')
        return self

    def __load_image(self, image_path):
        """
        加载图像
        """
        self.image = face_recognition.load_image_file(image_path)
        return self

    def __detect_face(self):
        """
        检测人脸
        """
        face_locations = face_recognition.face_locations(self.image)
        if len(face_locations) == 0:
            raise Exception('No face detected')
        else:
            self.landmarks = face_recognition.face_landmarks(self.image)[0]
            return self

    def __extract_key_points(self):
        """
        提取关键点
        """
        landmarks = self.landmarks
        left_eye = np.mean(landmarks['left_eye'], axis=0).astype(int)
        right_eye = np.mean(landmarks['right_eye'], axis=0).astype(int)

        # 计算两眼中心点
        eyes_center_x = (left_eye[0] + right_eye[0]) / 2
        eyes_center_y = (left_eye[1] + right_eye[1]) / 2
        self.eyes_center = (eyes_center_x, eyes_center_y)

        # 找到下巴最下方的点
        chin_points = landmarks['chin']
        bottom_chin_y = max([p[1] for p in chin_points])
        bottom_chin_x = chin_points[np.argmax([p[1] for p in chin_points])][0]
        self.bottom_chin = (bottom_chin_x, bottom_chin_y)
        
        return self

    def __caculate_face_target_y_coords(self):
        """
        计算目标图像中眼睛和下巴的Y坐标
        用于脸部图像
        """
        target_width, target_height = self.target_size

        # 1:1 的情况
        if target_width == target_height:
            target_eyes_y = int(target_height * 0.5)    # 眼睛Y坐标在高度的一半
            target_chin_y = int(target_height * 0.84)   # 下巴Y坐标在高度的84%处
        
        # 3:4 的情况
        if target_width * 4 == target_height * 3:
            target_eyes_y = int(target_height * 0.37)   # 眼睛Y坐标在高度的37%处
            target_chin_y = int(target_height * 0.72)   # 下巴Y坐标在高度的72%处

        # 4:3 的情况
        target_eyes_y = int(target_height * 0.47)        # 眼睛Y坐标在高度的一半
        target_chin_y = int(target_height * 0.81)       # 下巴Y坐标在高度的87%处

        self.face_target_eyes_y = target_eyes_y
        self.face_target_chin_y = target_chin_y
        return self

    def __caculate_upper_body_target_y_coords(self):
        """
        根据目标尺寸计算目标眼睛Y坐标和目标下巴Y坐标
        用于胸部以上图像
        """
        target_width, target_height = self.target_size

        # 1:1 的情况
        if target_width == target_height:
            target_eyes_y = int(target_height * 0.32)   # 眼睛Y坐标在高度的32%处
            target_chin_y = int(target_height * 0.54)   # 下巴Y坐标在高度的54%处

        # 3:4 的情况
        if target_width * 4 == target_height * 3:
            target_eyes_y = int(target_height * 0.38)   # 眼睛Y坐标在高度的22%处
            target_chin_y = int(target_height * 0.58)   # 下巴Y坐标在高度的40%处

        # 4:3 的情况
        target_eyes_y = int(target_height * 0.40)       # 眼睛Y坐标在高度的37%处
        target_chin_y = int(target_height * 0.61)       # 下巴Y坐标在高度的50%处

        self.upper_body_target_eyes_y = target_eyes_y
        self.upper_body_target_chin_y = target_chin_y
        return self

    def __caculate_half_body_target_y_coords(self):
        """
        根据目标尺寸计算目标眼睛Y坐标和目标下巴Y坐标
        用于半身图像
        """
        target_width, target_height = self.target_size

        # 1:1 的情况
        if target_width == target_height:
            target_eyes_y = int(target_height * 0.22)   # 眼睛Y坐标在高度的32%处
            target_chin_y = int(target_height * 0.35)   # 下巴Y坐标在高度的54%处

        # 3:4 的情况    
        if target_width * 4 == target_height * 3:
            target_eyes_y = int(target_height * 0.22)   # 眼睛Y坐标在高度的22%处
            target_chin_y = int(target_height * 0.35)    # 下巴Y坐标在高度的40%处

        # 4:3 的情况
        target_eyes_y = int(target_height * 0.22)  # 眼睛Y坐标在高度的37%处
        target_chin_y = int(target_height * 0.34)   # 下巴Y坐标在高度的50%处

        self.half_body_target_eyes_y = target_eyes_y
        self.half_body_target_chin_y = target_chin_y
        return self

    def __calulate_scale_and_paste_coords(self, target_eyes_y, target_chin_y):
        """
        根据目标眼睛Y坐标和目标下巴Y坐标计算缩放因子和粘贴坐标
        """
        target_width = self.target_size[0]

        # 计算眼睛到下巴的距离
        eyes_to_chin_distance = self.bottom_chin[1] - self.eyes_center[1]
        target_eyes_to_chin_distance = target_chin_y - target_eyes_y

        # 计算缩放因子
        scale_factor = target_eyes_to_chin_distance / eyes_to_chin_distance

        # 计算粘贴坐标
        paste_x = int(target_width / 2 - self.eyes_center[0] * scale_factor)
        paste_y = int(target_eyes_y - self.eyes_center[1] * scale_factor)

        paste_coords = (paste_x, paste_y)

        return scale_factor, paste_coords
    
    def __resize_and_paste_image_cv2(self, scale_factor, paste_coords):
        """
        缩放并粘贴图像
        """
        target_w, target_h = self.target_size

        cv2 = self.image_lib

        # 缩放图像
        resized = cv2.resize(self.image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

        # 创建黑色背景图像
        result = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        # 粘贴位置
        paste_x, paste_y = paste_coords
        
        # 计算实际可粘贴的区域
        src_x1 = max(-paste_x, 0)  # 源图像起始x坐标
        src_y1 = max(-paste_y, 0)  # 源图像起始y坐标
        dst_x1 = max(paste_x, 0)   # 目标图像起始x坐标
        dst_y1 = max(paste_y, 0)   # 目标图像起始y坐标

        # 计算可复制的宽度和高度
        copy_w = min(resized.shape[1] - src_x1, target_w - dst_x1)
        copy_h = min(resized.shape[0] - src_y1, target_h - dst_y1)
        
        # 确保复制区域有效
        if copy_w > 0 and copy_h > 0:
            src_x2 = src_x1 + copy_w
            src_y2 = src_y1 + copy_h
            dst_x2 = dst_x1 + copy_w
            dst_y2 = dst_y1 + copy_h
            
            # 安全地复制图像区域
            result[dst_y1:dst_y2, dst_x1:dst_x2] = resized[src_y1:src_y2, src_x1:src_x2]

        has_blank_area = (
            dst_x1 > 0              # 左侧会有空白
            or dst_y1 > 0           # 顶部会有空白
            or dst_x2 < target_w    # 右侧会有空白
            or dst_y2 < target_h    # 底部会有空白
        )
        if has_blank_area and self.fill_blank:
            top = max(0, dst_y1)
            bottom = max(0, target_h - dst_y2)
            left = max(0, dst_x1)
            right = max(0, target_w - dst_x2)

            # 使用粘贴图像的边缘像素填充空白区域
            if top > 0 and src_y1 == 0:
                # 填充顶部：使用 resized 图像的顶部一行
                top_line = resized[0:1, :, :]
                result[0:top, :, :] = cv2.resize(top_line, (target_w, top), interpolation=cv2.INTER_NEAREST)

            if bottom > 0 and src_y2 == resized.shape[0]:
                # 填充底部：使用 resized 图像的底部一行
                bottom_line = resized[-1:, :, :]
                result[target_h - bottom:target_h, :, :] = cv2.resize(bottom_line, (target_w, bottom), interpolation=cv2.INTER_NEAREST)

            if left > 0 and src_x1 == 0:
                # 填充左侧：使用 resized 图像的最左一列
                left_line = resized[:, 0:1, :]
                result[:, 0:left, :] = cv2.resize(left_line, (left, target_h), interpolation=cv2.INTER_NEAREST)

            if right > 0 and src_x2 == resized.shape[1]:
                # 填充右侧：使用 resized 图像的最右一列
                right_line = resized[:, -1:, :]
                result[:, target_w - right:target_w, :] = cv2.resize(right_line, (right, target_h), interpolation=cv2.INTER_NEAREST)

        return result

    def __resize_and_paste_image_pillow(self, scale_factor, paste_coords):
        """
        缩放并粘贴图像
        """
        target_width, target_height = self.target_size
        Image = self.image_lib

        # 调整图片大小
        resized_height = int(self.image.shape[0] * scale_factor)
        resized_width = int(self.image.shape[1] * scale_factor)
        resized_image = Image.fromarray(self.image).resize((resized_width, resized_height), Image.Resampling.LANCZOS)

        # 创建一个新的图片对象，大小为目标尺寸
        result_image = Image.new('RGB', self.target_size)

        # 粘贴原图到目标图片上
        paste_x, paste_y = paste_coords
        result_image.paste(resized_image, (paste_x, paste_y))

        # 检查是否有空白区域，并根据fill_blank决定是否填充
        has_blank_area = (
            paste_x > 0                                 # 左侧会有空白
            or paste_y > 0                              # 顶部会有空白
            or paste_x + resized_width < target_width   # 右侧会有空白
            or paste_y + resized_height < target_height # 底部会有空白
        )
        if has_blank_area and self.fill_blank:
            result_array = np.array(result_image)

            # 填充水平方向的空白
            # 填充左边
            if paste_x > 0:
                left_edge = result_array[:, paste_x-1:paste_x+1, :]
                result_array[:, :paste_x, :] = np.repeat(left_edge, paste_x, axis=1)
            # 填充右边
            if paste_x + resized_width < target_width:
                right_edge = result_array[:, paste_x+resized_width-1:paste_x+resized_width, :]
                result_array[:, paste_x+resized_width:, :] = np.repeat(right_edge, target_width-paste_x-resized_width, axis=1)

            # 填充垂直方向的空白
            # 填充上边
            if paste_y > 0:
                top_edge = result_array[paste_y-1:paste_y+1, :, :]
                result_array[:paste_y, :, :] = np.repeat(top_edge, paste_y, axis=0)
            # 填充下边
            if paste_y + resized_height < target_height:
                bottom_edge = result_array[paste_y+resized_height-1:paste_y+resized_height, :, :]
                result_array[paste_y+resized_height:, :, :] = np.repeat(bottom_edge, target_height-paste_y-resized_height, axis=0)

            return Image.fromarray(result_array)
        else:
            # 不填充或没有空白区域，直接返回
            return result_image

if __name__ == '__main__':
    pg = PortraitGenerator('cv2')
    pg.load_image('e:/codes/face-preprocess/src/demo/President_Barack_Obama.jpg')

    face_image = pg.generate_face_portrait()
    pg.save_image(face_image, 'e:/codes/face-preprocess/src/demo-out/President_Barack_Obama_face.jpg')

    upper_body_image = pg.generate_upper_body_portrait()
    pg.save_image(upper_body_image, 'e:/codes/face-preprocess/src/demo-out/President_Barack_Obama_upper_body.jpg')

    half_body_image = pg.generate_half_body_portrait()
    pg.save_image(half_body_image, 'e:/codes/face-preprocess/src/demo-out/President_Barack_Obama_half_body.jpg')