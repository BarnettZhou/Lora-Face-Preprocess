import os
import glob
from pathlib import Path

def get_image_files(directory):
    """
    获取目录中的所有图片文件
    """
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIFF']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory, ext)))
    
    # 去除重复文件（使用set去重，然后转回list并排序）
    return sorted(list(set(image_files)))

def ensure_output_directory(output_dir):
    """
    确保输出目录存在
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)