import os
import glob
from pathlib import Path
from core.portrait import generate_portrait_face, generate_portrait_upper_body, generate_portrait_half_body
from core.config import Config

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

def generate_portraits(config):
    """
    根据选择的选项生成人像图片
    """
    # 获取图片文件
    image_files = get_image_files(config['src_dir'])
    
    if not image_files:
        print(f"在目录 {config['src_dir']} 中未找到图片文件")
        return
    
    # 创建输出目录
    output_base = os.path.join('output', os.path.basename(config['src_dir']))
    ensure_output_directory(output_base)
    
    print(f"找到 {len(image_files)} 个图片文件，开始处理...")
    
    # 生成脸部图片
    if "face" in config["type"]:
        print("\n正在生成face图片...")
        for i, image_file in enumerate(image_files, 1):
            output_path = os.path.join(output_base, f"face_{i:03d}.jpg")
            try:
                generate_portrait_face(image_file, output_path)
            except Exception as e:
                print(f"处理 {image_file} 时出错：{e}")
    
    # 根据选项生成额外的图片
    if "upper_body" in config["type"]:
        print("\n正在生成upper body图片...")
        for i, image_file in enumerate(image_files, 1):
            output_path = os.path.join(output_base, f"upper_body_{i:03d}.jpg")
            try:
                generate_portrait_upper_body(image_file, output_path)
            except Exception as e:
                print(f"处理 {image_file} 时出错：{e}")
    
    if "half_body" in config["type"]:
        print("\n正在生成half body图片...")
        for i, image_file in enumerate(image_files, 1):
            output_path = os.path.join(output_base, f"half_body_{i:03d}.jpg")
            try:
                generate_portrait_half_body(image_file, output_path)
            except Exception as e:
                print(f"处理 {image_file} 时出错：{e}")
    
    print(f"\n所有图片处理完成！输出目录：{output_base}")