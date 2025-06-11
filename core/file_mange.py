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

def rename_images_by_creation_time(directory):
    """
    将目录中的所有图片按照创建时间正序重命名为三位数字格式
    """
    try:
        # 获取目录中的所有图片文件
        image_files = get_image_files(directory)
        
        if not image_files:
            return {"success": False, "message": "目录中没有找到图片文件"}
        
        # 获取文件信息并按创建时间排序
        file_info = []
        for file_path in image_files:
            stat = os.stat(file_path)
            file_info.append({
                'path': file_path,
                'name': os.path.basename(file_path),
                'creation_time': stat.st_ctime,
                'extension': os.path.splitext(file_path)[1].lower()
            })
        
        # 按创建时间排序
        file_info.sort(key=lambda x: x['creation_time'])
        
        # 重命名文件
        renamed_files = []
        for i, info in enumerate(file_info, 1):
            old_path = info['path']
            new_name = f"{i:03d}{info['extension']}"
            new_path = os.path.join(directory, new_name)
            
            # 如果新文件名已存在且不是当前文件，则跳过
            if os.path.exists(new_path) and old_path != new_path:
                continue
                
            # 重命名文件
            if old_path != new_path:
                os.rename(old_path, new_path)
                renamed_files.append({
                    'old_name': info['name'],
                    'new_name': new_name
                })
        
        return {
            "success": True, 
            "message": f"成功重命名 {len(renamed_files)} 个文件",
            "renamed_files": renamed_files
        }
        
    except Exception as e:
        return {"success": False, "message": f"重命名失败: {str(e)}"}