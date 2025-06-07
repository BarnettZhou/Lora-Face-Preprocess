import os
import glob
from pathlib import Path
from utils.portrait import generate_portrait_face, generate_portrait_upper_body, generate_portrait_half_body

def get_src_directories():
    """
    获取./src目录下的所有子目录
    """
    src_path = Path('./src')
    if not src_path.exists():
        print("错误：./src目录不存在")
        return []
    
    directories = []
    for item in src_path.iterdir():
        if item.is_dir():
            directories.append({
                'name': item.name,
                'value': str(item.relative_to('.'))
            })
    
    return directories

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

def generate_portraits(selected_dir, generation_option):
    """
    根据选择的选项生成人像图片
    """
    # 获取图片文件
    image_files = get_image_files(selected_dir)
    print(image_files)
    
    if not image_files:
        print(f"在目录 {selected_dir} 中未找到图片文件")
        return
    
    # 创建输出目录
    output_base = os.path.join('output', os.path.basename(selected_dir))
    ensure_output_directory(output_base)
    
    print(f"找到 {len(image_files)} 个图片文件，开始处理...")
    
    # 生成face图片（所有选项都需要）
    print("\n正在生成face图片...")
    for i, image_file in enumerate(image_files, 1):
        output_path = os.path.join(output_base, f"face_{i:03d}.jpg")
        try:
            generate_portrait_face(image_file, output_path)
        except Exception as e:
            print(f"处理 {image_file} 时出错：{e}")
    
    # 根据选项生成额外的图片
    if generation_option in ['upper_body', 'all']:
        print("\n正在生成upper body图片...")
        for i, image_file in enumerate(image_files, 1):
            output_path = os.path.join(output_base, f"upper_body_{i:03d}.jpg")
            try:
                generate_portrait_upper_body(image_file, output_path)
            except Exception as e:
                print(f"处理 {image_file} 时出错：{e}")
    
    if generation_option == 'all':
        print("\n正在生成half body图片...")
        for i, image_file in enumerate(image_files, 1):
            output_path = os.path.join(output_base, f"half_body_{i:03d}.jpg")
            try:
                generate_portrait_half_body(image_file, output_path)
            except Exception as e:
                print(f"处理 {image_file} 时出错：{e}")
    
    print(f"\n所有图片处理完成！输出目录：{output_base}")

def main():
    """
    主函数
    """
    print("人像图片批量生成工具")
    print("=" * 30)
    
    # 获取src目录下的所有子目录
    directories = get_src_directories()
    
    if not directories:
        print("在./src目录下未找到任何子目录")
        return
    
    # 选择目录
    try:
        from InquirerPy import inquirer
        
        selected_dir = inquirer.select(
            message="请选择要处理的目录：",
            choices=[
                {"name": dir_info['name'], "value": dir_info['value']} 
                for dir_info in directories
            ],
        ).execute()
        
        print(f"已选择目录：{selected_dir}")
        
        # 选择生成选项
        generation_option = inquirer.select(
            message="请选择生成选项：",
            choices=[
                {"name": "仅生成face", "value": "face_only"},
                {"name": "额外生成upper body", "value": "upper_body"},
                {"name": "额外生成upper body和half body", "value": "all"}
            ],
        ).execute()
        
        print(f"已选择生成选项：{generation_option}")
        
        # 开始生成图片
        generate_portraits(selected_dir, generation_option)
        
    except ImportError:
        print("错误：请先安装InquirerPy库")
        print("运行命令：pip install InquirerPy")
    except KeyboardInterrupt:
        print("\n操作已取消")
    except Exception as e:
        print(f"发生错误：{e}")

if __name__ == "__main__":
    main()