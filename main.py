from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from typing import List, Dict, Any, Optional
import asyncio
import os
import uuid
from pathlib import Path
import threading
from core.run_as_task import generate_portraits
from core.config import Config
from fastapi.responses import FileResponse
import mimetypes

from models.TaskRequest import TaskRequest
from models.ConfigRequest import ConfigRequest

from concurrent.futures import ThreadPoolExecutor

# 在文件顶部添加线程池
executor = ThreadPoolExecutor(max_workers=1)

app = FastAPI(title="人脸预处理工具", description="基于FastAPI的人脸预处理Web应用")

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 全局变量存储任务状态和WebSocket连接
tasks: Dict[str, Dict] = {}
websocket_connections: Dict[str, WebSocket] = {}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """返回主页面"""
    return FileResponse('static/index.html')

@app.post("/create_task")
async def create_task(request: TaskRequest):

    """创建处理任务"""
    try:
        # 生成任务ID
        task_id = str(uuid.uuid4())
        
        # 解析尺寸
        size_mapping = {
            "512x512": (512, 512),
            "768x768": (768, 768),
            "1024x1024": (1024, 1024),
            "576x768": (576, 768),
            "768x1024": (768, 1024),
            "768x576": (768, 576),
            "1024x768": (1024, 768)
        }
        
        target_size = size_mapping.get(request.size, (1024, 1024))
        
        # 创建任务配置
        config = {
            "src_dir": request.src_dir,
            "output_dir": request.output_dir,
            "threshold": request.threshold,
            "size": {
                "width": target_size[0],
                "height": target_size[1]
            },
            "format": request.format,
            "types": request.types,
            "center": request.center,
            "blank": request.blank
        }
        
        # 初始化任务状态
        tasks[task_id] = {
            "id": task_id,
            "status": "pending",
            "progress": 0,
            "message": "任务已创建",
            "config": config,
            "results": []
        }
        
        return {
            "success": True,
            "message": "任务创建成功",
            "data": {
                "task_id": task_id,
                "status": "pending",
                "progress": 0
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建任务失败: {str(e)}")

@app.websocket("/progress/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    """WebSocket连接处理任务进度"""
    await websocket.accept()
    websocket_connections[task_id] = websocket
    
    try:
        if task_id not in tasks:
            await websocket.send_json({"type": "error", "message": "任务不存在"})
            return
            
        # 发送初始状态
        task = tasks[task_id]
        await websocket.send_json({
            "type": "status",
            "status": task["status"],
            "progress": task["progress"],
            "message": task["message"]
        })
        
        # 如果任务还未开始，启动任务
        if task["status"] == "pending":
            # 在后台线程中运行任务
            thread = threading.Thread(target=run_task_in_background, args=(task_id,))
            thread.start()
        
        # 保持连接
        while True:
            try:
                await websocket.receive_text()
            except WebSocketDisconnect:
                break
                
    except WebSocketDisconnect:
        pass
    finally:
        if task_id in websocket_connections:
            del websocket_connections[task_id]

def run_task_in_background(task_id: str):
    """在后台运行任务"""
    try:
        task = tasks[task_id]
        config = task["config"]
        
        # 更新任务状态
        update_task_status(task_id, "running", 0, "开始处理图片...")
        
        # 获取图片文件列表
        from core.run_as_task import get_image_files
        image_files = get_image_files(config["src_dir"])
        
        if not image_files:
            update_task_status(task_id, "failed", 0, "未找到图片文件")
            return
            
        total_files = len(image_files)
        processed_files = 0
        
        # 确保输出目录存在
        from core.run_as_task import ensure_output_directory
        ensure_output_directory(config["output_dir"])
        
        # 处理每个图片
        for i, image_file in enumerate(image_files):
            try:
                filename = os.path.basename(image_file)
                update_task_status(task_id, "running", 
                                 int((i / total_files) * 100), 
                                 f"正在处理: {filename}")
                
                # 根据类型生成不同的图片
                from core.portrait import generate_portrait_face, generate_portrait_upper_body, generate_portrait_half_body
                
                target_size = (config["size"]["width"], config["size"]["height"])
                
                for img_type in config["types"]:
                    output_filename = f"{img_type}_{i+1:03d}.{config['format']}"
                    output_path = os.path.join(config["output_dir"], output_filename)
                    
                    if img_type == "face":
                        generate_portrait_face(image_file, output_path, target_size)
                    elif img_type == "upper_body":
                        generate_portrait_upper_body(image_file, output_path, target_size)
                    elif img_type == "half_body":
                        generate_portrait_half_body(image_file, output_path, target_size)
                        
                    # 记录处理结果
                    task["results"].append({
                        "source": filename,
                        "output": output_filename,
                        "type": img_type,
                        "status": "success"
                    })
                    
                processed_files += 1
                
            except Exception as e:
                # 记录失败结果
                task["results"].append({
                    "source": os.path.basename(image_file),
                    "output": "",
                    "type": "error",
                    "status": "failed",
                    "error": str(e)
                })
                
        # 任务完成
        update_task_status(task_id, "completed", 100, 
                         f"处理完成! 共处理 {processed_files}/{total_files} 个文件")
        
    except Exception as e:
        update_task_status(task_id, "failed", 0, f"任务执行失败: {str(e)}")

def update_task_status(task_id: str, status: str, progress: int, message: str):
    """更新任务状态并通过WebSocket发送"""
    if task_id in tasks:
        tasks[task_id]["status"] = status
        tasks[task_id]["progress"] = progress
        tasks[task_id]["message"] = message
        
        # 通过WebSocket发送更新
        if task_id in websocket_connections:
            websocket = websocket_connections[task_id]
            try:
                # 使用线程池在事件循环中发送消息
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(websocket.send_json({
                    "type": "progress",
                    "status": status,
                    "progress": progress,
                    "message": message
                }))
                loop.close()
            except Exception as e:
                print(f"WebSocket发送失败: {e}")
            except:
                pass

@app.post("/save_config")
async def save_config(request: ConfigRequest):
    """保存自定义配置"""
    try:
        config_manager = Config()
        
        config_data = {
            "src_dir": request.src_dir,
            "output_dir": request.output_dir,
            "threshold": request.threshold,
            "size": request.size,
            "format": request.format,
            "types": request.types,
            "center": request.center,
            "blank": request.blank
        }
        
        # 保存配置
        config_manager.save_config(request.name, config_data)
        
        return {
            "success": True,
            "message": "配置保存成功",
            "data": {"name": request.name}
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存配置失败: {str(e)}")

@app.get("/list_images/{path:path}")
async def list_images(path: str):
    """获取指定目录下的图片列表"""
    try:
        from core.run_as_task import get_image_files
        image_files = get_image_files(path)
        
        images = []
        for img_path in image_files:
            stat = os.stat(img_path)
            images.append({
                "name": os.path.basename(img_path),
                "path": img_path,
                "size": stat.st_size,
                "modified": stat.st_mtime
            })
            
        return {
            "success": True,
            "data": images
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取图片列表失败: {str(e)}")

@app.get("/task_status/{task_id}")
async def get_task_status(task_id: str):
    """获取任务状态"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
        
    return {
        "success": True,
        "data": tasks[task_id]
    }

@app.get("/serve_image/{path:path}")
async def serve_image(path: str):
    """提供图片文件服务"""
    try:
        # 确保路径安全，防止目录遍历攻击
        if ".." in path:
            raise HTTPException(status_code=400, detail="无效的文件路径")
        
        # 检查文件是否存在
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="文件不存在")
        
        # 检查是否为图片文件
        mime_type, _ = mimetypes.guess_type(path)
        if not mime_type or not mime_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="不是有效的图片文件")
        
        return FileResponse(path, media_type=mime_type)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取图片失败: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)