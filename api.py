
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any 
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# --- 核心模块导入 ---
# 【修正】从正确的 vclip_service 路径导入我们的模块
import sys
import os
sys.path.append(os.path.abspath('service'))
from database import create_task as db_create_task
from database import get_task as db_get_task
from database import update_task_as_completed, update_task_as_failed
from tasks import process_video_task

# ==========================================================
#              【保留】您不想改动的部分
# ==========================================================
app = FastAPI(root_path="/vc")

app.mount("/swagger-static", StaticFiles(directory="static/swagger"), name="swagger")

@app.get("/docs", include_in_schema=False) # 在自动生成的文档中隐藏这个自定义接口
async def custom_swagger_ui():
    return FileResponse("static/swagger/index.html")
# ==========================================================


# --- API 数据模型 (Pydantic) ---

# 【修正】请求模型，允许 configs 接收任意类型的值
class TaskRequest(BaseModel):
    video_url: str
    api_keys: Optional[Dict[str, str]] = None
    configs: Optional[Dict[str, Any]] = None

class TaskResponse(BaseModel):
    task_id: str
    status: str

# 【修正】回调负载模型，使其与 core_logic.py 发送的数据结构完全匹配
class CallbackPayload(BaseModel):
    task_id: str
    status: str
    results: Dict[str, Any]


# --- API 接口定义 ---

@app.post("/tasks/", response_model=TaskResponse, status_code=202)
def submit_task(req: TaskRequest):
    """接收新的视频处理任务，并将其推送到Celery队列。"""
    
    # 【优化】调用 database.py 中的函数来创建任务
    task_id = db_create_task(req.video_url)
    if not task_id:
        raise HTTPException(status_code=500, detail="Failed to create task in database.")

    api_keys_to_pass = req.api_keys if req.api_keys is not None else {}
    configs_to_pass = req.configs if req.configs is not None else {}

    # 异步调用Celery任务
    process_video_task.delay(
        task_id=task_id,
        video_url=req.video_url,
        api_keys=api_keys_to_pass,
        configs=configs_to_pass
    )

    return {"task_id": task_id, "status": "pending"}

@app.get("/tasks/{task_id}")
def get_task_status(task_id: str):
    """根据 task_id 查询任务的状态和结果。"""
    
    # 【优化】调用 database.py 中的函数来查询
    task = db_get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task

@app.post("/tasks/callback")
def task_callback(payload: CallbackPayload):
    """【内部接口】接收来自 Celery Worker 的回调通知，更新数据库。"""
    print(f"收到回调: task_id={payload.task_id}, status={payload.status}")
    
    # 【优化】调用 database.py 中的函数来更新
    if payload.status == "completed":
        update_task_as_completed(payload.task_id, payload.results)
    elif payload.status == "failed":
        error_message = payload.results.get("error", "Unknown error from worker.")
        update_task_as_failed(payload.task_id, error_message)
    
    return {"message": "Callback received and processed."}
