from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
from uuid import uuid4
from datetime import datetime
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pymongo import MongoClient
from service.tasks import process_video_task   

app = FastAPI(root_path="/vc")

app.mount("/swagger-static", StaticFiles(directory="static/swagger"), name="swagger")

@app.get("/docs")
async def custom_swagger_ui():
    return FileResponse("static/swagger/index.html")

# MongoDB连接
client = MongoClient("mongodb://localhost:27017")
db = client["vclip"]
tasks_collection = db["tasks"]

# 请求模型
class CreateTaskRequest(BaseModel):
    video_url: str
    api_keys: Optional[Dict[str, str]] = {}
    configs: Optional[Dict[str, str]] = {}

class CallbackRequest(BaseModel):
    task_id: str
    status: str
    result_url: Optional[str] = None

# 创建任务
@app.post("/tasks/")
def create_task(req: CreateTaskRequest):
    task_id = str(uuid4())

    # 新建数据库记录
    tasks_collection.insert_one({
        "task_id": task_id,
        "video_url": req.video_url,
        "api_keys": req.api_keys,
        "configs": req.configs,
        "status": "pending",
        "created_at": datetime.utcnow()
    })

    # 异步调用Celery
    process_video_task.delay(
        task_id,
        req.video_url,
        req.api_keys,
        req.configs
    )

    return {
        "task_id": task_id,
        "status": "queued"
    }

# 查询任务
@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    task = tasks_collection.find_one({"task_id": task_id})
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "task_id": task["task_id"],
        "status": task["status"],
        "result_url": task.get("result_url")
    }

# 回调接口
@app.post("/tasks/callback")
def task_callback(req: CallbackRequest):
    result = tasks_collection.update_one(
        {"task_id": req.task_id},
        {"$set": {
            "status": req.status,
            "result_url": req.result_url,
            "completed_at": datetime.utcnow()
        }}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Task not found")

    return {"message": "Task status updated"}
