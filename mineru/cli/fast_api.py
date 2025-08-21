import uuid
import os
import uvicorn
import click
from pathlib import Path
from glob import glob
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict
from loguru import logger
from base64 import b64encode
import datetime
import threading
import time
import json
import pickle
import signal
import sys
from contextlib import asynccontextmanager

from mineru.cli.common import aio_do_parse, read_fn, pdf_suffixes, image_suffixes
from mineru.utils.cli_parser import arg_parse
from mineru.version import __version__
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI"""
    # Startup
    global cleanup_thread, cleanup_stop_event
    
    # Setup signal handlers for graceful shutdown
    setup_signal_handlers()
    
    # Load existing task map from persistent storage
    load_task_map()
    
    cleanup_stop_event = threading.Event()
    cleanup_thread = threading.Thread(target=cleanup_old_tasks, daemon=True)
    cleanup_thread.start()
    logger.info("Cleanup thread started")
    
    yield
    
    # Shutdown
    if cleanup_stop_event:
        cleanup_stop_event.set()
        if cleanup_thread and cleanup_thread.is_alive():
            cleanup_thread.join(timeout=10)  # Wait up to 10 seconds for thread to finish
            logger.info("Cleanup thread stopped")
    
    # Save task map before shutdown
    save_task_map()


app = FastAPI(lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Global task map to store task information
task_map: Dict[str, Dict] = {}
task_map_lock = threading.Lock()

# Global variables for cleanup thread
cleanup_thread = None
cleanup_stop_event = None

# Persistence file path
TASK_MAP_FILE = "/mnt/hwfile/opendatalab/MinerU4S/zenghuazheng/mineru_store/task_map.pkl"

def save_task_map():
    """Save task map to persistent storage"""
    try:
        with task_map_lock:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(TASK_MAP_FILE), exist_ok=True)
            
            # Convert datetime objects to string for serialization
            serializable_task_map = {}
            for task_id, task_info in task_map.items():
                task_copy = task_info.copy()
                if task_copy.get("start_time"):
                    task_copy["start_time"] = task_copy["start_time"].isoformat()
                if task_copy.get("completion_time"):
                    task_copy["completion_time"] = task_copy["completion_time"].isoformat()
                serializable_task_map[task_id] = task_copy
            
            # Save using pickle for better datetime handling
            with open(TASK_MAP_FILE, 'wb') as f:
                pickle.dump(serializable_task_map, f)
            
            logger.info(f"Task map saved to {TASK_MAP_FILE} with {len(task_map)} tasks")
    except Exception as e:
        logger.error(f"Failed to save task map: {e}")

def load_task_map():
    """Load task map from persistent storage"""
    try:
        if os.path.exists(TASK_MAP_FILE):
            with open(TASK_MAP_FILE, 'rb') as f:
                loaded_task_map = pickle.load(f)
            
            # Convert string back to datetime objects
            with task_map_lock:
                for task_id, task_info in loaded_task_map.items():
                    if task_info.get("start_time"):
                        task_info["start_time"] = datetime.datetime.fromisoformat(task_info["start_time"])
                    if task_info.get("completion_time"):
                        task_info["completion_time"] = datetime.datetime.fromisoformat(task_info["completion_time"])
                    task_map[task_id] = task_info
            
            logger.info(f"Task map loaded from {TASK_MAP_FILE} with {len(task_map)} tasks")
        else:
            logger.info("No existing task map file found, starting with empty map")
    except Exception as e:
        logger.error(f"Failed to load task map: {e}")

def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown"""
    logger.info(f"Received signal {signum}, saving task map before exit...")
    try:
        save_task_map()
        logger.info("Task map saved successfully before exit")
    except Exception as e:
        logger.error(f"Failed to save task map during signal handling: {e}")
    
    # Exit gracefully
    sys.exit(0)

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    logger.info("Signal handlers setup completed")

def cleanup_old_tasks():
    """Clean up tasks that have been completed for more than 3 hours"""
    while not cleanup_stop_event.is_set():
        try:
            current_time = datetime.datetime.now()
            tasks_to_remove = []
            
            with task_map_lock:
                for task_id, task_info in task_map.items():
                    if task_info.get("status") == "completed":
                        completion_time = task_info.get("completion_time")
                        if completion_time:
                            time_diff = current_time - completion_time
                            if time_diff.total_seconds() > 3 * 3600:  # 3 hours
                                tasks_to_remove.append(task_id)
                
                # Remove old tasks and their storage paths
                for task_id in tasks_to_remove:
                    task_info = task_map.pop(task_id)
                    storage_path = task_info.get("storage_path")
                    if storage_path and os.path.exists(storage_path):
                        try:
                            import shutil
                            shutil.rmtree(storage_path)
                            logger.info(f"Cleaned up old task {task_id} storage: {storage_path}")
                        except Exception as e:
                            logger.error(f"Failed to clean up storage for task {task_id}: {e}")
            
            # Save task map after cleanup (only if there were changes)
            if tasks_to_remove:
                save_task_map()
            
            # Sleep for 1 hour or until stop event is set
            if cleanup_stop_event.wait(timeout=3600):
                break
        except Exception as e:
            logger.error(f"Error in cleanup thread: {e}")
            if cleanup_stop_event.wait(timeout=3600):
                break



def encode_image(image_path: str) -> str:
    """Encode image using base64"""
    with open(image_path, "rb") as f:
        return b64encode(f.read()).decode()


def get_infer_result(file_suffix_identifier: str, pdf_name: str, parse_dir: str) -> Optional[str]:
    """从结果文件中读取推理结果"""
    result_file_path = os.path.join(parse_dir, f"{pdf_name}{file_suffix_identifier}")
    if os.path.exists(result_file_path):
        with open(result_file_path, "r", encoding="utf-8") as fp:
            return fp.read()
    return None


@app.post(path="/file_parse",)
async def parse_pdf(
        files: List[UploadFile] = File(...),
        output_dir: str = Form("./output"),
        lang_list: List[str] = Form(["ch"]),
        backend: str = Form("pipeline"),
        parse_method: str = Form("auto"),
        formula_enable: bool = Form(True),
        table_enable: bool = Form(True),
        server_url: Optional[str] = Form(None),
        return_md: bool = Form(True),
        return_middle_json: bool = Form(False),
        return_model_output: bool = Form(False),
        return_content_list: bool = Form(False),
        return_images: bool = Form(False),
        start_page_id: int = Form(0),
        end_page_id: int = Form(99999),
):

    # 获取命令行配置参数
    config = getattr(app.state, "config", {})

    try:
        # 创建唯一的输出目录
        unique_dir = output_dir
        os.makedirs(unique_dir, exist_ok=True)

        # 处理上传的PDF文件
        pdf_file_names = []
        pdf_bytes_list = []

        for file in files:
            content = await file.read()
            file_path = Path(file.filename)

            # 如果是图像文件或PDF，使用read_fn处理
            if file_path.suffix.lower() in pdf_suffixes + image_suffixes:
                # 创建临时文件以便使用read_fn
                temp_path = Path(unique_dir) / file_path.name
                with open(temp_path, "wb") as f:
                    f.write(content)

                try:
                    pdf_bytes = read_fn(temp_path)
                    pdf_bytes_list.append(pdf_bytes)
                    pdf_file_names.append(file_path.stem)
                    os.remove(temp_path)  # 删除临时文件
                except Exception as e:
                    return JSONResponse(
                        status_code=400,
                        content={"error": f"Failed to load file: {str(e)}"}
                    )
            else:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Unsupported file type: {file_path.suffix}"}
                )


        # 设置语言列表，确保与文件数量一致
        actual_lang_list = lang_list
        if len(actual_lang_list) != len(pdf_file_names):
            # 如果语言列表长度不匹配，使用第一个语言或默认"ch"
            actual_lang_list = [actual_lang_list[0] if actual_lang_list else "ch"] * len(pdf_file_names)
        print(f"start_to_parse_pdf")
        # 调用异步处理函数
        await aio_do_parse(
            output_dir=unique_dir,
            pdf_file_names=pdf_file_names,
            pdf_bytes_list=pdf_bytes_list,
            p_lang_list=actual_lang_list,
            backend=backend,
            parse_method=parse_method,
            formula_enable=formula_enable,
            table_enable=table_enable,
            server_url=server_url,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_dump_md=return_md,
            f_dump_middle_json=return_middle_json,
            f_dump_model_output=return_model_output,
            f_dump_orig_pdf=False,
            f_dump_content_list=return_content_list,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
            **config
        )

        # 构建结果路径
        result_dict = {}
        for pdf_name in pdf_file_names:
            result_dict[pdf_name] = {}
            data = result_dict[pdf_name]

            if backend.startswith("pipeline"):
                parse_dir = os.path.join(unique_dir, pdf_name, parse_method)
            else:
                parse_dir = os.path.join(unique_dir, pdf_name, "vlm")

            if os.path.exists(parse_dir):
                if return_md:
                    data["md_content"] = get_infer_result(".md", pdf_name, parse_dir)
                if return_middle_json:
                    data["middle_json"] = get_infer_result("_middle.json", pdf_name, parse_dir)
                if return_model_output:
                    if backend.startswith("pipeline"):
                        data["model_output"] = get_infer_result("_model.json", pdf_name, parse_dir)
                    else:
                        data["model_output"] = get_infer_result("_model_output.txt", pdf_name, parse_dir)
                if return_content_list:
                    data["content_list"] = get_infer_result("_content_list.json", pdf_name, parse_dir)
                if return_images:
                    image_paths = glob(f"{parse_dir}/images/*.jpg")
                    data["images"] = {
                        os.path.basename(
                            image_path
                        ): f"data:image/jpeg;base64,{encode_image(image_path)}"
                        for image_path in image_paths
                    }
        return JSONResponse(
            status_code=200,
            content={
                "backend": backend,
                "version": __version__,
                "storage_path": unique_dir
            }
        )
    except Exception as e:
        logger.exception(e)
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process file: {str(e)}"}
        )



@app.post(path="/forward")
async def forward_parse(
        files: List[UploadFile] = File(...),
        username: str = Form(...),
        lang_list: List[str] = Form(["ch"]),
        backend: str = Form("pipeline"),
        parse_method: str = Form("auto"),
        formula_enable: bool = Form(True),
        table_enable: bool = Form(True),
        server_url: Optional[str] = Form(None),
        return_md: bool = Form(True),
        return_middle_json: bool = Form(False),
        return_model_output: bool = Form(False),
        return_content_list: bool = Form(False),
        return_images: bool = Form(False),
        start_page_id: int = Form(0),
        end_page_id: int = Form(99999),
):
    try:
        # Generate task ID and create storage directory
        task_id = str(uuid.uuid4())
        current_time = datetime.datetime.now()
        timestamp = current_time.strftime("%Y%m%d_%H%M%S")
        storage_path = f"/mnt/hwfile/opendatalab/MinerU4S/zenghuazheng/mineru_store/{username}/{timestamp}"
        
        # Create storage directory
        os.makedirs(storage_path, exist_ok=True)
        
        # Add task to map
        with task_map_lock:
            task_map[task_id] = {
                "username": username,
                "start_time": current_time,
                "completion_time": None,
                "storage_path": storage_path,
                "status": "processing",
                "total_files": len(files),
                "batches_processed": 0
            }
        
        # Split files into batches of at most 20
        batch_size = 20
        file_batches = [files[i:i + batch_size] for i in range(0, len(files), batch_size)]
        logger.info(f"forward_parse: Processing {len(files)} files in {len(file_batches)} batches for user {username}")
        
        # Split lang_list to match batches (if needed)
        actual_lang_list = lang_list
        if len(actual_lang_list) != len(files):
            actual_lang_list = [actual_lang_list[0] if actual_lang_list else "ch"] * len(files)
        
        lang_batches = [actual_lang_list[i:i + batch_size] for i in range(0, len(actual_lang_list), batch_size)]
        
        # Process each batch by calling parse_pdf
        for i, (file_batch, lang_batch) in enumerate(zip(file_batches, lang_batches)):
            logger.info(f"Processing batch {i+1}/{len(file_batches)} with {len(file_batch)} files")
            
            # Create form data for this batch
            form_data = {
                "output_dir": storage_path,
                "lang_list": lang_batch,
                "backend": backend,
                "parse_method": parse_method,
                "formula_enable": formula_enable,
                "table_enable": table_enable,
                "server_url": server_url,
                "return_md": return_md,
                "return_middle_json": return_middle_json,
                "return_model_output": return_model_output,
                "return_content_list": return_content_list,
                "return_images": return_images,
                "start_page_id": start_page_id,
                "end_page_id": end_page_id
            }
            
            # Call parse_pdf directly without storing results
            await parse_pdf(
                files=file_batch,
                **form_data
            )
            
            # Update batch progress
            with task_map_lock:
                if task_id in task_map:
                    task_map[task_id]["batches_processed"] = i + 1
        
        # Mark task as completed
        completion_time = datetime.datetime.now()
        with task_map_lock:
            if task_id in task_map:
                task_map[task_id]["completion_time"] = completion_time
                task_map[task_id]["status"] = "completed"
        
        # Return success response with task information
        return JSONResponse(
            status_code=200,
            content={
                "task_id": task_id,
                "message": f"Successfully processed {len(files)} files in {len(file_batches)} batches",
                "batches_processed": len(file_batches),
                "total_files": len(files),
                "storage_path": storage_path,
                "username": username,
                "start_time": current_time.isoformat(),
                "completion_time": completion_time.isoformat(),
                "version": __version__
            }
        )
    except Exception as e:
        logger.exception(e)
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process files: {str(e)}"}
        )


@app.get(path="/task_status/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a specific task"""
    with task_map_lock:
        if task_id in task_map:
            task_info = task_map[task_id].copy()
            # Convert datetime objects to ISO format for JSON serialization
            if task_info.get("start_time"):
                task_info["start_time"] = task_info["start_time"].isoformat()
            if task_info.get("completion_time"):
                task_info["completion_time"] = task_info["completion_time"].isoformat()
            return JSONResponse(
                status_code=200,
                content=task_info
            )
        else:
            return JSONResponse(
                status_code=404,
                content={"error": f"Task {task_id} not found"}
            )


@app.get(path="/user_tasks/{username}")
async def get_user_tasks(username: str):
    """Get all tasks for a specific user"""
    with task_map_lock:
        user_tasks = {}
        for task_id, task_info in task_map.items():
            if task_info.get("username") == username:
                task_copy = task_info.copy()
                # Convert datetime objects to ISO format for JSON serialization
                if task_copy.get("start_time"):
                    task_copy["start_time"] = task_copy["start_time"].isoformat()
                if task_copy.get("completion_time"):
                    task_copy["completion_time"] = task_copy["completion_time"].isoformat()
                user_tasks[task_id] = task_copy
        
        return JSONResponse(
            status_code=200,
            content={
                "username": username,
                "tasks": user_tasks,
                "total_tasks": len(user_tasks)
            }
        )


@app.get(path="/all_tasks")
async def get_all_tasks():
    """Get all tasks in the system"""
    with task_map_lock:
        all_tasks = {}
        for task_id, task_info in task_map.items():
            task_copy = task_info.copy()
            # Convert datetime objects to ISO format for JSON serialization
            if task_copy.get("start_time"):
                task_copy["start_time"] = task_copy["start_time"].isoformat()
            if task_copy.get("completion_time"):
                task_copy["completion_time"] = task_copy["completion_time"].isoformat()
            all_tasks[task_id] = task_copy
        
        return JSONResponse(
            status_code=200,
            content={
                "total_tasks": len(all_tasks),
                "tasks": all_tasks
            }
        )


@app.delete(path="/task/{task_id}")
async def delete_task(task_id: str):
    """Manually delete a specific task and its storage"""
    with task_map_lock:
        if task_id in task_map:
            task_info = task_map.pop(task_id)
            storage_path = task_info.get("storage_path")
            
            # Delete storage directory if it exists
            if storage_path and os.path.exists(storage_path):
                try:
                    import shutil
                    shutil.rmtree(storage_path)
                    logger.info(f"Manually deleted task {task_id} storage: {storage_path}")
                except Exception as e:
                    logger.error(f"Failed to delete storage for task {task_id}: {e}")
                    return JSONResponse(
                        status_code=500,
                        content={"error": f"Failed to delete storage: {str(e)}"}
                    )
            
            return JSONResponse(
                status_code=200,
                content={
                    "message": f"Task {task_id} deleted successfully",
                    "deleted_task": task_info
                }
            )
        else:
            return JSONResponse(
                status_code=404,
                content={"error": f"Task {task_id} not found"}
            )


@app.post(path="/save_tasks")
async def save_tasks():
    """Manually save current task map to persistent storage"""
    try:
        save_task_map()
        return JSONResponse(
            status_code=200,
            content={
                "message": "Task map saved successfully",
                "total_tasks": len(task_map)
            }
        )
    except Exception as e:
        logger.error(f"Failed to save tasks: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to save tasks: {str(e)}"}
        )


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.pass_context
@click.option('--host', default='127.0.0.1', help='Server host (default: 127.0.0.1)')
@click.option('--port', default=8000, type=int, help='Server port (default: 8000)')
@click.option('--reload', is_flag=True, help='Enable auto-reload (development mode)')
def main(ctx, host, port, reload, **kwargs):

    kwargs.update(arg_parse(ctx))

    # 将配置参数存储到应用状态中
    app.state.config = kwargs

    """启动MinerU FastAPI服务器的命令行入口"""
    print(f"Start MinerU FastAPI Service: http://{host}:{port}")
    print("The API documentation can be accessed at the following address:")
    print(f"- Swagger UI: http://{host}:{port}/docs")
    print(f"- ReDoc: http://{host}:{port}/redoc")
    print("Press Ctrl+C to stop the server")

    try:
        uvicorn.run(
            "mineru.cli.fast_api:app",
            host=host,
            port=port,
            reload=reload
        )
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, saving task map...")
        try:
            save_task_map()
            logger.info("Task map saved successfully before exit")
        except Exception as e:
            logger.error(f"Failed to save task map during KeyboardInterrupt: {e}")
        sys.exit(0)


if __name__ == "__main__":
    main()