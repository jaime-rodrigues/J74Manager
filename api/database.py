import datetime
from fastapi import APIRouter, BackgroundTasks, Form
from fastapi.responses import JSONResponse

from services.backup import BackupManager

router = APIRouter(
    prefix="/database",
    tags=["Database"]
)

# Dependency placeholder
def get_backup_manager() -> BackupManager:
    from main import backup_manager
    return backup_manager

@router.post("/backup")
async def backup_database_endpoint(background_tasks: BackgroundTasks):
    """Triggers an asynchronous database backup in the background."""
    try:
        backup_manager = get_backup_manager()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"backup_{timestamp}.dump"
        
        # The task is async, so it will run in the background event loop
        background_tasks.add_task(backup_manager.backup_database, backup_file)
        
        return {"message": "Database backup process started.", "backup_file": backup_file}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

@router.post("/restore")
async def restore_database_endpoint(background_tasks: BackgroundTasks, filename: str = Form(None)):
    """Asynchronously restores the database from a backup file."""
    try:
        backup_manager = get_backup_manager()
        # get_latest_backup is sync and fast, so no need to make it async
        backup_to_restore = filename or backup_manager.get_latest_backup()
        
        if not backup_to_restore:
            return JSONResponse(status_code=404, content={"message": "No backup file found to restore."})
        
        background_tasks.add_task(backup_manager.restore_database, backup_to_restore)
        
        return {"message": "Database restore process started.", "restored_from": backup_to_restore}
    except FileNotFoundError as e:
        return JSONResponse(status_code=404, content={"message": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})
