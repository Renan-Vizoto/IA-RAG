from fastapi import APIRouter, Depends
from app.core.entities.file import File
from io import BytesIO
from fastapi import UploadFile
from app.core.services.bucket_service import BucketService
from app.infrastructure.clients.bucket_client import client


router = APIRouter(
    prefix="/files",
    tags=["files"],
    responses={404: {"description": "Not found"}},
)

bucketService = BucketService(client)

@router.post("/upload-file")
async def upload_file(file: UploadFile):
    file_obj = File(BytesIO(file.file.read()),file.filename, file.content_type)
    file.file.close()
    file_name = bucketService.add_object("bronze", file_obj)
    return { "status": "file created with success", "file_name" : file_name }

@router.delete("/remove-file")
async def remove_file(file_name: str):
    bucketService.remove_object("bronze", file_name)
    return { "status": "file removed with success" }

@router.put("/update-file")
async def update_file(file_name: str, file: UploadFile):
    bucketService.update_object("bronze", file_name, file.file, file.size)
    return { "status": "file updated with success" }