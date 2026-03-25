import asyncio
from minio.datatypes import Object
from typing import Iterator
from app.core.entities.file import File
import time
import os 
from io import BytesIO
from minio import Minio

class BucketService: 

    def __init__(self, client: Minio):
        self._client = client

    async def list_objects(self,bucket:str, prefix:str | None) -> Iterator[Object]:
        return await asyncio.to_thread(
            self._client.list_objects, bucket_name=bucket, prefix=prefix
        )
        
    async def get_object(self, file_name:str, bucket:str = "bronze"):
        def _read():
            response = None
            try:
                response = self._client.get_object(bucket, file_name)
                return BytesIO(response.read())
            finally:
                if response is not None:
                    response.close()
                    response.release_conn()

        return await asyncio.to_thread(_read)
        
    async def list_buckets(self):
        return await asyncio.to_thread(self._client.list_buckets)
    
    async def add_object(self, bucket:str, file: File):
        obj_name = self._gen_name(file_name=file.filename)
        def _add():
            self._client.put_object(bucket_name=bucket, object_name=obj_name, data=file.data, length=file.size, content_type=file.content_type)
            return obj_name

        await asyncio.to_thread(_add)

        return obj_name

    async def remove_object(self, bucket:str, obj_name:str):
        return await asyncio.to_thread(self._client.remove_object, bucket_name=bucket, object_name=obj_name)

    async def save_as_txt(self, text: list[str], name: str, target_bucket: str):  
        file = File(BytesIO("\n".join(text).encode("utf-8")), name+".txt", "text")
        return await self.add_object(target_bucket, file)
    
    def _gen_name(self, file_name:str):
        root, extension = os.path.splitext(file_name)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        return root + timestr + extension