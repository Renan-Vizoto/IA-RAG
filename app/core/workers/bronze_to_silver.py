from app.core.services.bucket_service import BucketService
from app.core.services.extractor_service import ExtractTextService
import os

from apscheduler.schedulers.asyncio import AsyncIOScheduler

class BronzeToSilverWorker:

    def __init__(self, bucket: BucketService, extractor: ExtractTextService):
        self._bucket = bucket
        self._extractor = extractor


    async def run(self):

        files = await self._bucket.list_objects("bronze", None)

        for file_meta in files:
            print(file_meta.object_name)
            file_stream = await self._bucket.get_object(file_meta.object_name)

            file_name, ext = os.path.splitext(file_meta.object_name)
            file_type = str(ext).removeprefix(".")

            text = self._extractor.process(file_stream, file_type)

            if not text:
                print(f"No means to extract from {file_meta.object_name}")
                continue

            print(f"extracted txt size {len(text)}")
            await self._bucket.save_as_txt(text, file_name, "silver")

        print("extracted with success")

def start_worker(bucket_service: BucketService, extrac_service: ExtractTextService):
    worker = BronzeToSilverWorker(bucket_service, extrac_service)
    scheduler = AsyncIOScheduler()
    scheduler.add_job(worker.run, "interval", minutes=5)
    scheduler.start()