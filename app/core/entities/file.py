from io import BytesIO

class File():
    data: BytesIO
    filename: str
    size: int
    content_type: str

    def __init__(self,data: BytesIO, filename: str, content_type: str):
        self.data = data
        self.filename = filename
        self.content_type = content_type
        self.size = data.getbuffer().nbytes
