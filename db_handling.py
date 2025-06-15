from sqlalchemy import Column, String, JSON, DateTime
from sqlalchemy.sql import func
from db_init import Base

class FaceEncoding(Base):
    __tablename__ = "face_encodings"

    username = Column(String, primary_key=True, index=True)
    encoding = Column(JSON, nullable=False)
    registered_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
