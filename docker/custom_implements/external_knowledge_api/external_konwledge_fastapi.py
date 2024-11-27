from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from jose import JWTError, jwt
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")
POSTGRES_DB = os.getenv("POSTGRES_DB")

SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = os.getenv("JWT_ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = 30

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GOOGLE_API_KEY,
)

connection = f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

app = FastAPI()
security = HTTPBearer()


# 添加錯誤響應模型
class ErrorResponse(BaseModel):
    error_code: int
    error_msg: str


# 定義錯誤代碼
ERROR_CODES = {
    "INVALID_AUTH_FORMAT": 1001,
    "AUTH_FAILED": 1002,
    "KNOWLEDGE_NOT_FOUND": 2001,
}


# 基础模型保持不变
class RetrievalSetting(BaseModel):
    top_k: int = Field(gt=0)
    score_threshold: float = Field(ge=0.0, le=1.0)


class RetrievalRequest(BaseModel):
    knowledge_id: str
    query: str
    retrieval_setting: RetrievalSetting


class Record(BaseModel):
    content: str
    score: float
    title: str
    metadata: Optional[Dict] = None


class RetrievalResponse(BaseModel):
    records: List[Record]


# Token请求模型
class TokenRequest(BaseModel):
    collection_name: str


def create_access_token(data: dict):
    encoded_jwt = jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error_code": ERROR_CODES["INVALID_AUTH_FORMAT"],
                "error_msg": "無效的 Authorization 頭格式。預期格式為 'Bearer '",
            },
        )
    try:
        payload = jwt.decode(
            credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM]
        )
        collection_name = payload.get("collection_name")
        if collection_name is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error_code": ERROR_CODES["AUTH_FAILED"],
                    "error_msg": "授權失敗",
                },
            )
        return {"collection_name": collection_name}
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error_code": ERROR_CODES["AUTH_FAILED"], "error_msg": "授權失敗"},
        )


@app.post("/token", response_model=Dict[str, str])
async def create_token(request: TokenRequest):
    access_token = create_access_token(
        data={"collection_name": request.collection_name}
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.post(
    "/retrieval",
    response_model=RetrievalResponse,
    responses={403: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def retrieval(
    request: RetrievalRequest, token_data: dict = Depends(verify_token)
):
    try:
        collection_name = request.knowledge_id
        query = request.query
        retrieval_setting = request.retrieval_setting

        if collection_name != token_data["collection_name"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error_code": ERROR_CODES["AUTH_FAILED"],
                    "error_msg": "Knowledge ID 與 token 的 collection name 不匹配",
                },
            )

        vector_store = PGVector(
            embeddings=embeddings,
            collection_name=collection_name,
            connection=connection,
            use_jsonb=True,
        )

        # 嘗試進行向量搜索
        try:
            results = vector_store.similarity_search_with_score(
                query, k=retrieval_setting.top_k
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error_code": ERROR_CODES["KNOWLEDGE_NOT_FOUND"],
                    "error_msg": "知識庫不存在",
                },
            )

        filtered_results = [
            result
            for result in results
            if result[1] > retrieval_setting.score_threshold
        ]
        sorted_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)[
            : retrieval_setting.top_k
        ]

        records = []
        for doc, score in sorted_results:
            records.append(
                Record(
                    content=doc.page_content,
                    score=score,
                    title=doc.metadata.get("title", doc.page_content[:30]),
                    metadata=doc.metadata,
                )
            )

        return RetrievalResponse(records=records)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": 500,
                "error_msg": "發生內部服務器錯誤。請重試你的請求。",
            },
        )


# 添加全局異常處理
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict) and "error_code" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error_code": 500, "error_msg": str(exc.detail)},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error_code": 500,
            "error_msg": "發生內部服務器錯誤。請重試你的請求。",
        },
    )
