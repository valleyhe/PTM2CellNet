"""
FastAPI应用模块
功能概述: 创建和配置FastAPI应用
设计思路: 提供应用工厂函数，支持CORS配置
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


def create_app(
    title: str = "PTM2CellNet API",
    description: str = "蛋白质PTM与细胞状态预测API",
    version: str = "1.0.0",
    cors_origins: Optional[list[str]] = None,
) -> FastAPI:
    """
    创建FastAPI应用

    参数:
        title: 应用标题
        description: 应用描述
        version: 应用版本
        cors_origins: 允许的CORS源列表

    返回:
        配置好的FastAPI应用
    """
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        logger.info("PTM2CellNet API 启动")
        yield
        logger.info("PTM2CellNet API 关闭")

    app = FastAPI(
        title=title,
        description=description,
        version=version,
        lifespan=lifespan,
    )

    if cors_origins is None:
        env_origins = os.environ.get("PTM2CELLNET_CORS_ORIGINS", "")
        if env_origins:
            cors_origins = [origin.strip() for origin in env_origins.split(",") if origin.strip()]
        else:
            cors_origins = []

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1")

    @app.get("/")
    async def root():
        """根路径"""
        return {
            "message": "Welcome to PTM2CellNet API",
            "version": version,
            "docs": "/docs",
        }

    logger.info("FastAPI应用已创建")
    return app
