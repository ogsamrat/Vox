#!/usr/bin/env python3
import uvicorn
from src.utils.logger import setup_logger

if __name__ == "__main__":
    setup_logger()
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
