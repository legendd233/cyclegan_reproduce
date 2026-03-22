FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple/

# 复制代码和模型权重
COPY model.py demo.py ./
COPY weights/latest_net_G_B.pth /app/weights/latest_net_G_B.pth
COPY images/ /app/images/

ENV MODEL_PATH=/app/weights/latest_net_G_B.pth

EXPOSE 7860

CMD ["python", "demo.py"]