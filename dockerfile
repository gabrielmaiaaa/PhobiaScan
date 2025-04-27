FROM alpine:3.20
WORKDIR /app
COPY requirements.txt .
RUN apk add --no-cache python3.11.9
RUN pip3 install --no-cache-dir -r requirements.txt
COPY . .
CMD [ "python3", "-m", "src.cam.yolo_cam" ]