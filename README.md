# Wan2.2-UI

Wan2.2 I2V 图生视频 WebUI，基于 Gradio。

## 使用方式

### Docker 运行

```bash
docker run --gpus all -p 17861:17861 \
  -v /path/to/Wan2.2-I2V-A14B:/app/Wan2.2-I2V-A14B \
  -v /path/to/output:/app/output \
  registry.cn-hangzhou.aliyuncs.com/${ACR_NAMESPACE}/wan22-ui:latest
```

访问 `http://localhost:17861/wan22`

### 模型权重

模型权重需要单独下载并挂载到容器内：

```bash
# 从 ModelScope 下载
pip install modelscope
modelscope download --model Wan-AI/Wan2.2-I2V-A14B --local_dir ./Wan2.2-I2V-A14B
```
