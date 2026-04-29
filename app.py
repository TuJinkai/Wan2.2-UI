import os
import sys
import subprocess
from datetime import datetime
import gradio as gr

# ===== 配置区 =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WAN22_REPO = os.path.join(BASE_DIR, "Wan2.2-main")
I2V_CKPT_DIR = os.path.join(BASE_DIR, "Wan2.2-I2V-A14B")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
GENERATE_SCRIPT = os.path.join(WAN22_REPO, "generate.py")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELSCOPE_MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B"

DEFAULT_I2V_PROMPT = (
    "A young woman smiling in the snow, wearing a white coat and pink blush, "
    "with visible dimples and sparkling eyes, spinning joyfully in a snowy landscape of Xinjiang."
)


# ===== 模型自动下载（启动时调用） =====
def ensure_model():
    """如果模型权重不存在，自动从 ModelScope 下载"""
    if os.path.exists(I2V_CKPT_DIR) and os.listdir(I2V_CKPT_DIR):
        print(f"[INFO] 模型权重已存在：{I2V_CKPT_DIR}")
        return True

    print(f"[INFO] 模型权重未找到，开始从 ModelScope 下载...")
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("[ERROR] 请先安装 modelscope：pip install modelscope")
        return False

    try:
        snapshot_download(
            MODELSCOPE_MODEL_ID,
            cache_dir=BASE_DIR,
            local_dir=I2V_CKPT_DIR,
        )
        print(f"[INFO] 模型权重下载完成：{I2V_CKPT_DIR}")
        return True
    except Exception as e:
        print(f"[ERROR] 模型下载失败：{str(e)}")
        return False


# ===== 日志读取 =====
def read_last_lines(log_file, n=20):
    if not log_file or not os.path.exists(log_file):
        return "⏳ 等待日志生成..."
    try:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            return "".join(lines[-n:])
    except Exception as e:
        return f"读取日志失败：{str(e)}"


def refresh_status(log_file, save_file):
    log_content = read_last_lines(log_file)
    video = save_file if save_file and os.path.exists(save_file) else None
    return log_content, video


# ===== I2V 生成 =====
def generate_i2v(
    image,
    prompt,
    frame_num=17,
    sample_steps=15,
    sample_solver="unipc",
    cuda_device=0,
):
    if not os.path.exists(GENERATE_SCRIPT):
        return f"❌ 未找到 generate.py\n请确保 Wan2.2 仓库已克隆到：{WAN22_REPO}", "", "", ""

    if image is None:
        return "❌ 请上传输入图像！", "", "", ""
    if not prompt.strip():
        return "❌ 提示词不能为空！", "", "", ""
    if not os.path.exists(I2V_CKPT_DIR):
        return f"❌ 模型权重目录不存在：{I2V_CKPT_DIR}\n请先在「模型下载」标签页下载模型。", "", "", ""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_file = os.path.join(OUTPUT_DIR, f"i2v_{timestamp}.mp4")
    log_file = os.path.join(OUTPUT_DIR, f"i2v_{timestamp}.log")

    cmd = [
        sys.executable, GENERATE_SCRIPT,
        "--task", "i2v-A14B",
        "--size", "832*480",
        "--ckpt_dir", I2V_CKPT_DIR,
        "--offload_model", "False",
        "--frame_num", str(frame_num),
        "--convert_model_dtype",
        "--prompt", prompt,
        "--image", image,
        "--save_file", save_file,
        "--sample_steps", str(sample_steps),
        "--sample_solver", sample_solver,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

    try:
        with open(log_file, "w") as logfile:
            subprocess.Popen(cmd, cwd=WAN22_REPO, env=env, stdout=logfile, stderr=subprocess.STDOUT)
        status_msg = f"✅ I2V 视频生成已启动！\nCUDA 设备：{cuda_device}\n保存路径：{save_file}\n日志文件：{log_file}"
        return status_msg, log_file, "", save_file
    except Exception as e:
        return f"❌ 启动失败：{str(e)}", "", "", ""


def check_environment():
    """检查运行环境"""
    msgs = []

    # 检查 generate.py
    if os.path.exists(GENERATE_SCRIPT):
        msgs.append(f"✅ generate.py 已就绪：{GENERATE_SCRIPT}")
    else:
        msgs.append(f"⚠️ generate.py 未找到，请将 Wan2.2 仓库克隆到 {WAN22_REPO}")

    # 检查模型权重
    if os.path.exists(I2V_CKPT_DIR) and os.listdir(I2V_CKPT_DIR):
        msgs.append(f"✅ 模型权重已就绪：{I2V_CKPT_DIR}")
    else:
        msgs.append(f"⚠️ 模型权重未下载，请先在「模型下载」标签页下载")

    # 检查 CUDA
    try:
        import torch
        if torch.cuda.is_available():
            msgs.append(f"✅ CUDA 可用，GPU 数量：{torch.cuda.device_count()}")
        else:
            msgs.append("⚠️ CUDA 不可用，将使用 CPU（非常慢）")
    except ImportError:
        msgs.append("⚠️ 未安装 PyTorch，无法检测 CUDA 状态")

    return "\n".join(msgs)


# ===== Gradio UI =====
with gr.Blocks(title="Wan2.2 I2V 图生视频 WebUI") as demo:
    gr.Markdown("# 🎥 Wan2.2 I2V 图生视频 WebUI")
    gr.Markdown("基于 Wan2.2-I2V-A14B 模型，输入一张图像和提示词，生成动态视频。")

    with gr.Tabs():
        # === Tab 0: 环境检查 ===
        with gr.Tab("🔧 环境检查"):
            gr.Markdown("### 启动前请确认以下项均已就绪")
            env_output = gr.Textbox(label="环境状态", value=check_environment(), interactive=False, lines=6)
            env_refresh = gr.Button("🔄 刷新环境检查")
            env_refresh.click(fn=check_environment, outputs=[env_output])

        # === Tab 1: 图生视频 (I2V) ===
        with gr.Tab("🖼️ 图生视频 (I2V)"):
            gr.Markdown("### 上传图像 + 提示词 → 生成 480P 动态视频")

            log_file_state = gr.State("")
            video_file_state = gr.State("")

            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(type="filepath", label="上传输入图像（建议比例 832:480）")
                    prompt_i2v = gr.Textbox(
                        label="提示词（Prompt）",
                        placeholder="描述你希望图像如何动起来...",
                        lines=5,
                        max_lines=12,
                        value=DEFAULT_I2V_PROMPT,
                    )
                    frame_num_i2v = gr.Slider(
                        5, 65, value=17, step=4,
                        label="帧数（必须为 4n+1；17≈1秒@16fps，65≈4秒@16fps）",
                    )
                    sample_steps_i2v = gr.Slider(
                        5, 50, value=15, step=1,
                        label="采样步数（越低越快，越高越精细）",
                    )
                    sample_solver_i2v = gr.Dropdown(
                        choices=["unipc", "dpm++", "ddim", "euler"],
                        value="unipc",
                        label="采样器",
                    )
                    cuda_device_i2v = gr.Number(value=0, precision=0, label="CUDA 设备 ID")
                    submit_i2v = gr.Button("🚀 生成 I2V 视频", variant="primary")
                    status_i2v = gr.Textbox(label="状态", interactive=False, lines=4)

                with gr.Column():
                    log_display = gr.Textbox(label="实时日志（最后 20 行）", interactive=False, lines=10, max_lines=20)
                    gr.Markdown("### 📽️ 生成结果预览")
                    video_preview = gr.Video(label="I2V 视频结果", interactive=False)

            def on_submit(*args):
                msg, log_path, _, save_path = generate_i2v(*args)
                return msg, log_path, save_path

            submit_i2v.click(
                fn=on_submit,
                inputs=[input_image, prompt_i2v, frame_num_i2v, sample_steps_i2v, sample_solver_i2v, cuda_device_i2v],
                outputs=[status_i2v, log_file_state, video_file_state],
            )

            refresh_timer = gr.Timer(3)
            refresh_timer.tick(
                fn=refresh_status,
                inputs=[log_file_state, video_file_state],
                outputs=[log_display, video_preview],
            )


if __name__ == "__main__":
    ensure_model()
    demo.launch(
        server_name="0.0.0.0",
        server_port=17861,
        share=False,
        root_path="/wan22",
    )
