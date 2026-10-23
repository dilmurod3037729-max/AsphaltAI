"""autoloop uchun yengil GPU o'lchagich (torch import qilmaydi)."""
import subprocess


def gpu_state():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        u, used, tot = [float(x) for x in out.split(",")]
        return u, used / tot * 100
    except Exception:
        return None, None
