# manage_module_memory.py
import multiprocessing as mp
import subprocess
import sys
import os


def _worker_target(func_path, args, kwargs):
    """子进程工作函数（必须是模块级别，供 mp.Process 使用）"""
    import importlib
    import torch

    # spawn 子进程不继承父进程 sys.path；在 .pth 文件未生效的情况下
    # 手动把项目根目录（本文件的两级父目录）插入 sys.path
    import os as _os
    _project_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    for _p in [_project_root, _os.getcwd()]:
        if _p not in sys.path:
            sys.path.insert(0, _p)

    try:
        module_path, func_name = func_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        func = getattr(module, func_name)
        func(*args, **kwargs)
    except Exception as e:
        print(f"子进程执行失败: {e}")
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def _conda_python(env_name):
    """根据 conda 环境名找到对应的 Python 可执行路径"""
    # 优先从 CONDA_PREFIX 推断 conda 根目录
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    if conda_prefix:
        # CONDA_PREFIX 形如 /path/to/anaconda3/envs/Agent
        # 取两级父目录得到 anaconda3 根
        conda_root = os.path.dirname(os.path.dirname(conda_prefix))
    else:
        # 回退：尝试常见路径
        for candidate in [
            os.path.expanduser('~/anaconda3'),
            os.path.expanduser('~/miniconda3'),
            '/opt/anaconda3',
            '/opt/miniconda3',
        ]:
            if os.path.isdir(candidate):
                conda_root = candidate
                break
        else:
            raise RuntimeError("找不到 conda 根目录，请设置 CONDA_PREFIX 环境变量")

    python_path = os.path.join(conda_root, 'envs', env_name, 'bin', 'python')
    if not os.path.isfile(python_path):
        raise RuntimeError(f"conda 环境 '{env_name}' 的 Python 不存在: {python_path}")
    return python_path


def _resolve_func_path(func):
    """
    把函数对象解析为可被 importlib 加载的 'module.funcname' 字符串。
    当函数定义在 __main__ 时（即直接运行的脚本），用文件路径推断真实模块名。
    """
    if not callable(func):
        return func  # 已经是字符串

    module = func.__module__
    if module == '__main__':
        import inspect
        src_file = inspect.getfile(func)
        # 将绝对路径转为相对于 cwd 的模块名，例如 agent.py → agent
        rel = os.path.relpath(src_file, os.getcwd())
        module = rel.replace(os.sep, '.').removesuffix('.py')
    return f"{module}.{func.__name__}"


def run_in_isolation(func, *args, conda_env=None, **kwargs):
    """
    在隔离子进程中运行函数，结束后彻底释放 GPU 显存。

    Args:
        func       : 函数对象或 'module.func' 字符串
        *args      : 传给 func 的位置参数
        conda_env  : conda 环境名（str）。指定时用该环境的 Python 以 subprocess
                     运行，适用于需要不同依赖的模块（如 whisperx）；
                     不指定时沿用当前解释器，走 mp.Process。
        **kwargs   : 传给 func 的关键字参数（conda_env 模式下不支持，会报错提示）
    """
    func_path = _resolve_func_path(func)

    if conda_env is None:
        # ── 原有逻辑：同解释器 spawn 子进程 ──────────────────────
        ctx = mp.get_context('spawn')
        p = ctx.Process(target=_worker_target, args=(func_path, args, kwargs))
        p.start()
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"子进程异常退出，exitcode={p.exitcode}")
    else:
        # ── 新逻辑：用目标 conda 环境的 Python 运行 ──────────────
        if kwargs:
            raise ValueError(
                "conda_env 模式下暂不支持 kwargs，请将关键字参数改为位置参数传入"
            )
        python_bin = _conda_python(conda_env)

        # 用 -c 内联脚本：把 func_path 和 args 序列化后传入
        import json
        # args 必须全部可 JSON 序列化（字符串、数字、None、列表）
        payload = json.dumps({'func_path': func_path, 'args': list(args)})

        script = (
            "import json, importlib, torch, sys\n"
            f"payload = json.loads({repr(payload)})\n"
            "module_path, func_name = payload['func_path'].rsplit('.', 1)\n"
            "sys.path.insert(0, '.')\n"
            "module = importlib.import_module(module_path)\n"
            "func = getattr(module, func_name)\n"
            "try:\n"
            "    func(*payload['args'])\n"
            "finally:\n"
            "    if torch.cuda.is_available():\n"
            "        torch.cuda.empty_cache()\n"
            "        torch.cuda.synchronize()\n"
        )

        env = os.environ.copy()
        # 让子进程的 CUDA_VISIBLE_DEVICES 与父进程一致
        result = subprocess.run(
            [python_bin, '-c', script],
            env=env,
            cwd=os.getcwd(),
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"conda 环境 '{conda_env}' 子进程异常退出，returncode={result.returncode}"
            )
