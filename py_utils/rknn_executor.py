import sys


class RKNN_model_container():
    """
    兼容 rknn-toolkit-lite2 与 rknn-toolkit 的执行容器。

    - 优先使用 rknn-toolkit-lite2 提供的 rknnlite.api.RKNNLite（设备端推理，需 rknn_server）。
    - 如未安装 lite2，则回退到 rknn-toolkit 的 rknn.api.RKNN（PC 端/转换工具包）。

    注意：lite2 的 RKNNLite.init_runtime 不接受 target/device_id 参数，直接本地连接 rknn_server。
    """

    def __init__(self, model_path, target=None, device_id=None) -> None:
        self.backend = None  # 'lite' | 'full'
        self.rknn = None

        # 1) 优先尝试 rknn-toolkit-lite2
        try:
            from rknnlite.api import RKNNLite  # type: ignore
            self.backend = 'lite'
            rknn = RKNNLite()
        except Exception as e_lite:
            # 2) 回退到 rknn-toolkit
            try:
                from rknn.api import RKNN  # type: ignore
                self.backend = 'full'
                rknn = RKNN()
            except Exception as e_full:
                # 均不可用，给出清晰错误
                msg = (
                    '未找到 rknnlite.api 或 rknn.api 模块。\n'
                    '- 您的设备端推理建议安装: rknn-toolkit-lite2 (并确保使用 Python 3.11/aarch64 对应的 whl)。\n'
                    '- 如果需要使用 rknn-toolkit，请确保已正确安装 rknn==2.x，并注意其与 lite2 的差异。\n'
                    f'lite 导入错误: {e_lite}\nfull 导入错误: {e_full}'
                )
                raise ImportError(msg)

        # 载入模型
        ret = rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f'加载 RKNN 模型失败: {model_path}, load_rknn 返回码: {ret}')

        # 初始化 runtime
        print('--> Init runtime environment')
        if self.backend == 'lite':
            # lite 模式：连接本地 rknn_server，不支持 target/device_id
            ret = rknn.init_runtime()
        else:
            # full 模式：保持原逻辑
            if target is None:
                ret = rknn.init_runtime()
            else:
                ret = rknn.init_runtime(target=target, device_id=device_id)

        if ret != 0:
            print('Init runtime environment failed')
            raise RuntimeError(f'init_runtime 失败, 返回码: {ret}, backend: {self.backend}')
        print('done')

        self.rknn = rknn

    # def __del__(self):
    #     self.release()

    def run(self, inputs):
        if self.rknn is None:
            print("ERROR: rknn has been released")
            return []

        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        result = self.rknn.inference(inputs=inputs)
        return result

    def release(self):
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None