from rknn.api import RKNN


class RKNN_model_container():
    def __init__(self, model_path, target=None, device_id=None, core_mask=None) -> None:
        rknn = RKNN()

        # Direct Load RKNN Model
        rknn.load_rknn(model_path)

        print('--> Init runtime environment')
        if target is None:
            ret = rknn.init_runtime()
        else:
            # core_mask: RKNN.NPU_CORE_0, RKNN.NPU_CORE_1, RKNN.NPU_CORE_2
            ret = rknn.init_runtime(target=target, device_id=device_id, core_mask=core_mask)
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)
        print('done')

        self.rknn = rknn

    # def __del__(self):
    #     self.release()

    def run(self, inputs):
        if self.rknn is None:
            print("ERROR: rknn has been released")
            return []

        if isinstance(inputs, list) or isinstance(inputs, tuple):
            pass
        else:
            inputs = [inputs]

        result = self.rknn.inference(inputs=inputs)
    
        return result

    def release(self):
        self.rknn.release()
        self.rknn = None