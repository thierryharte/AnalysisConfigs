from HH4b_parton_matching_config_DHH_reweight import onnx_model_dict

from utils.onnx_executor_common import OnnxExecutorFactory


def get_executor_factory(executor_name, **kwargs):
    return OnnxExecutorFactory(onnx_model_dict=onnx_model_dict, **kwargs)
