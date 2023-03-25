import os

import yaml
import torch
from omegaconf import OmegaConf

from saicinpainting.training.trainers import load_checkpoint
from to_jit import JITWrapper


def load_model_torch():
    predict_config = OmegaConf.load('configs/prediction/default.yaml')

    train_config_path = os.path.join(predict_config.model.path, "config.yaml")
    with open(train_config_path, "r") as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = "noop"

    checkpoint_path = os.path.join(
        predict_config.model.path, "models", predict_config.model.checkpoint
    )
    model = load_checkpoint(
        train_config, checkpoint_path, strict=False, map_location="cpu"
    )
    model = JITWrapper(model)

    model.eval()

    image = torch.rand(1, 3, 120, 120)
    mask = torch.rand(1, 1, 120, 120)

    try:
        _ = model(image, mask)
        print('Successfully ran model')
    except Exception as e:
        print(f'Failed to run model: {e}')
        print(e)

    return model


def _get_uncovnertible_ops(model, opset_version: int):
    args = (torch.randn(1, 3, 256, 256), torch.randn(1, 1, 256, 256))

    torch_script_graph, unconvertible_ops = torch.onnx.utils.unconvertible_ops(
        model=model,
        args=args,
        opset_version=opset_version,
    )

    return set(unconvertible_ops)


def _convert_to_onnx(model, output_path: str, opset_version: int):
    args = (torch.randn(1, 3, 256, 256), torch.randn(1, 1, 256, 256))

    torch.onnx.export(
        model=model,
        args=args,
        f=output_path,
        opset_version=opset_version,
        do_constant_folding=True,
        verbose=True,
        input_names=['img', 'mask'],
        output_names=['output'],
        dynamic_axes={
            'img': {0: 'batch_size', 2: 'width', 3: 'height'},
            'mask': {0: 'batch_size', 2: 'width', 3: 'height'},
            'output': {0: 'batch_size', 2: 'width', 3: 'height'},
        }
    )


if __name__ == '__main__':
    output_onnx_path = 'lama.onnx'

    torch_model = load_model_torch()
    _opset_version = 17

    _unconvertible_ops = _get_uncovnertible_ops(
        model=torch_model,
        opset_version=_opset_version,
    )
    print(f'Unconvertible ops: {_unconvertible_ops}')

    _convert_to_onnx(
        model=torch_model,
        output_path=output_onnx_path,
        opset_version=_opset_version,
    )


