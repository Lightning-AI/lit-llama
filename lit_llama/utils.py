import torch.utils._device


class EmptyInitOnDevice(torch.overrides.TorchFunctionMode):
    def __init__(self, device=None, dtype=None, quantization_mode=None):
        """
        Create tensors with given device and dtype and don't run initialization
           (but instead use "empty tensors", i.e. uninitialized memory).

            device: `torch.device` to work with
            dtype: `torch.dtype` to work with
            quantization_mode: optional string, quantization mode to work with, default `None`.
                 Available modes: `llm.int8` bitsnbytes LLM.int8 quantization (only on GPU)

        Example::
            with EmptyInitOnDevice("cuda", dtype=torch.bfloat16):
               model = LLaMA.from_name('7B')
            model.load_state_dict(torch.load('llama-lit/7B/state_dict.pth'))"""

        self.quantization_mode = quantization_mode
        if self.quantization_mode == 'llm.int8':
            if device.type != "cuda":
                raise ValueError("Quantization is only supported on the GPU.")
            from .quantization import Linear8bitLt
            self.Linear8bitLt = Linear8bitLt
        self.device = device
        self.dtype = dtype

    def __enter__(self):
        if self.quantization_mode == 'llm.int8':
            self.torch_linear_cls = torch.nn.Linear
            torch.nn.Linear = self.Linear8bitLt
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.quantization_mode == 'llm.int8':
            torch.nn.Linear = self.torch_linear_cls
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if getattr(func, "__module__", None) == "torch.nn.init":
            if "tensor" in kwargs:
                return kwargs["tensor"]
            else:
                return args[0]
        if (
            self.device is not None
            and func in torch.utils._device._device_constructors()
            and kwargs.get("device") is None
        ):
            kwargs["device"] = self.device
        if (
            self.dtype is not None
            and func in torch.utils._device._device_constructors()
            and kwargs.get("dtype") is None
        ):
            kwargs["dtype"] = self.dtype
        return func(*args, **kwargs)
