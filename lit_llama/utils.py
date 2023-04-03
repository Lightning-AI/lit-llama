import torch.utils._device


class EmptyInitOnDevice(torch.overrides.TorchFunctionMode):
    def __init__(self, device=None, dtype=None):
        """
        Create tensors with given device and dtype and don't run initialization
           (but instead use "empty tensors", i.e. uninitialized memory).

        Example::
            with EmptyInitOnDevice("cuda", dtype=torch.bfloat16):
               model = LLaMA.from_name('7B')
            model.load_state_dict(torch.load('llama-lit/7B/state_dict.pth'))"""
        self.device = device
        self.dtype = dtype

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
