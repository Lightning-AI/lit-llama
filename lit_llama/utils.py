"""Utility functions for training and inference."""

import functools
import pickle
import warnings
from io import BytesIO
from pathlib import Path
from contextlib import contextmanager

import torch
import torch.utils._device
from lightning.fabric.strategies import DeepSpeedStrategy, FSDPStrategy
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.serialization import normalize_storage_type

llama_model_sizes = {
    4096: "7B",  # 7B n_embd=4096
    5120: "13B",  # 13B n_embd=5120
    6656: "30B",  # 30B n_embd=6656
    8192: "65B",  # 65B n_embd=8192
}


def llama_model_lookup(checkpoint: dict) -> str:
    """Returns the LLaMA model name from the checkpoint.
    
    Checks the width of the lm_head.weight matrix, as these uniquely identify the model.
    """
    embedding_size = checkpoint['transformer.wte.weight'].shape[1]
    return llama_model_sizes[embedding_size]


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


def save_model_checkpoint(fabric, model, file_path):
    """Handles boilerplate logic for retrieving and saving the state_dict.
    
    This will be upstreamed to Fabric soon.
    """
    file_path = Path(file_path)

    if isinstance(fabric.strategy, DeepSpeedStrategy):
        from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

        fabric.save(file_path, {"model": model})
        fabric.barrier()
        if fabric.global_rank == 0:
            # Create a consolidated checkpoint with the same name next to the deepspeed checkpoint
            convert_zero_checkpoint_to_fp32_state_dict(file_path, file_path.with_suffix(".pth"))
        return

    if isinstance(fabric.strategy, FSDPStrategy):
        save_policy = FullStateDictConfig(offload_to_cpu=(fabric.world_size > 1), rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            state_dict = model._forward_module.state_dict()
    else:
        state_dict = model.state_dict()

    if fabric.global_rank == 0:
        torch.save(state_dict, file_path)
    fabric.barrier()


class EmptyInitOnDevice(torch.overrides.TorchFunctionMode):
    def __init__(self, device=None, dtype=None, quantization_mode=None):
        """
        Create tensors with given device and dtype and don't run initialization
           (but instead use "empty tensors", i.e. uninitialized memory).

            device: `torch.device` to work with
            dtype: `torch.dtype` to work with
            quantization_mode: optional string, quantization mode to work with, default `None`.
                 Available modes: `llm.int8` bitsnbytes LLM.int8 quantization (only on GPU)
                                  `gptq.int4`, `gptq.int8`: GPTQ pre-quantized models

        Example::
            with EmptyInitOnDevice("cuda", dtype=torch.bfloat16):
               model = LLaMA.from_name('7B')
            model.load_state_dict(torch.load('llama-lit/7B/lit-llama.pth'))"""

        self.quantization_mode = quantization_mode
        self.quantized_linear_cls = None
        if self.quantization_mode == 'llm.int8':
            if device.type != "cuda":
                raise ValueError("Quantization is only supported on the GPU.")
            from .quantization import Linear8bitLt
            self.quantized_linear_cls = Linear8bitLt
        elif self.quantization_mode == 'gptq.int4':
            from .quantization import ColBlockQuantizedLinear
            self.quantized_linear_cls = functools.partial(ColBlockQuantizedLinear, bits=4, tile_cols=-1)
        elif self.quantization_mode == 'gptq.int8':
            from .quantization import ColBlockQuantizedLinear
            self.quantized_linear_cls = functools.partial(ColBlockQuantizedLinear, bits=8, tile_cols=-1)
        elif self.quantization_mode is not None:
            raise RuntimeError(f"unknown quantization mode {self.quantization_mode}")
        self.device = device
        self.dtype = dtype

    def __enter__(self):
        if self.quantized_linear_cls != None:
            self.torch_linear_cls = torch.nn.Linear
            torch.nn.Linear = self.quantized_linear_cls
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.quantized_linear_cls != None:
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


@contextmanager
def quantization(mode: str = None):
    quantized_linear_cls = None
    if mode == 'llm.int8':
        from .quantization import Linear8bitLt
        quantized_linear_cls = Linear8bitLt
    elif mode == 'gptq.int4':
        from .quantization import ColBlockQuantizedLinear
        quantized_linear_cls = functools.partial(ColBlockQuantizedLinear, bits=4, tile_cols=-1)
    elif mode == 'gptq.int8':
        from .quantization import ColBlockQuantizedLinear
        quantized_linear_cls = functools.partial(ColBlockQuantizedLinear, bits=8, tile_cols=-1)
    elif mode is not None:
        raise ValueError(f"Unknown quantization mode: {mode}")

    enabled = mode is not None
    torch_linear_cls = torch.nn.Linear
    if enabled:
        torch.nn.Linear = quantized_linear_cls
    yield
    if enabled:
        torch.nn.Linear = torch_linear_cls


# this is taken from torchhacks https://github.com/lernapparat/torchhacks


class NotYetLoadedTensor:
    def __init__(self, metatensor, archiveinfo, storageinfo, rebuild_args):
        self.metatensor = metatensor
        self.archiveinfo = archiveinfo
        self.storageinfo = storageinfo
        self.rebuild_args = rebuild_args

    @classmethod
    def rebuild_from_type_v2(cls, func, new_type, args, state, *, archiveinfo=None):
        ret = func(*args)
        if isinstance(ret, NotYetLoadedTensor):
            old_lt = ret._load_tensor

            def _load_tensor():
                t = old_lt()
                return torch._tensor._rebuild_from_type_v2(
                    lambda: t, new_type, (), state
                )

            ret._load_tensor = _load_tensor
            return ret
        return torch._tensor._rebuild_from_type_v2(func, new_type, args, state)

    @classmethod
    def rebuild_parameter(
        cls, data, requires_grad, backward_hooks, *, archiveinfo=None
    ):
        if isinstance(data, NotYetLoadedTensor):
            old_lt = data._load_tensor

            def _load_tensor():
                t = old_lt()
                return torch._utils._rebuild_parameter(t, requires_grad, backward_hooks)

            data._load_tensor = _load_tensor
            return data
        return torch._utils._rebuild_parameter(data, requires_grad, backward_hooks)

    @classmethod
    def rebuild_tensor_v2(
        cls,
        storage,
        storage_offset,
        size,
        stride,
        requires_grad,
        backward_hooks,
        metadata=None,
        *,
        archiveinfo=None,
    ):
        rebuild_args = (
            storage_offset,
            size,
            stride,
            requires_grad,
            backward_hooks,
            metadata,
        )
        metatensor = torch._utils._rebuild_tensor_v2(
            storage,
            storage_offset,
            size,
            stride,
            requires_grad,
            backward_hooks,
            metadata,
        )
        storageinfo = storage.archiveinfo
        return NotYetLoadedTensor(metatensor, archiveinfo, storageinfo, rebuild_args)

    def _load_tensor(self):
        name, storage_cls, fn, device, size = self.storageinfo
        dtype = self.metatensor.dtype

        uts = (
            self.archiveinfo.zipfile_context.zf.get_storage_from_record(
                f"data/{fn}",
                size * torch._utils._element_size(dtype),
                torch.UntypedStorage,
            )
            ._typed_storage()
            ._untyped_storage
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            storage = torch.storage.TypedStorage(
                wrap_storage=uts, dtype=self.metatensor.dtype, _internal=True
            )
        tensor = torch._utils._rebuild_tensor_v2(storage, *self.rebuild_args)
        return tensor

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        loaded_args = [
            (a._load_tensor() if isinstance(a, NotYetLoadedTensor) else a) for a in args
        ]
        res = func(*loaded_args, **kwargs)
        # gc.collect would be costly here, maybe do it optionally
        return res

    def __getattr__(self, name):
        # properties
        ## TODO: device, is_...??
        ## TODO: mH, mT, H, T, data, imag, real
        ## name ???
        if name in {
            "dtype",
            "grad",
            "grad_fn",
            "layout",
            "names",
            "ndim",
            "output_nr",
            "requires_grad",
            "retains_grad",
            "shape",
            "volatile",
        }:
            return getattr(self.metatensor, name)
        if name in {"size"}:
            return getattr(self.metatensor, name)
        # materializing with contiguous is needed for quantization
        if name in {"contiguous"}:
            return getattr(self._load_tensor(), name)

        raise AttributeError(f"{type(self)} does not have {name}")

    def __repr__(self):
        return f"NotYetLoadedTensor({repr(self.metatensor)})"


class LazyLoadingUnpickler(pickle.Unpickler):
    def __init__(self, file, zipfile_context):
        super().__init__(file)
        self.zipfile_context = zipfile_context

    def find_class(self, module, name):
        res = super().find_class(module, name)
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return functools.partial(
                NotYetLoadedTensor.rebuild_tensor_v2, archiveinfo=self
            )
        elif module == "torch._tensor" and name == "_rebuild_from_type_v2":
            return functools.partial(
                NotYetLoadedTensor.rebuild_from_type_v2, archiveinfo=self
            )
        elif module == "torch._utils" and name == "_rebuild_parameter":
            return functools.partial(
                NotYetLoadedTensor.rebuild_parameter, archiveinfo=self
            )
        return res

    def persistent_load(self, pid):
        name, cls, fn, device, size = pid
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = torch.storage.TypedStorage(dtype=cls().dtype, device="meta")
        s.archiveinfo = pid
        return s


class lazy_load:
    def __init__(self, fn):
        self.zf = torch._C.PyTorchFileReader(str(fn))
        with BytesIO(self.zf.get_record("data.pkl")) as pkl:
            mup = LazyLoadingUnpickler(pkl, self)
            self.sd = mup.load()

    def __enter__(self):
        return self.sd

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.zf  # I don't think there is a way to force closing...
        self.zf = None


class SavingProxyForStorage:
    def __init__(self, obj, saver, protocol_version=5):
        self.protocol_version = protocol_version
        self.saver = saver
        if not (isinstance(obj, torch.storage.TypedStorage) or torch.is_storage(obj)):
            raise TypeError(f"expected storage, not {type(obj)}")

        # this logic is taken from PyTorch 2.0+ torch/serialization.py
        if isinstance(obj, torch.storage.TypedStorage):
            # PT upstream wants to deprecate this eventually...
            storage = obj._untyped_storage
            storage_type_str = obj._pickle_storage_type()
            storage_type = getattr(torch, storage_type_str)
            storage_numel = obj._size()
        else:
            storage = obj
            storage_type = normalize_storage_type(type(obj))
            storage_numel = storage.nbytes()

        storage_key = saver._write_storage_and_return_key(storage)
        location = torch.serialization.location_tag(storage)

        self.storage_info = (
            "storage",
            storage_type,
            storage_key,
            location,
            storage_numel,
        )

    def __reduce_ex__(self, protocol_version):
        assert False, "this should be handled with out of band"


class SavingProxyForTensor:
    def __init__(self, tensor, saver, protocol_version=5):
        self.protocol_version = protocol_version
        self.reduce_ret_fn, (storage, *other_reduce_args) = tensor.__reduce_ex__(
            protocol_version
        )
        assert isinstance(
            storage, torch.storage.TypedStorage
        ), "Please check for updates"
        storage_proxy = SavingProxyForStorage(
            storage, saver, protocol_version=protocol_version
        )
        self.reduce_args = (storage_proxy, *other_reduce_args)

    def __reduce_ex__(self, protocol_version):
        if protocol_version != self.protocol_version:
            raise RuntimeError(
                f"Unexpected protocol version: expected {self.protocol_version}, got {protocol_version}"
            )
        return self.reduce_ret_fn, self.reduce_args


class IncrementalPyTorchPickler(pickle.Pickler):
    def __init__(self, saver, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.storage_dtypes = {}
        self.saver = saver
        self.id_map = {}

    # this logic is taken from PyTorch 2.0+ torch/serialization.py
    def persistent_id(self, obj):
        # FIXME: the docs say that persistent_id should only return a string
        # but torch store returns tuples. This works only in the binary protocol
        # see
        # https://docs.python.org/2/library/pickle.html#pickling-and-unpickling-external-objects
        # https://github.com/python/cpython/blob/master/Lib/pickle.py#L527-L537
        if isinstance(obj, SavingProxyForStorage):
            return obj.storage_info

        if isinstance(obj, torch.storage.TypedStorage) or torch.is_storage(obj):
            if isinstance(obj, torch.storage.TypedStorage):
                # TODO: Once we decide to break serialization FC, this case
                # can be deleted
                storage = obj._untyped_storage
                storage_dtype = obj.dtype
                storage_type_str = obj._pickle_storage_type()
                storage_type = getattr(torch, storage_type_str)
                storage_numel = obj._size()

            else:
                storage = obj
                storage_dtype = torch.uint8
                storage_type = normalize_storage_type(type(obj))
                storage_numel = storage.nbytes()

            # If storage is allocated, ensure that any other saved storages
            # pointing to the same data all have the same dtype. If storage is
            # not allocated, don't perform this check
            if storage.data_ptr() != 0:
                if storage.data_ptr() in self.storage_dtypes:
                    if storage_dtype != self.storage_dtypes[storage.data_ptr()]:
                        raise RuntimeError(
                            "Cannot save multiple tensors or storages that "
                            "view the same data as different types"
                        )
                else:
                    self.storage_dtypes[storage.data_ptr()] = storage_dtype

            storage_key = self.id_map.get(storage._cdata)
            if storage_key is None:
                storage_key = self.saver._write_storage_and_return_key(storage)
                self.id_map[storage._cdata] = storage_key
            location = torch.serialization.location_tag(storage)

            return ("storage", storage_type, storage_key, location, storage_numel)

        return None


class incremental_save:
    def __init__(self, name):
        self.name = name
        self.zipfile = torch._C.PyTorchFileWriter(str(name))
        self.has_saved = False
        self.next_key = 0

    def __enter__(self):
        return self

    def store_early(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return SavingProxyForTensor(tensor, self)
        raise TypeError(f"can only store tensors early, not {type(tensor)}")

    def save(self, obj):
        if self.has_saved:
            raise RuntimeError("have already saved")
        # Write the pickle data for `obj`
        data_buf = BytesIO()
        pickler = IncrementalPyTorchPickler(self, data_buf, protocol=5)
        pickler.dump(obj)
        data_value = data_buf.getvalue()
        self.zipfile.write_record("data.pkl", data_value, len(data_value))
        self.has_saved = True

    def _write_storage_and_return_key(self, storage):
        if self.has_saved:
            raise RuntimeError("have already saved")
        key = self.next_key
        self.next_key += 1
        name = f"data/{key}"
        if storage.device.type != "cpu":
            storage = storage.cpu()
        num_bytes = storage.nbytes()
        self.zipfile.write_record(name, storage.data_ptr(), num_bytes)
        return key

    def __exit__(self, type, value, traceback):
        self.zipfile.write_end_of_file()
