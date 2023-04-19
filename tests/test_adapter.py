import torch
from dataclasses import asdict


def test_config_identical(lit_llama):
    import lit_llama.adapter as llama_adapter
    import lit_llama.model as llama
    from lit_llama.utils import EmptyInitOnDevice

    llama_config = asdict(llama.LLaMAConfig())
    adapter_config = asdict(llama_adapter.LLaMAConfig())

    del adapter_config["adapter_prompt_length"]
    del adapter_config["adapter_start_layer"]
    assert adapter_config == llama_config

    with EmptyInitOnDevice(device=torch.device("cpu")):
        llama_model = llama.LLaMA.from_name("7B")
        adapter_model = llama_adapter.LLaMA.from_name("7B")

        assert llama_model.lm_head.weight.shape == adapter_model.lm_head.weight.shape
