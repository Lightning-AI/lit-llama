from dataclasses import asdict
import pytest
import sys
import torch


@pytest.mark.skipif(sys.platform == "win32", reason="EmptyInitOnDevice on CPU not working for Windows.")
@pytest.mark.parametrize("model_size", ["7B", "13B", "30B", "65B"])
def test_config_identical(model_size, lit_llama):
    import lit_llama.adapter as llama_adapter
    import lit_llama.model as llama
    from lit_llama.utils import EmptyInitOnDevice

    llama_config = asdict(llama.LLaMAConfig.from_name(model_size))
    adapter_config = asdict(llama_adapter.LLaMAConfig.from_name(model_size))

    del adapter_config["adapter_prompt_length"]
    del adapter_config["adapter_start_layer"]
    assert adapter_config == llama_config

    with EmptyInitOnDevice():
        llama_model = llama.LLaMA.from_name(model_size)
        adapter_model = llama_adapter.LLaMA.from_name(model_size)
        assert llama_model.lm_head.weight.shape == adapter_model.lm_head.weight.shape


def test_adapter_load_gating_factor():
    """Tests backward-compatible loading of checkpoints after the `gating_factor` was extended per-head
    in PR #297.
    """
    import lit_llama.adapter as llama_adapter
    from lit_llama.utils import lazy_load

    config = llama_adapter.LLaMAConfig(n_head=4, block_size=100, n_embd=16)
    attn = llama_adapter.CausalSelfAttention(config=config, block_idx=3)

    state_dict={
        "gating_factor": torch.tensor(0.42),  # in old checkpoints, this was a scalar
        "c_attn.weight": torch.zeros(3 * 16, 16),
        "c_proj.weight": torch.zeros(16, 16),
        "adapter_wte.weight": torch.zeros(10, 16),
    }

    attn.load_state_dict(state_dict=state_dict)
    assert torch.equal(attn.gating_factor, torch.full((1, 4, 1, 1), 0.42))
