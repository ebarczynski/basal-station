from basal_station.catalog import (
    format_size_bytes,
    get_model,
    list_models,
)


def test_omni_max_repos():
    m = get_model("omni-1.0-max")
    assert m is not None
    assert m.media is True
    assert m.repos == {
        "safetensors": "Remek/basal-omni-1.0-max",
        "fp8": "Remek/basal-omni-1.0-max-FP8",
        "gguf": "Remek/basal-omni-1.0-max-GGUF",
        "mlx": "Remek/basal-omni-1.0-max-MLX",
    }


def test_all_entries_have_key_and_label():
    keys = set()
    for m in list_models():
        assert m.key not in keys, "duplicate catalog key"
        keys.add(m.key)
        assert m.label and m.repos


def test_format_sizes():
    m = get_model("omni-1.0-max")
    gguf_q4 = format_size_bytes(m, "gguf", "Q4_K_M")
    gguf_q8 = format_size_bytes(m, "gguf", "Q8_0")
    st = format_size_bytes(m, "safetensors")
    assert gguf_q4 < gguf_q8 < st
    # catalog sizes are decimal GB (matching HF file listings)
    assert abs(gguf_q4 / 1e9 - 7.4) < 0.3
    assert abs(st / 1e9 - 24.6) < 1.0


def test_unknown_params_give_no_size():
    m = get_model("omni-1.0-ultra")
    assert m.params_b is None
    assert format_size_bytes(m, "gguf", "Q4_K_M") is None
