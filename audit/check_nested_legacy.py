"""Supplemental audit of six nested legacy llm_blocks (not official problem IDs).
Preserve historical sources byte-for-byte; record failing checks without calling
these unmodified historical modules repaired. This is separate from tests/.
"""
import argparse, ast, hashlib, importlib.util, json, pathlib, shutil, warnings
import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=pathlib.Path, default=pathlib.Path('.'))
    parser.add_argument('--original', type=pathlib.Path)
    args = parser.parse_args()
    root = args.root.resolve()
    archived = root / 'archive/pytorch_legacy/llm_blocks'
    if args.original:
        source = args.original.resolve() / 'pytorch/llm_blocks'
        for p in sorted(source.glob('*.py')):
            target = archived / p.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                assert target.read_bytes() == p.read_bytes(), target
            else:
                target.write_bytes(p.read_bytes())
            current = root / 'pytorch/llm_blocks' / p.name
            if current.exists():
                assert current.read_bytes() == p.read_bytes(), current
                current.unlink()
        current_dir = root / 'pytorch/llm_blocks'
        if current_dir.exists():
            cache = current_dir / '__pycache__'
            if cache.exists():
                shutil.rmtree(cache)
            if not list(current_dir.iterdir()):
                current_dir.rmdir()
    torch.set_num_threads(1)
    torch.manual_seed(20261015)
    modules, files, checks = {}, [], []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        for p in sorted(archived.glob('*.py')):
            compile(ast.parse(p.read_text()), str(p), 'exec')
            spec = importlib.util.spec_from_file_location('nested_' + p.stem.split('-')[0], p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            modules[int(p.stem.split('-')[0])] = mod
            files.append({'original_path': 'pytorch/llm_blocks/' + p.name,
                          'archive_path': str(p.relative_to(root)),
                          'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                          'syntax': 'pass', 'import': 'pass'})
    assert len(files) == 6

    def check(name, fn):
        try:
            details = fn()
            checks.append({'name': name, 'status': 'pass', 'details': details})
        except Exception as exc:
            checks.append({'name': name, 'status': 'fail',
                           'error': f'{type(exc).__name__}: {exc}'})

    def compare(a, b, strict=False):
        error = (a - b).abs().max().item()
        torch.testing.assert_close(a, b, rtol=1e-12 if strict else 2e-5,
                                   atol=1e-12 if strict else 2e-6)
        return {'max_abs_error': error}

    x = torch.randn(2, 4, 8)
    norm = modules[1].RMSNorm(8)
    check('RMSNorm float32 reference', lambda: compare(norm(x), F.rms_norm(x, (8,), norm.weight, 1e-6)))
    double_norm = modules[1].RMSNorm(8).double()
    xd = x.double() + 1e-9
    check('RMSNorm float64 precision preservation', lambda: compare(double_norm(xd), F.rms_norm(xd, (8,), double_norm.weight, 1e-6), True))

    rope = modules[2]
    freq = rope.precompute_freqs_cis(4, 4)
    q = torch.randn(2, 4, 2, 4)
    k = torch.randn(2, 4, 1, 4)
    def rotary_reference():
        a, b = rope.apply_rotary_emb(q, k, freq)
        def rotate(t):
            even, odd = t[..., 0::2], t[..., 1::2]
            c, s = freq.real[None, :, None], freq.imag[None, :, None]
            return torch.stack((even*c-odd*s, even*s+odd*c), -1).flatten(-2)
        compare(a, rotate(q)); return compare(b, rotate(k))
    check('Standalone RoPE adjacent-pair reference', rotary_reference)
    ffn = modules[3].SwiGLUFFN(8, 12)
    check('SwiGLU forward reference', lambda: compare(ffn(x), ffn.w2(F.silu(ffn.w1(x))*ffn.w3(x))))

    gqa = modules[4].GroupedQueryAttention(8, 2, 1)
    def attention_reference(model, values, frequencies=None):
        b, length, _ = values.shape
        q = model.wq(values).reshape(b, length, 2, 4)
        k = model.wk(values).reshape(b, length, 1, 4)
        v = model.wv(values).reshape(b, length, 1, 4)
        if frequencies is not None:
            q, k = rope.apply_rotary_emb(q, k, frequencies)
        q = q.transpose(1, 2)
        k, v = (t.transpose(1, 2).repeat_interleave(2, 1) for t in (k, v))
        context = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, length, 8)
        return model.wo(context)
    check('GQA without RoPE SDPA reference', lambda: compare(gqa(x), attention_reference(gqa, x)))
    check('GQA supplied RoPE is applied', lambda: compare(gqa(x, freq), attention_reference(gqa, x, freq)))
    block = modules[5].TransformerBlock(8, 12, 2, 1)
    def block_reference():
        h = x + attention_reference(block.attention, block.attention_norm(x), freq)
        return compare(block(x, freq), h + block.ffn(block.ffn_norm(h)))
    check('Transformer block applies supplied RoPE', block_reference)

    model = modules[6].LlamaModel(vocab_size=20, dim=8, hidden_dim=12,
                                  n_layers=1, n_heads=2, n_kv_heads=1, max_seq_len=16).eval()
    tokens = torch.tensor([[1, 2, 3, 4]])
    def llama_forward():
        output = model(tokens)
        assert output.shape == (1, 4, 20) and torch.isfinite(output).all()
        return {'shape': list(output.shape)}
    check('Llama float32 forward smoke test', llama_forward)
    def llama_causal():
        altered = tokens.clone(); altered[:, 2:] = torch.tensor([[8, 9]])
        return compare(model(tokens)[:, :2], model(altered)[:, :2])
    check('Llama causal future-token isolation', llama_causal)
    def llama_uses_frequencies():
        actual = model(tokens)
        saved = model.freqs_cis
        try:
            model.freqs_cis = torch.ones_like(saved)
            without_rotation = model(tokens)
        finally:
            model.freqs_cis = saved
        delta = (actual - without_rotation).abs().max().item()
        assert delta > 1e-6, f'Changing all RoPE rotations to identity had max delta {delta}; the supplied frequencies are ignored.'
        return {'max_abs_delta': delta}
    check('Llama actually uses nontrivial RoPE', llama_uses_frequencies)
    def llama_half():
        out = model.half()(tokens)
        assert out.dtype == torch.float16 and torch.isfinite(out).all()
        return {'dtype': str(out.dtype)}
    check('Llama fp16 multi-token forward', llama_half)
    report = {'scope': 'six nested historical modules, not counted in 248 top-level problem files',
              'repaired': False, 'python_files': files, 'checks': checks,
              'replacement_learning_path': 'interview_core/15_decoder_block_with_cache.py',
              'notes': ['Legacy Llama has no KV cache implementation; start_pos alone does not create a cache.',
                        'The standalone RoPE and SwiGLU calculations pass their checks; integration of RoPE is missing.',
                        'Float64 precision and fp16 end-to-end support are separately checked, not assumed.']}
    (root/'audit/nested_legacy.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print('Nested original Python files:', len(files))
    for c in checks:
        print(c['status'], c['name'], c.get('error', ''))
    print('Checks:', len(checks), 'passed:', sum(c['status']=='pass' for c in checks),
          'failed:', sum(c['status']=='fail' for c in checks))

if __name__ == '__main__':
    main()
