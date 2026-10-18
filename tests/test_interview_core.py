"""Independent tests for all 14 interview modules, separate from website interfaces."""
import pytest
import torch
import torch.nn.functional as F

def close(a,b):torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)

def test_mha_reference(load):
    model=load(1,True).MultiHeadAttention(8,2).double();x=torch.randn(2,4,8,dtype=torch.float64)
    split=lambda z:z.reshape(2,4,2,4).transpose(1,2)
    q,k,v=(split(f(x)) for f in (model.q_proj,model.k_proj,model.v_proj))
    mask=torch.ones(4,4,dtype=torch.bool).tril()
    ref=F.scaled_dot_product_attention(q,k,v,attn_mask=mask).transpose(1,2).reshape(2,4,8)
    close(model(x,mask),model.out_proj(ref))

def test_mha_empty_mask_and_half_precision(load):
    model=load(1,True).MultiHeadAttention(8,2).half();x=torch.randn(1,3,8,dtype=torch.float16)
    out=model(x,torch.zeros(3,3,dtype=torch.bool))
    assert torch.isfinite(out).all();assert torch.equal(out,torch.zeros_like(out))

def test_gqa_module_reference(load):
    model=load(2,True).GroupedQueryAttention(8,4,2).double();x=torch.randn(2,3,8,dtype=torch.float64)
    q=model.q_proj(x).reshape(2,3,4,2).transpose(1,2)
    kv=lambda f:f(x).reshape(2,3,2,2).transpose(1,2).repeat_interleave(2,1)
    ref=F.scaled_dot_product_attention(q,kv(model.k_proj),kv(model.v_proj)).transpose(1,2).reshape(2,3,8)
    close(model(x),model.out_proj(ref))

def test_gqa_invalid_model_width(load):
    with pytest.raises(ValueError):load(2,True).GroupedQueryAttention(10,4,2)

def test_rms_module_fp16_accumulation(load):
    model=load(3,True).RMSNorm(8).half();x=torch.full((2,3,8),10000.,dtype=torch.float16)
    out=model(x);assert torch.isfinite(out).all();torch.testing.assert_close(out,torch.ones_like(out),rtol=.01,atol=.01)

def test_rotary_half_split_convention(load):
    mod=load(4,True);x=torch.randn(2,4,3,8)
    c,s=mod.precompute_freqs_cis(8,4);out=mod.apply_rotary_emb(x,c,s)
    a,b=x.chunk(2,-1);ref=torch.cat([a*c[None,:,None]-b*s[None,:,None],b*c[None,:,None]+a*s[None,:,None]],-1)
    close(out,ref);close(out.norm(dim=-1),x.norm(dim=-1))

def test_swiglu_module(load):
    model=load(5,True).SwiGLU(6,10).double();x=torch.randn(2,4,6,dtype=torch.float64,requires_grad=True)
    close(model(x),F.linear(F.silu(F.linear(x,model.w_gate.weight))*F.linear(x,model.w_up.weight),model.w_down.weight))
    model(x).sum().backward();assert x.grad is not None

@pytest.mark.parametrize('chunks',[[1,1,1,1,1,1],[2,3,1],[3,3],[6]])
def test_cached_attention_matches_full_causal(load,chunks):
    model=load(6,True).AttentionWithKVCache(8,2).double();x=torch.randn(2,6,8,dtype=torch.float64)
    split=lambda z:z.reshape(2,6,2,4).transpose(1,2)
    q,k,v=(split(f(x)) for f in (model.q_proj,model.k_proj,model.v_proj))
    reference=F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(2,6,8)
    reference=model.out_proj(reference);cache=None;outputs=[];start=0
    for length in chunks:
        out,cache=model(x[:,start:start+length],cache);outputs.append(out);start+=length
        assert cache[0].shape==(2,2,start,4)
    close(torch.cat(outputs,1),reference)

def test_cached_attention_no_future_leak(load):
    model=load(6,True).AttentionWithKVCache(8,2).double();x=torch.randn(1,5,8,dtype=torch.float64)
    changed=x.clone();changed[:,3:]=torch.randn_like(changed[:,3:])*100
    close(model(x)[0][:,:3],model(changed)[0][:,:3])

def test_lora_frozen_base_and_gradients(load):
    model=load(7,True).LoRALinear(4,6,2).double();x=torch.randn(3,4,dtype=torch.float64)
    close(model(x),model.pretrained(x));model(x).square().sum().backward()
    assert model.pretrained.weight.grad is None
    assert torch.count_nonzero(model.lora_A.grad)==0
    assert torch.count_nonzero(model.lora_B.grad)>0
    model.zero_grad()
    with torch.no_grad():model.lora_B.fill_(.1)
    model(x).square().sum().backward();assert torch.count_nonzero(model.lora_A.grad)>0

def test_softmax_cross_entropy_gradients(load):
    mod=load(8,True);logits=(torch.randn(5,7,dtype=torch.float64)+10000).requires_grad_();targets=torch.tensor([0,2,4,6,1])
    close(mod.stable_softmax(logits),logits.softmax(-1))
    out=mod.stable_cross_entropy(logits,targets);ref=F.cross_entropy(logits,targets);close(out,ref)
    close(torch.autograd.grad(out,logits)[0],torch.autograd.grad(ref,logits)[0])

def test_dpo_tensor_loss(load):
    pc=torch.tensor([-1.,-1000.],requires_grad=True);pr=torch.tensor([-3.,0.],requires_grad=True)
    refc=torch.tensor([-1.,-1.]);refr=torch.tensor([-2.,-2.])
    out=load(9,True).dpo_loss(pc,pr,refc,refr,.2)
    close(out,F.softplus(-.2*((pc-pr)-(refc-refr))).mean());out.backward();assert torch.isfinite(pc.grad).all()

def test_clip_symmetric_infonce(load):
    a=torch.randn(3,5,dtype=torch.float64,requires_grad=True);b=torch.randn(3,5,dtype=torch.float64,requires_grad=True)
    f=load(10,True).infonce_loss;logits=F.normalize(a,dim=-1)@F.normalize(b,dim=-1).T/.2
    ref=(F.cross_entropy(logits,torch.arange(3))+F.cross_entropy(logits.T,torch.arange(3)))/2
    close(f(a,b,.2),ref);close(f(a,b,.2),f(b,a,.2));f(a,b,.2).backward();assert a.grad is not None

def test_adam_steps_against_optimizer(load):
    x=torch.tensor([1.,-2.],dtype=torch.float64);parameter=x.clone().requires_grad_();m=torch.zeros_like(x);v=torch.zeros_like(x)
    optimizer=torch.optim.Adam([parameter],lr=.02)
    for step in range(1,6):
        grad=torch.tensor([step*.2,-step*.1],dtype=x.dtype)
        parameter.grad=grad;optimizer.step();optimizer.zero_grad()
        x,m,v=load(11,True).adam_step(x,grad,m,v,step,lr=.02)
        close(x,parameter.detach())

def test_manual_mlp_backward(load):
    x=torch.randn(5,3,dtype=torch.float64);y=torch.randn(5,2,dtype=x.dtype)
    weights=[torch.randn(*shape,dtype=x.dtype,requires_grad=True) for shape in [(3,4),(4,),(4,2),(2,)]]
    loss,grads=load(12,True).mlp_forward_backward(x,y,*weights)
    w1,b1,w2,b2=weights;ref=.5*((F.relu(x@w1+b1)@w2+b2-y)**2).mean()
    close(loss,ref)
    for a,b in zip(grads,torch.autograd.grad(ref,weights)):close(a,b)

def test_kmeans_labels_match_returned_centers(load):
    torch.manual_seed(2);x=torch.randn(30,2)
    centers,labels=load(13,True).kmeans(x,4,max_iters=1,tol=0.)
    expected=((x[:,None]-centers[None])**2).sum(-1).argmin(-1)
    assert torch.equal(labels,expected)

def test_kmeans_large_tolerance_updates_centers(load):
    x=torch.tensor([[0.],[2.],[9.],[11.]])
    centers,labels=load(13,True).kmeans(x,1,max_iters=2,tol=1e9)
    close(centers,x.mean(0,keepdim=True))

def test_pca_singular_value_energy(load):
    x=torch.randn(12,5,dtype=torch.float64);out=load(14,True).pca(x,3)
    centered=x-x.mean(0);singular=torch.linalg.svdvals(centered)
    close(out.T@out,torch.diag(singular[:3]**2));close(out.mean(0),torch.zeros(3,dtype=x.dtype))

@pytest.mark.parametrize('chunks',[[1,1,1,1,1,1],[2,3,1],[3,3]])
def test_composed_decoder_cache_and_rope_offsets(load,chunks):
    model=load(15,True).DecoderBlock(16,4,2,24).double()
    x=torch.randn(2,6,16,dtype=torch.float64);full,full_cache=model(x)
    cache=None;outputs=[];start=0
    for length in chunks:
        out,cache=model(x[:,start:start+length],cache);start+=length;outputs.append(out)
        assert cache[0].shape==(2,2,start,4)
    close(torch.cat(outputs,1),full);close(cache[0],full_cache[0]);close(cache[1],full_cache[1])

def test_composed_decoder_no_future_and_gradient(load):
    model=load(15,True).DecoderBlock(16,4,1,24).double()
    x=torch.randn(1,5,16,dtype=torch.float64,requires_grad=True)
    out,_=model(x);changed=x.detach().clone();changed[:,3:]*=100
    close(out[:,:3],model(changed)[0][:,:3]);out.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

def test_rotary_offset_matches_full_rotation(load):
    module=load(4,True);x=torch.randn(2,7,2,8,dtype=torch.float64)
    c,s=module.precompute_freqs_cis(8,7,dtype=torch.float64)
    whole=module.apply_rotary_emb(x,c,s)
    chunk=module.apply_rotary_emb(x[:,3:6],c,s,start_pos=3)
    close(chunk,whole[:,3:6]);assert chunk.dtype==x.dtype

def test_cross_entropy_sequence_shape(load):
    logits=torch.randn(2,3,5,dtype=torch.float64,requires_grad=True);labels=torch.tensor([[0,1,2],[2,3,4]])
    close(load(8,True).stable_cross_entropy(logits,labels),F.cross_entropy(logits.reshape(-1,5),labels.reshape(-1)))
