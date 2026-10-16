"""Independent numerical, gradient and boundary tests (not copied public tests)."""
import math
import pytest
import torch
import torch.nn.functional as F
DTYPES=[torch.float32,torch.float64]

def close(a,b):
    torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)

@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('shape',[(1,1,1,1),(3,5,4,2),(7,4,6,5)])
def test_attention_sdpa_noncontiguous(load,dtype,shape):
    s,t,d,v=shape
    q=torch.randn(s,d*2,dtype=dtype)[:,::2]
    k=torch.randn(t,d*2,dtype=dtype)[:,::2]
    value=torch.randn(t,v*2,dtype=dtype)[:,::2]
    close(load(53).self_attention(q,k,value),F.scaled_dot_product_attention(q,k,value))

@pytest.mark.parametrize('heads',[1,2,4])
@pytest.mark.parametrize('dtype',DTYPES)
def test_mha_sdpa(load,heads,dtype):
    q,k,v=[torch.randn(5,8,dtype=dtype) for _ in range(3)]
    split=lambda x:x.reshape(5,heads,8//heads).transpose(0,1)
    ref=F.scaled_dot_product_attention(split(q),split(k),split(v)).transpose(0,1).reshape(5,8)
    close(load(94).multi_head_attention(q,k,v,heads),ref)

@pytest.mark.parametrize('heads',[0,-1,3])
def test_mha_invalid_heads(load,heads):
    x=torch.randn(3,8)
    with pytest.raises(ValueError):load(94).multi_head_attention(x,x,x,heads)

@pytest.mark.parametrize('dtype',DTYPES)
def test_masks_and_empty_rows(load,dtype):
    q,k,v=[torch.randn(4,3,dtype=dtype,requires_grad=True) for _ in range(3)]
    mask=torch.ones(4,4,dtype=torch.bool).tril();mask[1]=False
    out=load(107).masked_attention(q,k,v,mask)
    ref=F.scaled_dot_product_attention(q,k,v,attn_mask=mask)
    close(out,ref); assert torch.equal(out[1],torch.zeros(3,dtype=dtype))
    additive=torch.zeros(4,4,dtype=dtype).masked_fill(~mask,float('-inf'))
    close(out,load(107).masked_attention(q,k,v,additive))
    for g in torch.autograd.grad(out.sum(),(q,k,v)):assert torch.isfinite(g).all()

@pytest.mark.parametrize('kv_heads',[1,2,4])
@pytest.mark.parametrize('dtype',DTYPES)
def test_gqa_matches_sdpa(load,kv_heads,dtype):
    q=torch.randn(2,3,16,dtype=dtype);k=torch.randn(2,5,kv_heads*4,dtype=dtype);v=torch.randn_like(k)
    qh=q.reshape(2,3,4,4).transpose(1,2)
    kh=k.reshape(2,5,kv_heads,4).transpose(1,2).repeat_interleave(4//kv_heads,1)
    vh=v.reshape(2,5,kv_heads,4).transpose(1,2).repeat_interleave(4//kv_heads,1)
    ref=F.scaled_dot_product_attention(qh,kh,vh).transpose(1,2).reshape(2,3,16)
    close(load(391).grouped_query_attention(q,k,v,4,kv_heads),ref)

@pytest.mark.parametrize('block',[1,2,3,16])
@pytest.mark.parametrize('dtype',DTYPES)
def test_blockwise_attention_values_and_gradients(load,block,dtype):
    q=torch.randn(5,4,dtype=dtype,requires_grad=True)
    k=torch.randn(7,4,dtype=dtype,requires_grad=True)
    v=torch.randn(7,3,dtype=dtype,requires_grad=True)
    out=load(208).flash_attention_forward(q,k,v,block)
    ref=F.scaled_dot_product_attention(q,k,v)
    close(out,ref)
    actual=torch.autograd.grad(out.square().sum(),(q,k,v),retain_graph=True)
    expected=torch.autograd.grad(ref.square().sum(),(q,k,v))
    for a,b in zip(actual,expected):close(a,b)

@pytest.mark.parametrize('which',[53,94,107,372,381,222,391])
def test_gradcheck(load,which):
    m=load(which); x=torch.randn(2,4,dtype=torch.float64,requires_grad=True)
    if which==53:
        args=tuple(torch.randn_like(x,requires_grad=True) for _ in range(3)); fn=m.self_attention
    elif which==94:
        args=tuple(torch.randn_like(x,requires_grad=True) for _ in range(3)); fn=lambda *a:m.multi_head_attention(*a,2)
    elif which==107:
        args=tuple(torch.randn_like(x,requires_grad=True) for _ in range(3));fn=lambda *a:m.masked_attention(*a,torch.ones(2,2,dtype=torch.bool).tril())
    elif which==372:args=(x,);fn=lambda a:m.rmsnorm(a,torch.ones(4,dtype=a.dtype))
    elif which==381:args=(x,);fn=lambda a:m.apply_rope(a,torch.tensor([3,4]))
    elif which==391:
        args=tuple(t.detach().clone().requires_grad_() for t in (x.unsqueeze(0),x[:,:2].unsqueeze(0),x[:,:2].unsqueeze(0)));fn=lambda *a:m.grouped_query_attention(*a,2,1)
    else:
        args=(x,);w=torch.randn(4,3,dtype=x.dtype);a=torch.randn(2,3,dtype=x.dtype);b=torch.randn(4,2,dtype=x.dtype);fn=lambda z:m.lora_forward(z,w,a,b,2)
    assert torch.autograd.gradcheck(fn,args,fast_mode=True)

@pytest.mark.parametrize('dtype',DTYPES)
@pytest.mark.parametrize('dim',[1,4,9])
def test_norms(load,dtype,dim):
    x=torch.randn(2,3,dim,dtype=dtype);g=torch.randn(dim,dtype=dtype);b=torch.randn_like(g)
    close(load(109).layer_normalization(x,g,b),F.layer_norm(x,(dim,),g,b,1e-5))
    close(load(372).rmsnorm(x,g),F.rms_norm(x,(dim,),g,1e-5))

@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_rmsnorm_low_precision_no_overflow(load,dtype):
    x=torch.full((2,8),10000.,dtype=dtype)
    out=load(372).rmsnorm(x,torch.ones(8,dtype=dtype))
    assert out.dtype==dtype and torch.isfinite(out).all()
    torch.testing.assert_close(out,torch.ones_like(out),rtol=0.01,atol=0.01)

@pytest.mark.parametrize('dtype',DTYPES)
def test_rope_norm_and_relative_position(load,dtype):
    x=torch.randn(6,8,dtype=dtype);y=torch.randn_like(x);p=torch.arange(6)
    rope=load(381).apply_rope
    close(rope(x,p).norm(dim=-1),x.norm(dim=-1))
    close(rope(x,torch.zeros(6,dtype=torch.long)),x)
    dot1=(rope(x,p)*rope(y,p+2)).sum(-1)
    dot2=(rope(x,p+5)*rope(y,p+7)).sum(-1)
    close(dot1,dot2)
    assert rope(x,p).dtype==dtype

def test_rope_rejects_odd_width(load):
    with pytest.raises(ValueError):load(381).apply_rope(torch.zeros(3,5),torch.arange(3))

@pytest.mark.parametrize('dim',[1,3,8])
def test_sinusoidal_odd_dimension(load,dim):
    out=load(85).pos_encoding(5,dim)
    assert out.shape==(5,dim) and out.dtype==torch.float16
    close(out[0,0::2],torch.zeros_like(out[0,0::2]));close(out[0,1::2],torch.ones_like(out[0,1::2]))

def test_qkv_projection_and_learned_positions(load):
    x=torch.randn(2,3,4);weights=[torch.randn(4,4) for _ in range(3)]
    for out,w in zip(load(94).compute_qkv(x,*weights),weights):close(out,x@w)
    table=torch.randn(8,4,requires_grad=True)
    out=load(375).learned_positional_encoding(x,table,3);close(out,x+table[3:6])
    out.sum().backward();close(table.grad[3:6],torch.full((3,4),2.))
    with pytest.raises(ValueError):load(375).learned_positional_encoding(x,table,7)

def test_swiglu_gate_convention_and_gradient(load):
    x=torch.randn(2,3,8,dtype=torch.float64,requires_grad=True)
    out=load(156).SwiGLU(x);value,gate=x.chunk(2,-1)
    close(out,value*F.silu(gate));out.sum().backward();assert x.grad is not None
    assert load(156).SwiGLU(torch.ones(2,4,dtype=torch.long)).is_floating_point()

@pytest.mark.parametrize('p',[0.0,0.4,1.0])
def test_ffn_train_eval_residual(load,p):
    block=load(178).FFNBlock(4,7,p).double();x=torch.randn(2,3,4,dtype=torch.float64,requires_grad=True)
    block.eval();ref=x+block.linear2(F.relu(block.linear1(x)));close(block(x),ref)
    block.train();out=block(x)
    if p==1:close(out,x)
    out.sum().backward();assert torch.isfinite(x.grad).all()

def test_dropout_forward_backward_and_eval(load):
    layer=load(151).DropoutLayer(.3);x=torch.ones(10000)
    with pytest.raises(ValueError):layer.backward(x)
    out=layer.forward(x);assert abs(out.mean().item()-1)<.04
    close(layer.backward(x),out);mask=layer.mask.clone();close(layer.forward(x,False),x);close(mask,layer.mask)
    for p in [-.1,1]:
        with pytest.raises(ValueError):load(151).DropoutLayer(p)

@pytest.mark.parametrize('k',[1,2,4])
def test_sparse_experts_vs_loop_reference(load,k):
    x=torch.randn(2,3,4,dtype=torch.float64,requires_grad=True)
    we=torch.randn(4,4,4,dtype=torch.float64,requires_grad=True);wg=torch.randn(4,4,dtype=torch.float64,requires_grad=True)
    out=load(125).moe(x,we,wg,4,k);result=[]
    for token in x.reshape(-1,4):
        logits=token@wg;idx=logits.argsort(descending=True)[:k];weights=logits[idx].softmax(0)
        result.append(sum((weights[j]*(token@we[i]) for j,i in enumerate(idx))))
    close(out,torch.stack(result).reshape_as(x));out.sum().backward()
    for t in (x,we,wg):assert t.grad is not None and torch.isfinite(t.grad).all()

@pytest.mark.parametrize('k',[1,2,4])
def test_moe_routing_and_noisy_gating(load,k):
    logits=torch.tensor([[1000.,1001.,998.,995.],[-1000.,-996.,-999.,-997.]])
    outputs=torch.randn(2,4,3)
    out=load(229).moe_topk_routing(logits,outputs,k)
    reference=[]
    for l,o in zip(logits,outputs):
        ix=l.argsort(descending=True)[:k];reference.append((l[ix].softmax(0)[:,None]*o[ix]).sum(0))
    close(out,torch.stack(reference))
    gates=load(124).noisy_topk_gating(logits,torch.eye(4),torch.zeros(4,4),torch.zeros_like(logits),k)
    close(gates.sum(-1),torch.ones(2));assert torch.equal((gates>0).sum(-1),torch.full((2,),k))

def test_shared_expert_decomposition_and_ties(load):
    x=torch.eye(2);experts=[torch.eye(2)*i for i in [2.,3.,4.]]
    out=load(409).moe_shared_expert_forward(x,torch.zeros(2,3),torch.eye(2),experts,2)
    close(out['routing_weights'],torch.full((2,2),.5))
    assert out['routing_indices'].tolist()==[[0,1],[0,1]]
    close(out['output'],3.5*x);close(out['output'],out['shared_output']+out['routed_output'])

def test_lora_and_qlora_conventions(load):
    x=torch.randn(2,3,dtype=torch.float64);w=torch.randn(3,5,dtype=x.dtype)
    a=torch.randn(2,5,dtype=x.dtype);b=torch.randn(3,2,dtype=x.dtype)
    close(load(222).lora_forward(x,w,a,b,4),x@(w+2*b@a))
    close(load(222).lora_forward(x,w,a,torch.zeros_like(b)),x@w)
    q=torch.randint(-4,4,(3,5));close(load(223).qlora_forward(x,q,.25,1.5,a,b,4),x@(q.double()*.25+1.5)+2*x@b@a)

def test_single_token_kv_cache_matches_causal_attention(load):
    x=torch.randn(6,4,dtype=torch.float64);wq=torch.randn(4,3,dtype=x.dtype);wk=torch.randn_like(wq);wv=torch.randn(4,2,dtype=x.dtype)
    cache=None;outputs=[]
    for i in range(6):
        out,cache=load(376).kv_cache_attention_step(x[i],wq,wk,wv,cache)
        outputs.append(out);assert cache[0].shape==(i+1,3)
    ref=F.scaled_dot_product_attention(x@wq,x@wk,x@wv,is_causal=True)
    close(torch.stack(outputs),ref)

def test_window_attention_is_bidirectional_radius(load):
    q=k=torch.zeros(4,2);v=torch.arange(4.).unsqueeze(-1)
    close(load(388).sliding_window_attention(q,k,v,1),torch.tensor([[.5],[1.],[2.],[2.5]]))

def test_mqa_projection(load):
    x=torch.randn(3,4,dtype=torch.float64);queries=[torch.randn(4,2,dtype=x.dtype) for _ in range(3)]
    wk=torch.randn(4,2,dtype=x.dtype);wv=torch.randn(4,3,dtype=x.dtype);wo=torch.randn(9,4,dtype=x.dtype)
    ref=torch.cat([F.scaled_dot_product_attention(x@q,x@wk,x@wv) for q in queries],-1)@wo
    close(load(390).multiquery_attention(x,queries,wk,wv,wo),ref.round(decimals=4))

def test_qk_norm_zero_rows(load):
    q=torch.tensor([[0.,0.],[3.,4.]]);k=torch.eye(2);v=torch.eye(2)
    out,weights=load(407).qk_norm_attention(q,k,v,.5)
    close(out,weights);close(weights[0],torch.tensor([.5,.5]));close(weights[1],torch.tensor([1.2,1.6]).softmax(0))

def test_pre_post_norm_residual(load):
    x=torch.randn(2,4,dtype=torch.float64);w=torch.zeros(4,4,dtype=x.dtype);b=torch.zeros(4,dtype=x.dtype);g=torch.ones_like(b)
    f=load(408).transformer_block
    close(f(x,w,b,w,b,g,b,g,b,'pre_norm'),x)
    norm=lambda a:F.layer_norm(a,(4,),g,b,1e-5)
    close(f(x,w,b,w,b,g,b,g,b,'post_norm'),norm(norm(x)))
    with pytest.raises(ValueError):f(x,w,b,w,b,g,b,g,b,'invalid')

def test_losses_and_stability(load):
    close(torch.tensor(load(23).softmax([10000.,10001.,10002.])),torch.tensor([0.,1.,2.]).softmax(0))
    close(load(39).log_softmax([10000.,10001.,10002.]),torch.tensor([0.,1.,2.],dtype=torch.float64).log_softmax(0).round(decimals=4))
    assert math.isfinite(load(263).binary_cross_entropy(torch.tensor([1.,0.]),torch.tensor([0.,1.])))
    result=load(382).dpo_loss([0.],[-10000.],[0.],[0.],.1);assert result['loss']==0
    zero=load(382).dpo_loss([0.],[0.],[0.],[0.],.1);assert zero['loss']==.6931
    z=torch.tensor([[1.,0.],[1.,0.],[0.,1.],[0.,1.]])
    expected=math.log(1+2/math.e)
    assert abs(load(384).contrastive_loss(z,1.)-expected)<1e-6

def test_grpo_advantage_and_kl(load):
    close(load(224).compute_group_relative_advantage(torch.ones(3)),torch.zeros(3))
    a=load(224).compute_group_relative_advantage(torch.arange(5,dtype=torch.float64))
    close(a.mean(),a.new_tensor(0.));close(a.square().mean(),a.new_tensor(1.))
    p=torch.tensor([.2,.5,.8],dtype=torch.float64);r=torch.tensor([.3,.5,.4],dtype=torch.float64)
    estimate=load(225).kl_divergence_estimator(p,r);close(estimate,r/p-(r/p).log()-1)
    objective=load(101).grpo_objective([1.,1.],[2.,-1.],[.5,.5],[.5,.5])
    close(objective,torch.tensor(.5,dtype=torch.float64))

def test_adam_optimizer_oracle(load):
    x=torch.tensor([1.,-3.],dtype=torch.float64,requires_grad=True)
    optimizer=torch.optim.Adam([x],lr=.03)
    for _ in range(7):optimizer.zero_grad();(x.square().sum()).backward();optimizer.step()
    out=load(49).adam_optimizer(lambda a:a.square().sum(),lambda a:2*a,torch.tensor([1.,-3.],dtype=torch.float64),learning_rate=.03,num_iterations=7)
    close(out,x.detach())

@pytest.mark.parametrize('p',[.01,.5,.8,1.])
def test_top_p_minimal_prefix(load,p):
    logits=[1.,1.,0.,-2.];actual=load(383).top_p_sampling(logits,p)
    probs=torch.tensor(logits,dtype=torch.float64).softmax(0);order=probs.argsort(descending=True,stable=True)
    count=int((actual>0).sum());assert count>=1 and actual.sum().item()==pytest.approx(1)
    assert probs[order[:count]].sum().item()+1e-15>=p
    if count>1:assert probs[order[:count-1]].sum().item()<p+1e-15
    if p==.01:assert actual.tolist()==[1.,0.,0.,0.]

def test_sampling_local_rng_and_greedy(load):
    state=torch.random.get_rng_state().clone()
    f=load(419).combined_sampling
    a=f([1.,4.,2.],.7,2,.9,10);b=f([1.,4.,2.],.7,2,.9,10)
    assert a==b and torch.equal(torch.random.get_rng_state(),state)
    assert f([3.,3.,1.],0)['sampled_token']==0

def test_perplexity_quantization_and_memory(load):
    assert load(320).calculate_perplexity([.25,.25])==pytest.approx(4.)
    assert load(429).quantization_quality_check([-2.,-3.],[-2.,-3.])['quality']=='excellent'
    zeros=load(294).int8_quantize([0.,0.]);assert zeros=={'quantized':[0,0],'scale':1.,'dequantized':[0.,0.]}
    config={'num_layers':2,'num_attention_heads':4,'num_kv_heads':1,'hidden_size':16,'dtype_bytes':2}
    result=load(418).estimate_kv_cache_size(config,3,5)
    assert result['kv_cache_elements']==2*2*3*5*1*4
    assert result['kv_cache_size_bytes']==result['kv_cache_elements']*2

def test_tokenizer_independently_not_public_redefinition(load):
    tok=load(374).CharTokenizer('cabca')
    assert tok.stoi=={'<BOS>':0,'<EOS>':1,'a':2,'b':3,'c':4}
    encoded=tok.encode('bac');assert encoded.dtype==torch.long and encoded.tolist()==[0,3,2,4,1]
    assert tok.decode(encoded)=='<BOS>bac<EOS>'
    assert tok.embedding(encoded).shape==(5,5)
    with pytest.raises(KeyError):tok.encode('z')
