"""Rebuild the reviewed mapping and reports from pinned source/public snapshots.
Used once for this PR; preparation archives originals before removing legacy aliases.
Never changes the NumPy directory. Public data is supplied locally, not fetched here.
"""
import argparse, ast, base64, collections, contextlib, hashlib, io, json, pathlib, re, signal

ALIAS = {
'104-Batch Normalization for BCHW Input':115,'108-Label Collinearity Control':360,
'110-Distance Correlation Squared':359,'115-Character-Level Tokenizer':374,
'118-RMSNorm':372,'119-RoPE':381,'120-KV Cache':376,
'128-Grouped Query Attention':391,"13-Determinant of a 4x4 Matrix using Laplace's Expansion":13,
'130-Mixture-of-Experts Gating':124,'131-Swish SwiGLU Activation':156,
'134-FP4 Quantization with Microscaling':427,'135-Attention Sinks':457,
'136-Implement GRPO Objective Function':101,'139-LoRA Forward Pass':222,
'140-QLoRA Forward Pass':223,'141-Calculate Perplexity':320,
'142-Continuous Batching Simulator':425,'143-Flash Attention Forward':208,
'144-Roofline Model Analysis':415,'145-Quantization Quality Check':429,
'59-Implement Long Short-Term Memory Network':59,
'67-Implement Compressed Column Sparse Matrix Format':67,
'83-Dot Product Calculator Easy':83,'98-Implement the PReLU Activation Function':98,
'99-Implement Local Response Normalization':189,
}
BASE_COMMIT='c201923f74cd181fb16259ff2cf42754334bf2e3'
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2))
def read(path):return json.loads(path.read_text())
def prepare(root,original,public,extra):
    import torch
    audit=root/'audit';audit.mkdir(exist_ok=True)
    catalog=read(public/'live-catalog.json')['problems']
    norm=lambda s:re.sub('[^a-z0-9]','',s.lower())
    bytitle=collections.defaultdict(list)
    for p in catalog:bytitle[norm(p['title'])].append(p)
    byid={int(p['id']):p for p in catalog}
    mapping=[];inventory=[];moves=[]
    for folder in ('pytorch','interview_core'):
        for p in sorted((original/folder).iterdir()):
            if not p.is_file():continue
            target=root/'archive'/('pytorch_legacy' if folder=='pytorch' else folder)/p.name
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():assert target.read_bytes()==p.read_bytes(),target
            else:target.write_bytes(p.read_bytes())
            if folder=='pytorch':
                existing=root/folder/p.name
                if existing.exists():
                    assert existing.read_bytes()==p.read_bytes(),('Legacy alias unexpectedly changed',existing)
                    existing.unlink()
            if p.suffix!='.py':continue
            path=f'{folder}/{p.name}';source=p.read_text();tree=ast.parse(source)
            rec={'path':path,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'syntax':'pass','import':'not_run','flags':[]}
            np_import=any(isinstance(n,(ast.Import,ast.ImportFrom)) and any(a.asname=='np' or a.name=='numpy' for a in n.names) for n in ast.walk(tree))
            if any(isinstance(n,ast.Name) and n.id=='np' for n in ast.walk(tree)) and not np_import:rec['flags'].append('unbound_numpy_name')
            for n in ast.walk(tree):
                if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):
                    if n.func.attr=='transpose' and len(n.args)>2:rec['flags'].append('numpy_style_transpose')
                    if n.func.attr=='round' and isinstance(n.func.value,ast.Name) and n.func.value.id=='torch' and len(n.args)>1:rec['flags'].append('positional_torch_round_decimals')
                    if n.func.attr=='sqrt' and isinstance(n.func.value,ast.Name) and n.func.value.id=='torch' and n.args and isinstance(n.args[0],ast.Name) and n.args[0].id in ['d_k','d','D','dim','n']:rec['flags'].append('possible_scalar_torch_sqrt')
            rec['flags']=sorted(set(rec['flags']))
            try:
                def expire(*_):raise TimeoutError('Import timeout')
                signal.signal(signal.SIGALRM,expire);signal.alarm(2)
                with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                    exec(compile(tree,str(p),'exec'),{'__name__':'audit_import','__file__':str(p)})
                rec['import']='pass'
            except Exception as e:rec.update({'import':'fail','error':f'{type(e).__name__}: {e}'})
            finally:signal.alarm(0)
            rec['functions']=[n.name for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
            inventory.append(rec)
            if folder!='pytorch':continue
            old,title=p.stem.split('-',1);old=int(old);exact=bytitle[norm(title)]
            match=next((x for x in exact if int(x['id'])==old),exact[0] if exact else None)
            status=('exact_id_title' if int(match['id'])==old else 'renumbered_exact_title') if match else 'unmatched_extension_no_official_id'
            if match is None and p.stem in ALIAS:match=byid[ALIAS[p.stem]];status='title_alias_contract_requires_audit'
            row={'path':path,'old_id':old,'old_title':title,'official_id':int(match['id']) if match else None,'official_title':match['title'] if match else None,'mapping':status}
            if match:row['official_url']=f"https://www.deep-ml.com/problems/{match['id']}"
            row['archive_path']=str(target.relative_to(root))
            if p.stem=='124-Calculate Number of Parameters in Neural Network':
                assert row['official_id']==291
                row['alternative_official_ids']=[371]
                row['mapping_note']='Official catalog has identical titles at #291 and #371; #291 is the canonical review choice, not a proven historical identification.'
            mapping.append(row)
    for p in sorted(original.iterdir()):
        if not p.is_file():continue
        if p.name=='README.md':dest='archive/README.original.md'
        elif p.name=='Deep_ML_Problems_List.md':dest='archive/Deep_ML_Problems_List.original.md'
        else:dest='archive/legacy_metadata/'+p.name
        target=root/dest;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert target.read_bytes()==p.read_bytes()
        else:target.write_bytes(p.read_bytes())
        if p.name not in ('README.md','Deep_ML_Problems_List.md') and (root/p.name).exists():
            assert (root/p.name).read_bytes()==p.read_bytes();(root/p.name).unlink()
        moves.append({'from':p.name,'to':dest,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    write(audit/'id_mapping.json',mapping);write(audit/'inventory_original.json',inventory);write(audit/'archive_moves.json',moves)
    snapshot={'provenance':read(public/'live-provenance.json'),'problems':{}}
    for folder in (public,extra):
        for p in sorted((folder/'problems').glob('*.json')):
            data=read(p)
            snapshot['problems'][p.stem]={'id':data.get('id',p.stem),'title':data.get('title'),'pytorch_starter_code':base64.b64decode(data.get('pytorch_starter_code') or '').decode(),'pytorch_test_cases':data.get('pytorch_test_cases') or []}
    snapshot['additional_provenance']=read(extra/'provenance-extra.json')
    text=json.dumps(snapshot,ensure_ascii=False,sort_keys=True,indent=2)
    assert hashlib.sha256(text.encode()).hexdigest()=='31a05a737687125dd198933055c34ea4fd9887e58f86c0bc8feda724035f8e7b','Public snapshot differs from local reviewed snapshot'
    (audit/'public_examples.json').write_text(text)
    assert len(inventory)==262 and len(mapping)==248
    assert sum(r['import']=='fail' for r in inventory)==12
    assert len({r['official_id'] for r in mapping if r['official_id'] is not None})==218
    print('Archived all original source files and reconstructed the pinned public examples.')

def finalize(root,public):
    audit=root/'audit';mapping=read(audit/'id_mapping.json')
    original=read(audit/'public_original.json');core=read(audit/'public_core.json')
    base={f['path']:f for f in original['files']};current={f['official_id']:f for f in core['files']}
    cat={int(p['id']):p for p in read(public/'live-catalog.json')['problems']}
    manifest=[]
    for pid in sorted({r['official_id'] for r in mapping if r['official_id'] is not None}|set(current)):
        new=current.get(pid)
        manifest.append({'official_id':pid,'official_title':cat[pid]['title'].strip(),'official_url':f'https://www.deep-ml.com/problems/{pid}',
            'solution_path':new['path'] if new else None,'review_status':'reviewed_core' if new else 'legacy_only_not_repaired',
            'public_examples_status':new['status'] if new else 'not_a_current_solution',
            'legacy_files':[{'original_path':r['path'],'archive_path':r['archive_path'],'old_id':r['old_id'],'baseline_status':base[r['archive_path']]['status']} for r in mapping if r['official_id']==pid]})
    write(audit/'manifest.json',manifest)
    assert len(manifest)==219
    assert collections.Counter(f['status'] for f in original['files'])=={'fail':198,'pass':30,'non_validating_public_examples':7,'unmapped_extension':3,'no_public_pytorch_tests':10}
    assert collections.Counter(t['status'] for f in core['files'] for t in f['tests'])=={'pass':167,'not_exercised':7}
    lines=['# Deep-ML 题号索引：PyTorch 复习版','','核对日期：2026-10-15（America/Indiana/Indianapolis）。题号与题名来自官网公开 API，来源和抓取时间见 audit/public_examples.json。','',
        '**每个官网 ID 只出现一行。** 40 道核心题已重写；其他题仅保留归档映射，不能当作已修复答案。原 248 个文件涉及 218 个官网 ID；新增正确的 #125 Sparse MoE 后，共 219 个 ID。','',
        '原 124-Calculate Number of Parameters... 本次规范到 #291；官网另有同名有效题 #371，无法仅凭题名唯一确定历史来源，映射中保留说明。','',
        '| 官网题号 | 官网题名 | 本轮实现 | 旧稿数 |','|---|---|---|---|']
    for row in manifest:
        code=f"[代码]({row['solution_path']})" if row['solution_path'] else '未修复；见归档映射'
        lines.append(f"| [#{row['official_id']}]({row['official_url']}) | {row['official_title'].replace('|','/')} | {code} | {len(row['legacy_files'])} |")
    lines+=['','## 自定义练习：不强行编造官网题号','','| 原文件 | 处理 |','|---|---|']
    for row in mapping:
        if row['official_id'] is None:lines.append(f"| {pathlib.Path(row['path']).name} | 未找到可确认的一一对应题；原稿保留在 archive/pytorch_legacy/ |")
    lines+=['','逐文件原号、新号和归档路径：[id_mapping.json](audit/id_mapping.json)。','复习顺序：[REVIEW_CHECKLIST.md](REVIEW_CHECKLIST.md)。测试边界：[AUDIT_REPORT.md](AUDIT_REPORT.md)。','']
    (root/'Deep_ML_Problems_List.md').write_text('\n'.join(lines))
    print('Canonical index:',len(manifest),'IDs; public results verified against recorded local counts.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,default=pathlib.Path('.'));p.add_argument('--original',type=pathlib.Path);p.add_argument('--public',type=pathlib.Path,required=True);p.add_argument('--extra',type=pathlib.Path);p.add_argument('--finalize',action='store_true');a=p.parse_args()
    if a.finalize:finalize(a.root,a.public)
    else:
        if not a.original or not a.extra:p.error('--original and --extra are required for preparation')
        prepare(a.root,a.original,a.public,a.extra)
