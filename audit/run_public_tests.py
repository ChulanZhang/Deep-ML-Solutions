"""Replay captured public PyTorch examples, not Deep-ML's private judge.

Usage: python audit/run_public_tests.py --source original|verified --output FILE
Comparison preserves output nesting/words, ignores whitespace/commas, and
compares numbers with atol=5.1e-5, rtol=1e-5 (published examples round to 4dp).
Each example runs in a fresh module with a 5-second POSIX timeout.
"""
import ast, argparse, contextlib, hashlib, io, json, math, pathlib, re, signal, sys, types
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1]
TOKEN=re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[A-Za-z_]\w*|[^\s,]')
def equivalent(actual,expected):
    a,b=TOKEN.findall(actual.strip()),TOKEN.findall(str(expected).strip())
    if len(a)!=len(b):return False
    for x,y in zip(a,b):
        try:
            fx,fy=float(x),float(y)
            if math.isnan(fx) or math.isnan(fy):
                if not (math.isnan(fx) and math.isnan(fy)):return False
            elif not math.isclose(fx,fy,rel_tol=1e-5,abs_tol=5.1e-5):return False
        except ValueError:
            if x!=y:return False
    return True
class Timeout(Exception):pass
def alarm(*_):raise Timeout('5-second public-example deadline')
def run_one(path,case):
    source=path.read_text(); mod=types.ModuleType('__main__');mod.__file__=str(path)
    oldmain=sys.modules['__main__']; buf=io.StringIO();err=io.StringIO()
    calls=set(); oldprofile=sys.getprofile()
    def profile(frame,event,arg):
        if event=='call' and frame.f_code.co_filename==str(path) and frame.f_code.co_name!='<module>':
            calls.add(frame.f_code.co_name)
    definitions=lambda text:{n.name for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    shadowed=sorted(definitions(source)&definitions(case['test']))
    # Skip the solution's own demo, then expose its functions for test imports.
    mod.__dict__['__name__']='solution_under_test'
    try:
        if hasattr(signal,'SIGALRM'):signal.signal(signal.SIGALRM,alarm);signal.alarm(5)
        with contextlib.redirect_stdout(buf),contextlib.redirect_stderr(err):
            exec(compile(source,str(path),'exec'),mod.__dict__)
            sys.modules['__main__']=mod
            sys.setprofile(profile)
            exec(compile(case['test'],'<official-public-example>','exec'),mod.__dict__)
            sys.setprofile(oldprofile)
        text=buf.getvalue().strip()
        return {'status':('pass' if calls else 'not_exercised') if equivalent(text,case['expected_output']) else 'mismatch', 'called_solution_functions':sorted(calls), 'shadowed_solution_definitions':shadowed,'actual':text,'expected':case['expected_output']}
    except BaseException as exc:
        return {'status':'error','error':f'{type(exc).__name__}: {exc}','actual':buf.getvalue().strip(),'expected':case['expected_output']}
    finally:
        sys.setprofile(oldprofile)
        sys.modules['__main__']=oldmain
        if hasattr(signal,'SIGALRM'):signal.alarm(0)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',choices=['original','verified'],default='verified');parser.add_argument('--output',required=True);parser.add_argument('--definitions',default=str(ROOT/'audit/public_examples.json'));args=parser.parse_args()
    torch.set_num_threads(1)
    definitions=json.loads(pathlib.Path(args.definitions).read_text())['problems']
    mapping=json.loads((ROOT/'audit/id_mapping.json').read_text())
    if args.source=='original':entries=[(ROOT/r['archive_path'],r['official_id']) for r in mapping]
    else:entries=[(p,int(p.name.split('_',1)[0])) for p in sorted((ROOT/'pytorch').glob('[0-9]*.py'))]
    results=[]
    for path,pid in entries:
        d=definitions.get(str(pid),{});cases=d.get('pytorch_test_cases') or []
        result={'path':str(path.relative_to(ROOT)),'official_id':pid,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'tests':[]}
        if not cases:result['status']='no_public_pytorch_tests' if pid is not None else 'unmapped_extension'
        else:
            result['tests']=[dict(index=i+1,**run_one(path,case)) for i,case in enumerate(cases)]
            statuses={t['status'] for t in result['tests']}
            result['status']=('non_validating_public_examples' if statuses=={'not_exercised'} else
                'pass' if statuses <= {'pass','not_exercised'} else 'fail')
        results.append(result)
    import platform,collections
    report={'source':args.source,'python':platform.python_version(),'torch':torch.__version__,'cuda_tested':False,'comparison':'numeric atol=5.1e-5, rtol=1e-5; nesting/words preserved; whitespace/commas ignored','official_judge_submission':False,'files':results}
    pathlib.Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2))
    counts=collections.Counter(r['status'] for r in results);tc=collections.Counter(t['status'] for r in results for t in r['tests'])
    print('Files:',dict(counts),'Examples:',dict(tc))
    for r in results:
        if r['status']=='fail':
            t=next(t for t in r['tests'] if t['status'] not in ('pass','not_exercised'));print(r['path'],f"#{t['index']}",t.get('error',str(t.get('actual'))[:130]))
    return 1 if args.source=='verified' and counts.get('fail') else 0
if __name__=='__main__':raise SystemExit(main())
