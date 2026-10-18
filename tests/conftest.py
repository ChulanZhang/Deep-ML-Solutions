import importlib.util, os
from pathlib import Path
import pytest, torch
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(20261015)
@pytest.fixture
def load():
    def get(number, interview=False):
        folder=(Path(os.environ.get('INTERVIEW_ROOT',str(ROOT)))/'interview_core'
                if interview else ROOT/'pytorch')
        matches=list(folder.glob(f'{number:02d}_*.py' if interview else f'{number:04d}_*.py'))
        assert len(matches)==1, (number,matches)
        spec=importlib.util.spec_from_file_location(f'test_module_{number}_{interview}',matches[0])
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module
    return get
