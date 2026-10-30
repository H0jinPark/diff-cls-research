import importlib.util
import json
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location('claude_baseline', Path(__file__).parents[1]/'scripts/claude_baseline.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def test_prompt_does_not_include_evaluation_label():
    row = {'child':'plastic pallet', 'parent':'pallet', 'label':'SENSITIVE_GOLD'}
    examples = [{'id':'train-1','compound':'coffee cup','head':'cup','label':'purpose'}]
    text = b.prompt(row, ['purpose','attribute'], examples)
    assert 'SENSITIVE_GOLD' not in text
    assert 'train-1' not in text
    assert 'coffee cup' in text
    assert b.parse_label('purpose\n', ['purpose','attribute']) == 'purpose'
    assert b.parse_label('purpose or attribute', ['purpose','attribute']) is None


def test_end_to_end_resume_and_invalid(monkeypatch, tmp_path):
    repo = Path(__file__).parents[1]
    output = tmp_path/'out'
    from diffcls.data import read_rows, split_rows, write_rows
    split = split_rows(read_rows(repo/'data/demo.jsonl'), mode='family', seed=13)
    for name in ('train', 'dev'):
        write_rows(tmp_path/'splits'/f'{name}.jsonl', split[name])
    requests = []
    def fake_call(key, model, text, max_tokens):
        requests.append(text)
        answer = 'not-a-label' if len(requests) == 1 else 'material'
        return {'model':model, 'content':[{'type':'text','text':answer}],
                'usage':{'input_tokens':10,'output_tokens':2}}
    monkeypatch.setenv('ANTHROPIC_API_KEY','fake-key')
    monkeypatch.setattr(b, 'call_api', fake_call)
    monkeypatch.setattr(sys, 'argv', ['baseline','--splits',str(tmp_path/'splits'),
        '--env-file', str(tmp_path/'none'), '--output',str(output),'--limit','3','--run'])
    b.main()
    summary = json.loads((output/'summary.json').read_text())
    assert len(requests) == 3
    assert summary['invalid_predictions'] == 1
    assert summary['count'] == 3
    assert summary['token_usage']['input_tokens'] == 30
    monkeypatch.setattr(sys, 'argv', sys.argv+['--resume'])
    b.main()
    assert len(requests) == 3
