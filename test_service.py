from service import handle_asr_task

def test_handle_asr_task(monkeypatch):
    result = handle_asr_task('https://media.xyzcdn.net/68a543be8c590c796c4e01bb/FujLApufGgocpqi2YU6Zuq9dckkL.m4a', 1, 600)
    print(result)
