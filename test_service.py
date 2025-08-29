from service import handle_asr_task

def test_handle_asr_task(monkeypatch):
    result = handle_asr_task('https://media.xyzcdn.net/65cef9e3cace72dff8d98de3/lnxOGQodYycgLLwFllTjRH7yzwHM.m4a')
    print(result)
    assert isinstance(result, list)
    assert len(result) == 7
    assert result == ["audio-clip-0.mp3", "audio-clip-1.mp3"]