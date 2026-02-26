use_amp = False

def set_amp(use_or_not):
    global use_amp
    use_amp = use_or_not

def get_amp():
    assert use_amp is not None, "CFG not initialized"
    return use_amp