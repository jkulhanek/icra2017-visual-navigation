import deep_rl
from threading import Lock
import os
import json

default_configuration = dict(
    #visdom = dict(
    #    server = 'http://localhost',
    #    port = 8097
    #),
    
    house3d = dict(
        framework_path = '/House3D', # '/House3D',
        dataset_path = '~/datasets/suncg' # '/datasets/suncg'
    ),

    models_path = '~/models',
    videos_path = '~/results/videos'
)

basepath = os.path.expanduser('~/.visual_navigation')
os.makedirs(basepath, exist_ok=True)
configuration = dict(**default_configuration)
if not os.path.exists(os.path.join(basepath, 'config')):
    with open(os.path.join(basepath, 'config'), 'w+') as f:
        json.dump(configuration, f)

with open(os.path.join(basepath, 'config'), 'r') as f:
    configuration.update(**json.load(f))

def expand_user(d):
    if isinstance(d, dict):
        dnew = dict()
        for key, v in d.items():
            if key.endswith('_path') and isinstance(v, str) and v.startswith('~'):
                dnew[key] = os.path.expanduser(v)
            else:
                dnew[key] = expand_user(v)
        return dnew

    return d

configuration = expand_user(configuration)
deep_rl.configure(**configuration)
configuration = deep_rl.configuration

logger = deep_rl.common.metrics.MetricWriter(session_name='icra-tensorflow')
oldsave = logger.save
def save(path):
    print('saving metrics to %s' % path)
    oldsave(path)
    print('metrics saved')

logger.save = save
metrics_lock = Lock()

save_path = os.path.join(configuration.get('models_path'), 'icra-tensorflow')
os.makedirs(save_path, exist_ok=True)

def get_logger():
    class _LoggerProxy:
        def __enter__(self):
            metrics_lock.acquire()
            return logger

        def __exit__(self, *args, **kwargs):
            metrics_lock.release()
            return None
    return _LoggerProxy()