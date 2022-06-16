import os
import os.path as op
import shutil

base = 'tensorboard/ExtraAdam'
basecp = 'tensorboard2/ExtraAdam'

for f in os.listdir(base):
    fr = f.replace("-", "_").replace(":", "_")
    if not op.isdir(op.join(basecp, fr)):
        os.makedirs(op.join(basecp, fr))
    for file in os.listdir(op.join(base,f)):
        shutil.copy(
            op.join(base, f, file),
            op.join(basecp, fr))
    print(fr)