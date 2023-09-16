import wandb
api = wandb.Api()

PROJECT = 'tpu_ldm_ddpm_v2'
runs = api.runs(f'tiewa_enguin/{PROJECT}')
MOD = None
KEEP_MAX = True
print(f'MOD: {MOD}, KEEP_MAX: {KEEP_MAX}')
if not KEEP_MAX:
    print('\x1b[1;33mKEEP_MAX is set to False. Are you sure you want to delete ALL models?\x1b[0m')
input('Press enter to confirm: ')

def should_delete(fnames, mod=MOD, keep_max=KEEP_MAX):
    if fnames == []:
        return []

    ret = []
    model_versions = []
    for fname, file in fnames:
        if not fname.endswith('pth'):
            model_versions.append(-1)
            continue
        try:
            model_version = int(fname[:-4].split('_')[-1])
        except:
            pass
        model_versions.append(model_version)

    max_model_version = max(model_versions)
    for (fname, file), model_version in zip(fnames, model_versions):
        if not fname.endswith('pth'):
            continue
        try:
            model_version = int(fname[:-4].split('_')[-1])
        except:
            pass
        is_mod = (mod is not None) and (model_version % mod == 0)
        is_max = (model_version == max_model_version) and keep_max
        if is_mod:
            print(f'Keeping {fname} because mod={mod} divides version')
        if is_max:
            print(f'Keeping {fname} because max version for run')
        if not (is_mod or is_max):
            ret.append((fname, file))
    return ret

for run in runs:
    run_fnames_files = []
    for f in run.files():
        run_fnames_files.append((f.name, f))
    for fname, f in should_delete(run_fnames_files):
        print('Deleting', fname, 'from run', run)
        f.delete()
