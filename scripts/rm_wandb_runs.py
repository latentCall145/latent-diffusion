import wandb
api = wandb.Api()

PROJECT = 'tpu_ldm_ddpm_v2'
runs = api.runs(f'tiewa_enguin/{PROJECT}')

def should_delete(fnames, mod=None):
    if fnames == []:
        return []

    ret = []
    model_versions = []
    for fname, file in fnames:
        if not fname.endswith('pth'):
            model_versions.append(-1)
            continue
        model_version = int(fname[:-4].split('_')[-1])
        model_versions.append(model_version)

    max_model_version = max(model_versions)
    for (fname, file), model_version in zip(fnames, model_versions):
        if not fname.endswith('pth'):
            continue
        model_version = int(fname[:-4].split('_')[-1])
        if (mod is None or model_version % mod != 0) and model_version != max_model_version:
            ret.append((fname, file))
    return ret

for run in runs:
    if 'pt9' in run.name:
        #run.upload_file('../trained_models/ddpm_0102_fixed.pth')
        print(list(run.files()))



    #run_fnames_files = []
    #for f in run.files():
    #    run_fnames_files.append((f.name, f))
    #for fname, f in should_delete(run_fnames_files):
    #    print('Deleting', fname, 'from run', run)
    #    #f.delete()
