import museval
import musdb
import os


musdb_path = os.path.join('..', 'Datasets', 'MUSDB18_eval')
estimates_path = os.path.join('..', 'Datasets', 'MUSDB18_predict_normed')
output_path = os.path.join('..', 'Results', 'MUSDB18_predict_normed')
csv = estimates_path.split(os.path.sep)[-1]
print(csv)

results = museval.EvalStore(frames_agg='median', tracks_agg='median')
for j, folder in enumerate(sorted(os.listdir(musdb_path))):
    if not os.path.exists(os.path.join(output_path, folder)):
        os.mkdir(os.path.join(output_path, folder))
    for i, song in enumerate(sorted(os.listdir(os.path.join(estimates_path, folder)))):
        # if i == 1: break
        print(i, song)
        if not os.path.exists(os.path.join(output_path, folder, song)):
            os.mkdir(os.path.join(output_path, folder, song))

        score = museval.eval_dir(reference_dir=os.path.join(musdb_path, folder, song),
                                 estimates_dir=os.path.join(estimates_path, folder, song),
                                 output_dir=os.path.join(output_path, folder, song))
        results.add_track(score)

print(results)
print('----------------------------------------------------------------------')
print(results.df)
results.df.to_csv(os.path.join('..', 'Results', csv + '.csv'))