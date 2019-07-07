from src.loaders import DataLoader
import matplotlib.pyplot as plt

loader = DataLoader()

datasets_dict = {
    'ERN': loader.get_ern,
    'SMR': lambda validation=False, subject=2: loader.get_smr(subject, validation),     # noqa
    'BMNIST': loader.get_bmnist11,
    # 'BMNIST_2': loader.get_bmnist2,
    'SEED': loader.get_seed,
    # 'ThoughtViz': loader.get_thoughtviz,
}

datasets = [[k, datasets_dict[k]] for k in datasets_dict]

k = 'A'

for dname, ldr in datasets:
    fig, axes = plt.subplots(1, 1)
    p = ldr(validation=True)
    data = p['data'][0][-10,:,0].reshape(-1,)
    loader.topoplot(p['name'], data)
    axes.set_title('({}) Topoplot for {}'.format(k, p['name']))
    fig.savefig('./topoplot_new/{}_topoplot_new.png'.format(dname))
    k = chr(ord(k)+1)
    print('Done {}, {}'.format(dname, p['name']))
