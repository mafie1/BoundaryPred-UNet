import splitfolders


def split_dataset(path, output_path, ratio = (0.8, 0.1, 0.1), random_seed = 1337):
    print('Start Splitting Data in {} according to {} : {} : {} ratio into train, test, val sets'.format(path, *ratio) )
    splitfolders.ratio(input=path, output = output_path, seed = random_seed, ratio = ratio )
    print('Data Splitting Done')


if __name__ == '__main__':
    folder = '/Users/luisaneubauer/Documents/BA_Thesis/CVPPP2017_instances/training/'
    output_folder = '/Users/luisaneubauer/Documents/BA_Thesis/CVPPP2017_instances/output'

    split_dataset(folder, output_folder)