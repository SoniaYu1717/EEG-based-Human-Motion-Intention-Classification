from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import reader
import params


data_reader = reader.Reader('./project_datasets/')
features, labels = data_reader.get_mesh(params.BATCH_SIZE, 0, 'training')

print(features[0, 0, :, :])
