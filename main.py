import muller

ds = muller.dataset("test_dataset")
ds.create_tensor('labels', htype='generic', dtype='int')
ds.labels.extend([0, 1, 1, 2, 3])
ds.summary()
