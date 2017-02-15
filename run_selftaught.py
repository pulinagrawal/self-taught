
# coding: utf-8

# In[ ]:

ae=autencoder([12,34,35])
ae.input_data(input_)
ae.sparse=True
ae.load_model()
ae.train()

