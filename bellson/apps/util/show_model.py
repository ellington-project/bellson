#!/usr/bin/env python3
from ...libbellson import model as tmodel

model = tmodel.gen_latest_model()

print(model.summary())
