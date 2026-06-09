from harvest_model import HarvestModel

model = HarvestModel(seed=None)

for _ in range(100):
    model.step()

