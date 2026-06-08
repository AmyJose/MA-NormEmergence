from harvest_model import HarvestModel

model = HarvestModel(seed=None)

for _ in range(500):
    model.step()

model.write_emerged_norms()