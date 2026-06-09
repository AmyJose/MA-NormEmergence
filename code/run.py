from harvest_model import HarvestModel

model = HarvestModel(seed=None)

for i in range(10):
    model.step()
    print(f"Completed step {i}")

