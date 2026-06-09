from harvest_model import HarvestModel

model = HarvestModel(seed=None)

for i in range(100):
    model.step()
    print(f"Completed step {i}")

