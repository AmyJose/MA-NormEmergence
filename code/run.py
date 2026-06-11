from harvest_model import HarvestModel

model = HarvestModel(rng=42)

for i in range(2):
    model.step()
    print(f"Completed step {i}")

