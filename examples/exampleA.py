from fgem.world import World
from fgem.utils.config import get_config_from_json
from tqdm import tqdm

# Read configuration file
config = get_config_from_json('configs/exampleA.json')

project = World(config)

# Iterate over project lifetime
for i in tqdm(range(project.max_simulation_steps)):
    project.step_update_record()

# Compute economics and summary results
project.compute_economics()
