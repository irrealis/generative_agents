import os
from dotenv import load_dotenv

load_dotenv()

this_dir = os.path.abspath(os.path.dirname(__file__))
environment_loc = os.path.abspath(f"{this_dir}/../../environment/")

# Copy and paste your OpenAI API Key
openai_api_key = os.environ.get('OPENAI_API_KEY')
# Put your name
key_owner = os.environ.get('KEY_OWNER')


maze_assets_loc = f"{environment_loc}/frontend_server/static_dirs/assets"
env_matrix = f"{maze_assets_loc}/the_ville/matrix"
env_visuals = f"{maze_assets_loc}/the_ville/visuals"

fs_storage = f"{environment_loc}/frontend_server/storage"
fs_temp_storage = f"{environment_loc}/frontend_server/temp_storage"

collision_block_id = "32125"

# Verbose 
debug = True
