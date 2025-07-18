import os

from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv(verbose=True, override=True)

# Iterate over environment variables and set them as module attributes
for key, value in os.environ.items():
    if key in os.environ:
        globals()[key] = value
