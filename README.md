# PRD-Forecasting-Server

# Create a virtual environment to isolate our package dependencies locally
python3 -m venv env
source env/bin/activate

# Install Flask
pip install -r requirements.txt

# Start PRD Flask Server
flask --app main run