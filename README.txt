1. Environment installation
Run the next commands in command line:
sudo pip install virtualenv
cd /path/to/folder/Waste-classification
virtualenv venv --python=python3.8
source venv/bin/activate
pip install -r requirements.txt

2. To see all possible options run:
python main.py -h

3. Run code with options your need:
python main.py

4. Outputs:

If you run code in train mode:
- graph with the results of epoch validation at ./results/
- the best model is saved at ./results/
- accuracy of the prediction for test dataset

If you run code in default_test mode:
- the prediction is held by the best model saved at ./results/
- accuracy of the prediction for test dataset

If you ran code in own_test mode:
- make sure that the images you want to test are in example folder
- the prediction is held by the best model saved at ./results/
- the prediction results are saved at ./results/