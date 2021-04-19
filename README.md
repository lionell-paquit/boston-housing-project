# Boston House Price Prediction App

Clone the repo to use the app. Follow these steps to run the prediction app:

**Step 1: Create a virtual environment**  

From a terminal, go to the folder and create the virtual environment:  

```
$ python -m venv myvenv
``` 

**Step 2: Activate the environment and install the python packages**    

Ensure you have cloned this repository, then navigate to the root directory of the repository.  

```
$ source myvenv/bin/activate  
(myvenv) $ python -m pip install --upgrade pip setuptools wheel  
(myvenv) $ pip install -r requirements.txt
``` 

**Step 3: To run the app, change directory to code and run the streamlit** :

```
$ cd code/
$ streamlit run app.py
```
You'll see to URL - Network and External URL. `Ctlr+Click` the Network URL to open in browser.

**Step 4: Run unit test**

To run unit test change directory to tests and enter pytest command.

```
$ cd tests/
$ py.test --tb=line
```
