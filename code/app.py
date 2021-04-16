import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt

from make_dataset import DataPipeline
from predict_model import Predict

class App:
    def __init__(self):
        # Loads the Boston House Price Dataset
        boston = DataPipeline()
        
        self.X = boston.data.loc[:,boston.feature_names]
        self.y = boston.data['CMEDV']

    def user_input_features(self):
        LSTAT = st.sidebar.slider('LSTAT', self.X.LSTAT.min(), self.X.LSTAT.max(), float(self.X.LSTAT.mean()))
        PTRATIO = st.sidebar.slider('PTRATIO', self.X.PTRATIO.min(), self.X.PTRATIO.max(), float(self.X.PTRATIO.mean()))
        TAX = st.sidebar.slider('TAX', int(self.X.TAX.min()), int(self.X.TAX.max()), int(self.X.TAX.mean()))
        AGE = st.sidebar.slider('AGE', self.X.AGE.min(), self.X.AGE.max(), float(self.X.AGE.mean()))
        RM = st.sidebar.slider('RM', self.X.RM.min(), self.X.RM.max(), float(self.X.RM.mean()))
        NOX = st.sidebar.slider('NOX', self.X.NOX.min(), self.X.NOX.max(), float(self.X.NOX.mean()))
        INDUS = st.sidebar.slider('INDUS', self.X.INDUS.min(), self.X.INDUS.max(), float(self.X.INDUS.mean()))
        CRIM = st.sidebar.slider('CRIM', self.X.CRIM.min(), self.X.CRIM.max(), float(self.X.CRIM.mean()))

        data = {'LSTAT': LSTAT,
                'PTRATIO': PTRATIO,
                'TAX': TAX,
                'AGE': AGE,
                'RM': RM,
                'NOX': NOX,
                'INDUS': INDUS,
                'CRIM': CRIM
               }

        features = pd.DataFrame(data, index=[0])
        return features

    def run(self):
        st.write("""
        # Boston House Price Prediction App
        This app predicts the **Boston House Price**
        
        ### Format
        The data contains the following columns:

        * MEDV a numeric vector of corrected median values of owner-occupied housing in USD 1000
        * CRIM a numeric vector of per capita crime        
        * INDUS a numeric vector of proportions of non-retail business acres per town (constant for all Boston tracts)
        * NOX a numeric vector of nitric oxides concentration (parts per 10 million) per town
        * RM a numeric vector of average numbers of rooms per dwelling
        * AGE a numeric vector of proportions of owner-occupied units built prior to 1940
        * TAX a numeric vector full-value property-tax rate per USD 10,000 per town (constant for all Boston tracts)
        * PTRATIO a numeric vector of pupil-teacher ratios per town (constant for all Boston tracts)
        * LSTAT a numeric vector of percentage values of lower status population
        """)
        st.write('---')



        # Sidebar
        # Header of Specify Input Parameters
        st.sidebar.header('Specify Input Parameters')

        df = self.user_input_features()

        # Main Panel

        # Print specified input parameters
        st.header('Specified Input parameters')
        st.write(df)
        st.write('---')

        # Load model to predict new data
        # Build Regression Model
        predict = Predict()
        # Apply Model to Make Prediction
        prediction = predict.get_prediction(df)

        st.header('Prediction of MEDV')
        st.write(prediction)
        st.write('---')

        # Explaining the model's predictions using SHAP values
        # https://github.com/slundberg/shap
        explainer = shap.TreeExplainer(predict.model.best_estimator_)
        shap_values = explainer.shap_values(self.X)
        
        # To disable warning with st.pylot no argument
        st.set_option('deprecation.showPyplotGlobalUse', False)

        st.header('Feature Importance')
        plt.title('Feature importance based on SHAP values')
        shap.summary_plot(shap_values, self.X)
        st.pyplot(bbox_inches='tight')
        st.write('---')

        plt.title('Feature importance based on SHAP values (Bar)')
        shap.summary_plot(shap_values, self.X, plot_type="bar")
        st.pyplot(bbox_inches='tight')

if __name__ == "__main__":
    app = App()
    app.run()