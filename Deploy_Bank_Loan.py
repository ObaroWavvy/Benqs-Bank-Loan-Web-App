import pickle
import sklearn
import numpy as np
import streamlit as st
#To Display Images
from PIL import Image

#loading the saved model
loaded_model = pickle.load(open('trained_model_loan.sav', 'rb'))

def loan_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'The person will take a loan'
    else:
        return 'The person will not take a loan'

def main():
    # display image
    img = Image.open("bankloan.jpg")
    new_image = img.resize((700, 200))
    st.image(new_image)
    # let's display
    # st.image(img, width=700)

    # giving a title
    st.title('Benqs Bank Personal Loan Prediction Web App')

    # getting the input data from the user


    Age = st.number_input('Age: Input a number')
    Experience = st.number_input('Experience: Input a number')
    Income = st.number_input('Income: Input a number')
    Family = st.number_input('Family Size: Should be 0 to 4')
    Education = st.number_input('Education Level: Input a number: Should be 1 - 3')
    Mortgage = st.number_input('Mortgage: Input a number')
    Securities_Account = st.number_input('Security Account: Input a number: 0 - No Security Account or 1 - Security Account')
    CD_Account = st.number_input('CD_Account: Input a number: 0 - No Credit Account or 1 - Credit Account')
    CreditCard = st.number_input('CreditCard: Input a number: 0 - No Credit Card or 1 - Credit Card')

    # code for Prediction
    Loan = ''

    # creating a button for Prediction

    if st.button('Loan Prediction Test Result'):
        Loan = loan_prediction([Age, Experience, Income, Family, Education, Mortgage, Securities_Account, CD_Account, CreditCard])

    st.success(Loan)


if __name__ == '__main__':
    main()