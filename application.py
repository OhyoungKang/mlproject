# mlproject/application.py

from flask import Flask, render_template, request

from src.logger import logging
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)
application=app

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=int(request.form.get('writing_score')),
            writing_score=int(request.form.get('reading_score'))
        )
        pred_df=data.get_data_as_dataframe()
        prediction_pipeline=PredictPipeline()
        results=prediction_pipeline.predict(pred_df)

        logging.info("Prediction Dataframe")
        logging.info(pred_df)
        logging.info("Prediction")
        logging.info(results)

        return render_template('home.html',results=f"{results[0]:.2f}")


if __name__=="__main__":
        app.run(debug=True)