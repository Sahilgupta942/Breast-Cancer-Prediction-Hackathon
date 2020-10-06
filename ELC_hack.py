from flask import Flask,render_template,session,url_for,redirect
import numpy as numpy
import joblib
from wtforms import TextField,SubmitField
from flask_wtf import FlaskForm


def return_prediction(model,scaler,sample_json):
	
	s_radius_mean=sample_json['radius_mean']
	s_perimeter_mean=sample_json['perimeter_mean']
	s_area_mean=sample_json['area_mean']
	s_concavity_mean=sample_json['concavity_mean']
	s_concave_points_mean=sample_json['concave_points_mean']
	s_radius_worst=sample_json['radius_worst']
	s_perimeter_worst=sample_json['perimeter_worst']
	s_area_worst=sample_json['area_worst']
	s_compactness_worst=sample_json['compactness_worst']
	s_concavity_worst=sample_json['concavity_worst']
	s_concave_points_worst=sample_json['concave_points_worst']
	s_compactness_mean=sample_json['compactness_mean']
	
	cancer=[[s_radius_mean,s_perimeter_mean,s_area_mean,s_concavity_mean,s_concave_points_mean,s_radius_worst,s_perimeter_worst,s_area_worst,s_compactness_worst,s_concavity_worst,s_concave_points_worst,s_compactness_mean]]
	
	cancer_class=['Benign','Malignant']
	
	cancer=scaler.transform(cancer)
	
	class_ind=model.predict(cancer)[0]
	
	return cancer_class[class_ind]


app=Flask(__name__)
app.config['SECRET_KEY']='mysecretkey'

class CancerForm(FlaskForm):

	radius_mean=TextField("radius_mean")
	perimeter_mean=TextField("perimeter_mean")
	area_mean=TextField("area_mean")
	concavity_mean=TextField("concavity_mean")
	concave_points_mean=TextField("concave points_mean")
	radius_worst=TextField("radius_worst")
	perimeter_worst=TextField("perimeter_worst")
	area_worst=TextField("area_worst")
	compactness_worst=TextField("compactness_worst")
	concavity_worst=TextField("concavity_worst")
	concave_points_worst=TextField("concave_points_worst")
	compactness_mean=TextField("compactness_mean")

	submit=SubmitField("Submit")


@app.route("/",methods=['GET','POST'])
def index():
	
	form=CancerForm()

	if form.validate_on_submit():

		session['radius_mean']=form.radius_mean.data
		session['perimeter_mean']=form.perimeter_mean.data
		session['area_mean']=form.area_mean.data
		session['concavity_mean']=form.concavity_mean.data
		session['concave_points_mean']=form.concave_points_mean.data
		session['radius_worst']=form.radius_worst.data
		session['perimeter_worst']=form.perimeter_worst.data
		session['area_worst']=form.area_worst.data
		session['compactness_worst']=form.compactness_worst.data
		session['concavity_worst']=form.concavity_worst.data
		session['concave_points_worst']=form.concave_points_worst.data
		session['compactness_mean']=form.compactness_mean.data

		return redirect(url_for("prediction"))
	return render_template('home_cancer.html',form=form)



elc_model1=joblib.load('elc_model1.sav')
elc_scaler1=joblib.load('cancer_scaler.pkl')

@app.route('/prediction')
def prediction():
	content={}

	content['radius_mean']=float(session['radius_mean'])
	content['perimeter_mean']=float(session['perimeter_mean'])
	content['area_mean']=float(session['area_mean'])
	content['concavity_mean']=float(session['concavity_mean'])
	content['concave_points_mean']=float(session['concave_points_mean'])
	content['radius_worst']=float(session['radius_worst'])
	content['perimeter_worst']=float(session['perimeter_worst'])
	content['area_worst']=float(session['area_worst'])
	content['compactness_worst']=float(session['compactness_worst'])
	content['concavity_worst']=float(session['concavity_worst'])
	content['concave_points_worst']=float(session['concave_points_worst'])
	content['compactness_mean']=float(session['compactness_mean'])

	results=return_prediction(elc_model1,elc_scaler1,content)

	return render_template('prediction_cancer.html',results=results)

if __name__=='__main__':
	app.run()