from flask import Flask, request, render_template
from ibm_main import predict_sentiment  

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    good_feedback = " Thank you for your positive feedback! We're delighted to hear that you enjoyed the food "
    bad_feedback = "We apologize for any inconvenience you experienced. Your feedback is important to us, and we'll take it into consideration to improve our services "
    sentiment = "Neutral"  
    scent="good"
    if request.method == 'POST':
        user_input = request.form['user_input']
        sentiment_result = predict_sentiment(user_input)
        if sentiment_result == [0]:
            sentiment = bad_feedback
            scent="bad"
        else:
            sentiment = good_feedback
            scent="good"
    
    return render_template('index.html', sentiment=sentiment, scent=scent)

if __name__ == '__main__':
    app.run(debug=True)
