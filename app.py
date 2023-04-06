from flask import Flask, render_template, request, url_for, redirect
from Music_Recommendation import recommend_songs

app = Flask(__name__)


@app.route('/')
def home():
    return render_template("Trial.html")

@app.route('/about')
def about():
    return render_template("Trail_Landing.html")

@app.route("/contact", methods=['POST'])
def contact():
        title = request.form['title']
        pred_args = [title]
        df = recommend_songs(pred_args)

        if df.dtype=="String":
            res = df
        else:
            res = df.to_string()
        return redirect(url_for("recommended.html", recommend=res))

@app.route("/recommend")
def recommend():
    return render_template("recommended.html")

if __name__ == '__main__':
    app.run(debug=True)

