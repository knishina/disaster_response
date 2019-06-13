# disaster_response

### Summary.
This project utilizes data provided by [Figure Eight](https://www.figure-eight.com/). The dataset contains tweets and their designated labels from a real-life disaster.  The purpose of the project is to build an NLP (Natural Language Processing) tool that can accurate categorize tweets.

There are three major sections to the parts project.  First is an ETL pipeline that extracts, cleans, and loads the data into a database.  Second is a ML pipeline that utilizes the extracted data to train, predict, and classify the given text into discernable categories via machine learning models.  Finally, the third is a web application that demonstrates the model's efficacy in real time using the user's input.

<br />

### Dependencies.
- Python 3.6.7
- Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
- Natural Language Process Libraries: NLTK
- SQLlite Database Libraqries: SQLalchemy
- Web App and Data Visualization: Flask, Plotly

<br />

### Running the Application.
- clone the repository.<br>
    `git clone https://github.com/knishina/disaster_response.git`
- cd into the folder.
- run the application<br>
    `python app.py`
- open a chrome browser, go to `http://0.0.0.0:5000/` &nbsp; &nbsp; (Note: there may be a slight delay in the contents displaying.  If the content fails to display, reload the page.)

<br />

### Features.
This page is an interactive one where the user can enter a phrase or the text of a tweet. Once the text has been entered, click on the green button. In this example, the message for analysis is `earthquakes and severe rain`.
![top_page](https://github.com/knishina/disaster_response/blob/master/Images/01.png)
<br>

<br>

Once the button has been engaged, the querried text will be displayed.
![button](https://github.com/knishina/disaster_response/blob/master/Images/02.png)
<br>

<br>

The text is then categorized according to a set of pre-set labels.  If the text pertains to any of the pre-set labels, they will turn green. If the text is not associated with the pre-set labels, the labels will remain grey.
<p align="center">
    <img src="https://github.com/knishina/disaster_response/blob/master/Images/03.png" width=50% alt="pre-set labels">
</p>
<br>

<br>

Finally, there are two visualizations of the data that of which the model was trained.  The first visualization is a pie chart that takes a look at the three different genres.  The second visualization is a bar chart that looks at the distribution of data per pre-set label.  Both visualizations were produced using `plotly.js`.
<p align="center">
    <img src="https://github.com/knishina/disaster_response/blob/master/Images/04.png" width=40% alt="circle_chart">
    <img src="https://github.com/knishina/disaster_response/blob/master/Images/05.png" width=40% alt="bar_chart">
</p>
<br>

### License.
This project is licensed under the MIT License - see the [LICENSE](https://github.com/knishina/diaster_response/blob/master/LICENSE) file for details.
