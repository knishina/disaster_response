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
- open a chrome browser, go to `http://0.0.0.0:5000/`

<br />

### Features.


### License.
This project is licensed under the MIT License - see the [LICENSE](https://github.com/knishina/diaster_response/blob/master/LICENSE) file for details.
