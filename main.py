from App import App as AppCustom

CSV_FILE_PATH = 'datasets/heart.csv'


def heart_disease_prediction_custom():
    app = AppCustom(CSV_FILE_PATH)
    app.standardize_data()
    app.fit()
    app.set_x_predictions()
    app.accuracy_score()

    input_data = (53, 1, 0, 140, 203, 1, 0, 155, 1, 3.1, 0, 0, 3)
    app.predict(input_data)


heart_disease_prediction_custom()

