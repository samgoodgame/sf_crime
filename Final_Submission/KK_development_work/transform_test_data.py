# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 11:25:06 2017

@author: kalvi
"""

#required imports
import pandas as pd
import numpy as np

import csv
import time
import calendar

def get_test_data(test_transformed_path, test_path, earlyWeatherDataPath, weatherData1, weatherData2):
    x_data = pd.read_csv(test_transformed_path, header=0)

    ########## Adding the date back into the data
    dataCSV = open(test_path, 'rt')
    csvData = list(csv.reader(dataCSV))
    csvFields = csvData[0] #['Dates', 'Category', 'Descript', 'DayOfWeek', 'PdDistrict', 'Resolution', 'Address', 'X', 'Y']
    allData = csvData[1:]
    dataCSV.close()

    df = pd.DataFrame(allData)
    df.columns = csvFields
    dates = df['Dates']
    dates = dates.apply(time.strptime, args=("%Y-%m-%d %H:%M:%S",))
    dates = dates.apply(calendar.timegm)

    x_data['secondsFromEpoch'] = dates
    colnames = x_data.columns.tolist()
    colnames = colnames[-1:] + colnames[:-1]
    x_data = x_data[colnames]
    ##########
    
    #functions for processing the sunrise and sunset times of each day
    def get_hour_and_minute(milTime):
        hour = int(milTime[:-2])
        minute = int(milTime[-2:])
        return [hour, minute]

    def get_date_only(date):
        return time.struct_time(tuple([date[0], date[1], date[2], 0, 0, 0, date[6], date[7], date[8]]))

    def structure_sun_time(timeSeries, dateSeries):
        sunTimes = timeSeries.copy()
        for index in range(len(dateSeries)):
            sunTimes[index] = time.struct_time(tuple([dateSeries[index][0], dateSeries[index][1], dateSeries[index][2], timeSeries[index][0], timeSeries[index][1], dateSeries[index][5], dateSeries[index][6], dateSeries[index][7], dateSeries[index][8]]))
        return sunTimes
    
    def get_weather_data(data_path):
        dataCSV = open(data_path, 'rt')
        csv_data = list(csv.reader(dataCSV))
        csv_fields = csv_data[0] #['Dates', 'Category', 'Descript', 'DayOfWeek', 'PdDistrict', 'Resolution', 'Address', 'X', 'Y']
        weather_data = csv_data[1:]
        dataCSV.close()

        weather_df = pd.DataFrame(weather_data)
        weather_df.columns = csv_fields
        dates = weather_df['DATE']
        sunrise = weather_df['DAILYSunrise']
        sunset = weather_df['DAILYSunset']

        dates = dates.apply(time.strptime, args=("%Y-%m-%d %H:%M",))
        sunrise = sunrise.apply(get_hour_and_minute)
        sunrise = structure_sun_time(sunrise, dates)
        sunrise = sunrise.apply(calendar.timegm)

        sunset = sunset.apply(get_hour_and_minute)
        sunset = structure_sun_time(sunset, dates)
        sunset = sunset.apply(calendar.timegm)
        dates = dates.apply(calendar.timegm)

        weather_df['DATE'] = dates
        weather_df['DAILYSunrise'] = sunrise
        weather_df['DAILYSunset'] = sunset

        return weather_df
    
    ########## Adding the weather data into the original crime data
    
    earlyWeatherDF = get_weather_data(earlyWeatherDataPath)
    weatherDF1 = get_weather_data(weatherData1)
    weatherDF2 = get_weather_data(weatherData2)
    
    weatherDF = pd.concat([earlyWeatherDF[450:975],weatherDF1,weatherDF2[32:]],ignore_index=True)
    
    # weather feature selection
    weatherMetrics = weatherDF[['DATE','HOURLYDRYBULBTEMPF','HOURLYRelativeHumidity', 'HOURLYWindSpeed', \
                                'HOURLYSeaLevelPressure', 'HOURLYVISIBILITY', 'DAILYSunrise', 'DAILYSunset']]
    weatherMetrics = weatherMetrics.convert_objects(convert_numeric=True)
    weatherDates = weatherMetrics['DATE']
    #'DATE','HOURLYDRYBULBTEMPF','HOURLYRelativeHumidity', 'HOURLYWindSpeed',
    #'HOURLYSeaLevelPressure', 'HOURLYVISIBILITY'
    timeWindow = 10800 #3 hours
    hourlyDryBulbTemp = []
    hourlyRelativeHumidity = []
    hourlyWindSpeed = []
    hourlySeaLevelPressure = []
    hourlyVisibility = []
    dailySunrise = []
    dailySunset = []
    daylight = []
    test = 0
    for timePoint in dates:#dates is the epoch time from the kaggle data
        relevantWeather = weatherMetrics[(weatherDates <= timePoint) & (weatherDates > timePoint - timeWindow)]
        hourlyDryBulbTemp.append(relevantWeather['HOURLYDRYBULBTEMPF'].mean())
        hourlyRelativeHumidity.append(relevantWeather['HOURLYRelativeHumidity'].mean())
        hourlyWindSpeed.append(relevantWeather['HOURLYWindSpeed'].mean())
        hourlySeaLevelPressure.append(relevantWeather['HOURLYSeaLevelPressure'].mean())
        hourlyVisibility.append(relevantWeather['HOURLYVISIBILITY'].mean())
        dailySunrise.append(relevantWeather['DAILYSunrise'].iloc[-1])
        dailySunset.append(relevantWeather['DAILYSunset'].iloc[-1])
        daylight.append(1.0*((timePoint >= relevantWeather['DAILYSunrise'].iloc[-1]) and (timePoint < relevantWeather['DAILYSunset'].iloc[-1])))

        if test%100000 == 0:
            print(relevantWeather)
        test += 1

    hourlyDryBulbTemp = pd.Series.from_array(np.array(hourlyDryBulbTemp))
    hourlyRelativeHumidity = pd.Series.from_array(np.array(hourlyRelativeHumidity))
    hourlyWindSpeed = pd.Series.from_array(np.array(hourlyWindSpeed))
    hourlySeaLevelPressure = pd.Series.from_array(np.array(hourlySeaLevelPressure))
    hourlyVisibility = pd.Series.from_array(np.array(hourlyVisibility))
    dailySunrise = pd.Series.from_array(np.array(dailySunrise))
    dailySunset = pd.Series.from_array(np.array(dailySunset))
    daylight = pd.Series.from_array(np.array(daylight))

    x_data['HOURLYDRYBULBTEMPF'] = hourlyDryBulbTemp
    x_data['HOURLYRelativeHumidity'] = hourlyRelativeHumidity
    x_data['HOURLYWindSpeed'] = hourlyWindSpeed
    x_data['HOURLYSeaLevelPressure'] = hourlySeaLevelPressure
    x_data['HOURLYVISIBILITY'] = hourlyVisibility
    #x_data['DAILYSunrise'] = dailySunrise
    #x_data['DAILYSunset'] = dailySunset
    x_data['Daylight'] = daylight
    x_data = x_data.drop('secondsFromEpoch', 1)
    x_data = x_data.drop('pd_bayview_binary', 1)
    
    return x_data

test_transformed_path = "./data/test_transformed.csv"
test_path = "./data/test.csv"
earlyWeatherDataPath = "./data/1049158.csv"
weatherData1 = "./data/1027175.csv"
weatherData2 = "./data/1027176.csv"

write_path = "C:/MIDS/W207 final project/test_data_with_weather.csv"

x_data = get_test_data(test_transformed_path, test_path, earlyWeatherDataPath, weatherData1, weatherData2)
x_data.to_csv(path_or_buf=write_path,index=0)