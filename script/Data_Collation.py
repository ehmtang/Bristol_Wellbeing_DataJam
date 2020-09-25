#File containing data collation functions
#Created on 25th Jan 2020 14:20:10
#Author: Daniel Rodrigues

#Import relevant packages
import pandas as pd
import numpy as np
import datetime

#FUNCTIONS
def reformat(date, delimiter = "/", join = "_", order = [2,0,1]):
    date = date.split(delimiter)
    
    return join.join([date[order[0]], date[order[1]], date[order[2]]])

def dat(date, delimiter = "_", order = [0,1,2]):
    date = date.split(delimiter)
    return datetime.datetime(int(date[order[0]]), int(date[order[1]]), int(date[order[2]]))


#CONSTANTS
day_dict = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}


def collate(df):
    #PREPROCESSING
    #Load data
    #df = pd.read_csv("Staff absence.csv")

    #Drop irrelevant data
    dropcols = ["FTE Days Lost", "Calendar Days Lost", "Total FTE Calendar Days ", "First Day Absent"]
    df.drop(dropcols, inplace = True, axis = 1)
    df.drop("FTE", inplace = True, axis = 1) #Perhaps first run a function to split into different DF's for different FTE ranges...

    #Delete invalid data
    df = df[df["Absence End Date"] != "12/31/4712"] #hardcoded for now, could include timestamp

    #Reformat dates to desired format
    dates = df.applymap(reformat)
    dates.sort_values("Absence Start Date", inplace = True, ascending = True)

    #Extract start and end dates, and use to generate dates in between (in string and datetime format)
    start = dates["Absence Start Date"].tolist()[0]
    end = dates["Absence Start Date"].tolist()[-1]
    dr = pd.date_range(dat(start), dat(end)).strftime("%Y_%m_%d")
    _dr = pd.date_range(dat(start), dat(end))



    absences = [0] * len(dr)

    #Combines data across all days
    for index, row in dates.iterrows():
        for ind, day in enumerate(dr):
            if row["Absence Start Date"] <= day and row["Absence End Date"] >= day:
                absences[ind] += 1


    #text = True
    text = False

    #Adds day of week to extra column
    days = _dr.tolist()
    for index, day in enumerate(days):
        if text:
            days[index] = day_dict[days[index].weekday()]
        else:
            days[index] = days[index].weekday()

    df4 = pd.DataFrame([dr, days, absences]).T
    df4.columns = ["Date", "Day", "Absences"]

    #"""
    ms = pd.date_range(dat(start), dat(end)).strftime("%m")
    ds = pd.date_range(dat(start), dat(end)).strftime("%d")
    df4 = pd.DataFrame([dr, ms, ds, days, absences]).T
    df4.columns = ["Date", "Month", "Day", "Weekday", "Absences"]#isweekend
    #"""
    #print(df4.head())

    df4.to_csv("Date_Day_Absence.csv", index = False)

    return df4



if __name__ == "__main__":
    df = pd.read_csv("Staff absence.csv")
    df_collated = collate(df)
    print(df_collated.head())

